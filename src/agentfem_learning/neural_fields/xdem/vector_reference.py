"""Independent mixed-mode vector-elastic XDEM reference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import pi
from pathlib import Path

import numpy as np
import torch
from agentfem import fracture

from .reference import (
    ReferenceTrainingOptions,
    _crack_trace_grid,
    _evaluation_grid,
    _integration_rule,
    _resolve_device,
    _sample_annulus,
    _sample_boundaries,
)


class WilliamsVectorNetwork(torch.nn.Module):
    """Regular vector MLP plus differentiable Mode-I/II Williams bases."""

    def __init__(
        self,
        *,
        radius: float,
        tip_core_radius: float,
        young_modulus: float,
        poisson_ratio: float,
        assumption: str,
        reference_k_i: float,
        reference_k_ii: float,
        hidden_layers: tuple[int, ...] = (24, 24),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.tip_core_radius = float(tip_core_radius)
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.assumption = str(assumption)
        widths = (6, *(int(item) for item in hidden_layers), 2)
        layers: list[torch.nn.Module] = []
        for input_width, output_width in pairwise(widths):
            linear = torch.nn.Linear(input_width, output_width, dtype=dtype)
            torch.nn.init.xavier_uniform_(linear.weight, gain=0.08)
            torch.nn.init.zeros_(linear.bias)
            layers.append(linear)
            if output_width != 2:
                layers.append(torch.nn.Tanh())
        self.regular = torch.nn.Sequential(*layers)
        self.tip_amplitudes = torch.nn.Parameter(
            torch.tensor(
                (0.5 * float(reference_k_i), 0.5 * float(reference_k_ii)),
                dtype=dtype,
            )
        )

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        radial = torch.sqrt(torch.clamp(x.square() + y.square(), min=1.0e-24))
        angle = torch.atan2(y, x)
        return torch.cat(
            (
                x / self.radius,
                y / self.radius,
                radial / self.radius,
                torch.sin(0.5 * angle),
                torch.cos(0.5 * angle),
                torch.cos(angle),
            ),
            dim=1,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        radial = torch.linalg.vector_norm(coordinates, dim=1, keepdim=True)
        span = self.radius - self.tip_core_radius
        envelope = (
            (radial - self.tip_core_radius)
            * (self.radius - radial)
            / (span * span)
        )
        enriched = williams_vector_field(
            coordinates,
            young_modulus=self.young_modulus,
            poisson_ratio=self.poisson_ratio,
            assumption=self.assumption,
            k_i=self.tip_amplitudes[0],
            k_ii=self.tip_amplitudes[1],
        )
        return enriched + envelope * self.regular(self.features(coordinates))


@dataclass(frozen=True)
class VectorTrainingOutcome:
    """Vector field, optimization history, and common LEFM evidence."""

    model: WilliamsVectorNetwork
    epochs: np.ndarray
    losses: np.ndarray
    coordinates: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray
    stress: np.ndarray
    crack_trace_coordinates: np.ndarray
    crack_trace_side: np.ndarray
    crack_trace_prediction: np.ndarray
    crack_trace_reference: np.ndarray
    metrics: Mapping[str, float]
    stress_intensity: object
    device: str
    dtype: str

    def write_field(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            coordinates=self.coordinates,
            displacement=self.prediction,
            reference_displacement=self.reference,
            stress=self.stress,
            crack_trace_coordinates=self.crack_trace_coordinates,
            crack_trace_side=self.crack_trace_side,
            crack_trace_displacement=self.crack_trace_prediction,
            crack_trace_reference=self.crack_trace_reference,
        )
        return output


def williams_vector_field(
    coordinates: torch.Tensor,
    *,
    young_modulus: float,
    poisson_ratio: float,
    assumption: str,
    k_i,
    k_ii,
) -> torch.Tensor:
    """Return the differentiable leading mixed-mode Williams displacement."""

    ratio = float(poisson_ratio)
    selected = str(assumption)
    if selected not in {"plane_stress", "plane_strain"}:
        raise ValueError("assumption must be 'plane_stress' or 'plane_strain'.")
    shear = float(young_modulus) / (2.0 * (1.0 + ratio))
    kappa = 3.0 - 4.0 * ratio
    if selected == "plane_stress":
        kappa = (3.0 - ratio) / (1.0 + ratio)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    radial = torch.sqrt(torch.clamp(x.square() + y.square(), min=1.0e-24))
    angle = torch.atan2(y, x)
    half = 0.5 * angle
    cosine = torch.cos(half)
    sine = torch.sin(half)
    full_cosine = torch.cos(angle)
    mode_i = torch.as_tensor(k_i, dtype=coordinates.dtype, device=coordinates.device)
    mode_ii = torch.as_tensor(k_ii, dtype=coordinates.dtype, device=coordinates.device)
    first = (
        mode_i * cosine * (kappa - full_cosine)
        + mode_ii * sine * (kappa + 2.0 + full_cosine)
    )
    second = (
        mode_i * sine * (kappa - full_cosine)
        - mode_ii * cosine * (kappa - 2.0 + full_cosine)
    )
    scale = torch.sqrt(radial / (2.0 * pi)) / (2.0 * shear)
    return scale[:, None] * torch.stack((first, second), dim=1)


def train_vector_reference(
    spec,
    options: ReferenceTrainingOptions,
) -> VectorTrainingOutcome:
    """Optimize the declared vector Williams problem and extract SIF evidence."""

    parameters = _vector_parameters(spec)
    radius = parameters["radius"]
    core = parameters["tip_core_radius"]
    domain_count, boundary_count = _sample_counts(spec)
    dtype = torch.float64 if options.dtype == "float64" else torch.float32
    device = _resolve_device(options.device, dtype=options.dtype)
    torch.manual_seed(int(options.seed))
    model = WilliamsVectorNetwork(
        radius=radius,
        tip_core_radius=core,
        young_modulus=parameters["young_modulus"],
        poisson_ratio=parameters["poisson_ratio"],
        assumption=parameters["assumption"],
        reference_k_i=parameters["K_I"],
        reference_k_ii=parameters["K_II"],
        hidden_layers=options.hidden_layers,
        dtype=dtype,
    ).to(device)
    domain = _sample_annulus(
        domain_count,
        radius=radius,
        core=core,
        seed=options.seed,
        dtype=dtype,
        device=device,
    )
    boundary = _sample_boundaries(
        boundary_count,
        radius=radius,
        core=core,
        dtype=dtype,
        device=device,
    )
    boundary_target = _reference_field(boundary, parameters).detach()
    area = pi * (radius * radius - core * core)
    refinement_rule = _integration_rule(spec, "refinement")
    reference_points = _evaluation_grid(
        count=refinement_rule.count,
        seed=refinement_rule.seed,
        radius=radius,
        core=core,
        dtype=dtype,
        device=device,
    )
    _, _, reference_density = _elastic_state(
        _reference_field(reference_points, parameters),
        reference_points,
        parameters,
        create_graph=False,
    )
    reference_energy = float((area * reference_density.mean()).detach().cpu())
    displacement_scale = max(
        float(boundary_target.square().mean().detach().cpu()), 1.0e-24
    )

    def objective():
        prediction = model(domain)
        _, _, density = _elastic_state(
            prediction, domain, parameters, create_graph=True
        )
        internal_energy = area * density.mean()
        boundary_error = (model(boundary) - boundary_target).square().mean()
        loss = (
            internal_energy / reference_energy
            + options.boundary_penalty * boundary_error / displacement_scale
        )
        return loss, internal_energy, boundary_error

    epochs: list[float] = []
    losses: list[float] = []
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    for epoch in range(1, options.adam_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        optimizer.step()
        epochs.append(float(epoch))
        losses.append(float(loss.detach().cpu()))
        if options.progress and (epoch == 1 or epoch % 100 == 0):
            print(
                f"[agentfem-learning.xdem] vector Adam {epoch}/"
                f"{options.adam_epochs}: {losses[-1]:.6e}"
            )

    if options.lbfgs_steps:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=12,
            history_size=30,
            line_search_fn="strong_wolfe",
        )
        for step in range(1, options.lbfgs_steps + 1):

            def closure():
                optimizer_lbfgs.zero_grad(set_to_none=True)
                selected, _, _ = objective()
                selected.backward()
                return selected

            optimizer_lbfgs.step(closure)
            selected, _, _ = objective()
            epochs.append(float(options.adam_epochs + step))
            losses.append(float(selected.detach().cpu()))

    evaluation = _evaluation_grid(
        count=_integration_rule(spec, "validation").count,
        seed=_integration_rule(spec, "validation").seed,
        radius=radius,
        core=core,
        dtype=dtype,
        device=device,
    )
    prediction = model(evaluation)
    reference = _reference_field(evaluation, parameters)
    _, stress, density = _elastic_state(
        prediction, evaluation, parameters, create_graph=False
    )
    refined_prediction = model(reference_points)
    _, _, refined_density = _elastic_state(
        refined_prediction, reference_points, parameters, create_graph=False
    )
    training_energy = float(objective()[1].detach().cpu())
    predicted_energy = float((area * density.mean()).detach().cpu())
    refined_energy = float((area * refined_density.mean()).detach().cpu())
    report = _stress_intensity_report(model, parameters, dtype=dtype, device=device)
    trace_coordinates, trace_side = _crack_trace_grid(
        radius=radius,
        core=core,
        count=max(32, boundary_count // 2),
        dtype=dtype,
        device=device,
    )
    trace_prediction = model(trace_coordinates)
    trace_reference = _reference_field(trace_coordinates, parameters)
    metrics = _vector_metrics(
        prediction=prediction,
        reference=reference,
        boundary_prediction=model(boundary),
        boundary_reference=boundary_target,
        trace_prediction=trace_prediction,
        trace_reference=trace_reference,
        report=report,
        parameters=parameters,
        training_energy=training_energy,
        predicted_energy=predicted_energy,
        refined_energy=refined_energy,
        reference_energy=reference_energy,
    )
    return VectorTrainingOutcome(
        model=model,
        epochs=np.asarray(epochs, dtype=float),
        losses=np.asarray(losses, dtype=float),
        coordinates=evaluation.detach().cpu().numpy(),
        prediction=prediction.detach().cpu().numpy(),
        reference=reference.detach().cpu().numpy(),
        stress=stress.detach().cpu().numpy(),
        crack_trace_coordinates=trace_coordinates.detach().cpu().numpy(),
        crack_trace_side=trace_side.detach().cpu().numpy(),
        crack_trace_prediction=trace_prediction.detach().cpu().numpy(),
        crack_trace_reference=trace_reference.detach().cpu().numpy(),
        metrics=metrics,
        stress_intensity=report,
        device=str(device),
        dtype=options.dtype,
    )


def _reference_field(coordinates, parameters):
    return williams_vector_field(
        coordinates,
        young_modulus=parameters["young_modulus"],
        poisson_ratio=parameters["poisson_ratio"],
        assumption=parameters["assumption"],
        k_i=parameters["K_I"],
        k_ii=parameters["K_II"],
    )


def _displacement_gradient(displacement, coordinates, *, create_graph):
    rows = []
    for component in range(2):
        rows.append(
            torch.autograd.grad(
                displacement[:, component],
                coordinates,
                grad_outputs=torch.ones_like(displacement[:, component]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        )
    return torch.stack(rows, dim=1)


def _elastic_state(displacement, coordinates, parameters, *, create_graph):
    gradient = _displacement_gradient(
        displacement, coordinates, create_graph=create_graph
    )
    strain = 0.5 * (gradient + gradient.transpose(1, 2))
    modulus = parameters["young_modulus"]
    ratio = parameters["poisson_ratio"]
    shear = modulus / (2.0 * (1.0 + ratio))
    lame = 2.0 * shear * ratio / (1.0 - 2.0 * ratio)
    if parameters["assumption"] == "plane_stress":
        lame = 2.0 * shear * ratio / (1.0 - ratio)
    trace = strain[:, 0, 0] + strain[:, 1, 1]
    identity = torch.eye(2, dtype=coordinates.dtype, device=coordinates.device)
    stress = 2.0 * shear * strain + lame * trace[:, None, None] * identity
    density = shear * strain.square().sum(dim=(1, 2)) + 0.5 * lame * trace.square()
    return gradient, stress, density


def _stress_intensity_report(model, parameters, *, dtype, device):
    radius = parameters["radius"]
    core = parameters["tip_core_radius"]
    cracks = fracture.crack_set(
        fracture.segment("branch_cut", start=(-radius, 0.0), end=(0.0, 0.0))
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=parameters["young_modulus"],
        poisson_ratio=parameters["poisson_ratio"],
        assumption=parameters["assumption"],
    )
    tip = cracks.tip("branch_cut:end")
    auxiliary_i = fracture.WilliamsField2D(tip, material, k_i=1.0)
    auxiliary_ii = fracture.WilliamsField2D(tip, material, k_ii=1.0)
    radii = (0.25 * radius, 0.4 * radius, 0.6 * radius)
    mode_i = []
    mode_ii = []
    for outer in radii:
        inner = max(1.05 * core, 0.3 * outer)
        radial_count = 24
        angular_count = 96
        radial_fraction = (
            torch.arange(radial_count, dtype=dtype, device=device) + 0.5
        ) / radial_count
        radial = torch.sqrt(
            inner * inner + radial_fraction * (outer * outer - inner * inner)
        )
        angle = -pi + (
            torch.arange(angular_count, dtype=dtype, device=device) + 0.5
        ) * (2.0 * pi / angular_count)
        rr, tt = torch.meshgrid(radial, angle, indexing="ij")
        coordinates = torch.stack((rr * torch.cos(tt), rr * torch.sin(tt)), dim=-1)
        coordinates = coordinates.reshape(-1, 2).requires_grad_(True)
        prediction = model(coordinates)
        gradient, stress, _ = _elastic_state(
            prediction, coordinates, parameters, create_graph=False
        )
        points = coordinates.detach().cpu().numpy()
        distance = np.linalg.norm(points, axis=1)
        q_gradient = -points / distance[:, None] / (outer - inner)
        weights = np.full(
            len(points), pi * (outer * outer - inner * inner) / len(points)
        )

        actual_stress = stress.detach().cpu().numpy()
        actual_gradient = gradient.detach().cpu().numpy()
        for auxiliary, collected in (
            (auxiliary_i, mode_i),
            (auxiliary_ii, mode_ii),
        ):
            samples = fracture.InteractionIntegralSamples2D(
                actual_stress=actual_stress,
                actual_displacement_gradient=actual_gradient,
                auxiliary_stress=auxiliary.stress(points),
                auxiliary_displacement_gradient=auxiliary.displacement_gradient(
                    points
                ),
                q_gradient=q_gradient,
                weights=weights,
            )
            collected.append(fracture.interaction_integral(samples))
    return fracture.interaction_integral_report(
        crack=cracks,
        tip_id="branch_cut:end",
        integration_radii=radii,
        mode_i_integrals=mode_i,
        mode_ii_integrals=mode_ii,
        material=material,
        relative_path_tolerance=0.08,
        metadata={"provider": "agentfem-learning.xdem"},
    )


def _vector_metrics(
    *,
    prediction,
    reference,
    boundary_prediction,
    boundary_reference,
    trace_prediction,
    trace_reference,
    report,
    parameters,
    training_energy,
    predicted_energy,
    refined_energy,
    reference_energy,
):
    relative_l2 = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(
        reference
    )
    boundary_error = torch.linalg.vector_norm(
        boundary_prediction - boundary_reference
    ) / torch.linalg.vector_norm(boundary_reference)
    half = trace_prediction.shape[0] // 2
    predicted_jump = trace_prediction[:half] - trace_prediction[half:]
    reference_jump = trace_reference[:half] - trace_reference[half:]
    jump_error = torch.linalg.vector_norm(
        predicted_jump - reference_jump
    ) / torch.linalg.vector_norm(reference_jump)
    target = np.asarray((parameters["K_I"], parameters["K_II"]), dtype=float)
    measured = np.asarray((report.k_i, report.k_ii), dtype=float)
    k_error = np.linalg.norm(measured - target) / max(np.linalg.norm(target), 1.0e-24)
    effective = parameters["young_modulus"]
    if parameters["assumption"] == "plane_strain":
        effective /= 1.0 - parameters["poisson_ratio"] ** 2
    reference_j = float(np.dot(target, target) / effective)
    return {
        "relative_l2_error": float(relative_l2.detach().cpu()),
        "relative_boundary_error": float(boundary_error.detach().cpu()),
        "relative_energy_error": abs(predicted_energy - reference_energy)
        / reference_energy,
        "crack_jump_relative_error": float(jump_error.detach().cpu()),
        "stress_intensity_relative_error": float(k_error),
        "stress_intensity_path_variation": float(report.path_variation),
        "relative_j_error": abs(report.j_integral - reference_j) / reference_j,
        "predicted_energy": float(predicted_energy),
        "training_integral_energy": float(training_energy),
        "refined_integral_energy": float(refined_energy),
        "reference_energy": float(reference_energy),
        "learned_K_I": float(report.k_i),
        "learned_K_II": float(report.k_ii),
    }


def _vector_parameters(spec):
    metadata = dict(spec.metadata)
    if metadata.get("provider") != "agentfem-learning.xdem":
        raise ValueError("The vector trainer requires provider='agentfem-learning.xdem'.")
    if metadata.get("problem") != "williams_vector_tip":
        raise NotImplementedError(
            "The vector trainer supports only the williams_vector_tip reference."
        )
    geometry = dict(metadata["geometry"])
    material = dict(metadata["material"])
    loading = dict(metadata["loading"])
    return {
        "radius": float(geometry["radius"]),
        "tip_core_radius": float(geometry["tip_core_radius"]),
        "young_modulus": float(material["young_modulus"]),
        "poisson_ratio": float(material["poisson_ratio"]),
        "assumption": str(material["assumption"]),
        "K_I": float(loading["K_I"]),
        "K_II": float(loading["K_II"]),
    }


def _sample_counts(spec):
    plans = {item.name: item for item in spec.sampling}
    return (
        int(plans["vector_slit_annulus_energy_points"].count),
        int(plans["vector_circular_boundary_points"].count),
    )


__all__ = [
    "VectorTrainingOutcome",
    "WilliamsVectorNetwork",
    "train_vector_reference",
    "williams_vector_field",
]
