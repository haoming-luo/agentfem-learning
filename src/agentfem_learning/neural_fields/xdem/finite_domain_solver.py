"""Experimental finite-domain, multi-crack XDEM-D energy solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from agentfem import fracture

from .finite_domain import (
    StaticXDEMProblem2D,
    displacement_bc,
    rectangular_domain,
    static_crack_problem,
    traction_bc,
)
from .reference import ReferenceTrainingOptions, _resolve_device
from .tip_reports import stress_intensity_reports, tip_integration_plan
from .vector_reference import (
    TorchVectorFractureField,
    _elastic_state,
    williams_vector_field,
)


class FiniteDomainVectorNetwork(torch.nn.Module):
    """Vector MLP extended by one jump feature per crack and tip bases.

    The crack feature is confined to the declared segment, so adding a crack
    cannot create a hidden displacement jump along its infinite extension.
    Four leading branch functions per tip expose the square-root asymptotics
    without fixing their amplitudes in advance.
    """

    def __init__(
        self,
        problem: StaticXDEMProblem2D,
        *,
        displacement_scale: float,
        hidden_layers: tuple[int, ...] = (48, 48, 48),
        crack_decay: float = 8.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.displacement_scale = float(displacement_scale)
        self.crack_decay = float(crack_decay)
        if self.displacement_scale <= 0.0:
            raise ValueError("displacement_scale must be positive.")
        if self.crack_decay <= 0.0:
            raise ValueError("crack_decay must be positive.")
        xmin, xmax, ymin, ymax = problem.domain.bounds
        self.register_buffer(
            "domain_center",
            torch.tensor((0.5 * (xmin + xmax), 0.5 * (ymin + ymax)), dtype=dtype),
        )
        self.register_buffer(
            "domain_half_span",
            torch.tensor((0.5 * (xmax - xmin), 0.5 * (ymax - ymin)), dtype=dtype),
        )
        self.register_buffer(
            "crack_centers",
            torch.tensor(
                [
                    (0.5 * (item.start[0] + item.end[0]), 0.5 * (item.start[1] + item.end[1]))
                    for item in problem.cracks.cracks
                ],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "crack_tangents",
            torch.tensor([item.tangent for item in problem.cracks.cracks], dtype=dtype),
        )
        self.register_buffer(
            "crack_normals",
            torch.tensor([item.normal for item in problem.cracks.cracks], dtype=dtype),
        )
        self.register_buffer(
            "crack_lengths",
            torch.tensor([item.length for item in problem.cracks.cracks], dtype=dtype),
        )
        self.register_buffer(
            "tip_points",
            torch.tensor([item.point for item in problem.cracks.tips], dtype=dtype),
        )
        self.register_buffer(
            "tip_extensions",
            torch.tensor(
                [item.extension_direction for item in problem.cracks.tips], dtype=dtype
            ),
        )
        self.register_buffer(
            "tip_normals",
            torch.tensor([item.normal for item in problem.cracks.tips], dtype=dtype),
        )
        self.register_buffer(
            "tip_crack_lengths",
            torch.tensor(
                [problem.cracks.crack(item.crack_id).length for item in problem.cracks.tips],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "tip_angles",
            torch.tensor(
                [
                    np.arctan2(item.extension_direction[1], item.extension_direction[0])
                    for item in problem.cracks.tips
                ],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "tip_support_radii",
            torch.tensor(
                [
                    problem.cracks.admissible_tip_radius(
                        item.tip_id, bounds=problem.domain.bounds
                    )
                    for item in problem.cracks.tips
                ],
                dtype=dtype,
            ),
        )
        characteristic_length = max(xmax - xmin, ymax - ymin)
        material_summary = problem.material.summary()
        stress_scale = (
            float(material_summary["young_modulus"])
            * self.displacement_scale
            / characteristic_length
        )
        self.register_buffer(
            "tip_sif_scales",
            stress_scale * torch.sqrt(self.tip_crack_lengths.clone()),
        )
        self.tip_amplitudes = torch.nn.Parameter(
            torch.zeros((len(problem.cracks.tips), 2), dtype=dtype)
        )
        input_width = 2 + len(problem.cracks.cracks) + 4 * len(problem.cracks.tips)
        widths = (input_width, *(int(item) for item in hidden_layers), 2)
        layers: list[torch.nn.Module] = []
        for input_width, output_width in pairwise(widths):
            layer = torch.nn.Linear(input_width, output_width, dtype=dtype)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.3)
            torch.nn.init.zeros_(layer.bias)
            layers.append(layer)
            if output_width != 2:
                layers.append(torch.nn.Tanh())
        self.network = torch.nn.Sequential(*layers)

    def crack_features(self, coordinates: torch.Tensor) -> torch.Tensor:
        relative = coordinates[:, None, :] - self.crack_centers[None, :, :]
        along = torch.einsum("nci,ci->nc", relative, self.crack_tangents)
        normal = torch.einsum("nci,ci->nc", relative, self.crack_normals)
        half_length = 0.5 * self.crack_lengths[None, :]
        window = torch.relu(1.0 - (along / half_length).square()).square()
        sign = torch.where(normal >= 0.0, 1.0, -1.0)
        decay = torch.exp(-self.crack_decay * torch.abs(normal) / self.crack_lengths[None, :])
        return sign * window * decay

    def tip_features(self, coordinates: torch.Tensor) -> torch.Tensor:
        relative = coordinates[:, None, :] - self.tip_points[None, :, :]
        local_x = torch.einsum("nti,ti->nt", relative, self.tip_extensions)
        local_y = torch.einsum("nti,ti->nt", relative, self.tip_normals)
        radial = torch.sqrt(torch.clamp(local_x.square() + local_y.square(), min=1.0e-24))
        angle = torch.atan2(local_y, local_x)
        half = 0.5 * angle
        backward = torch.minimum(local_x, torch.zeros_like(local_x))
        branch_support = torch.clamp(
            1.0 + backward / self.tip_crack_lengths[None, :], min=0.0, max=1.0
        )
        radial_scale = torch.sqrt(radial / self.tip_crack_lengths[None, :]) * torch.exp(
            -radial / self.tip_crack_lengths[None, :]
        )
        scale = radial_scale * branch_support
        features = torch.stack(
            (
                scale * torch.sin(half),
                scale * torch.cos(half),
                scale * torch.sin(half) * torch.sin(angle),
                scale * torch.cos(half) * torch.sin(angle),
            ),
            dim=2,
        )
        return features.reshape(len(coordinates), -1)

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        normalized = (coordinates - self.domain_center) / self.domain_half_span
        return torch.cat(
            (normalized, self.crack_features(coordinates), self.tip_features(coordinates)),
            dim=1,
        )

    @staticmethod
    def _smooth_step(value):
        selected = torch.clamp(value, min=0.0, max=1.0)
        return selected.square() * (3.0 - 2.0 * selected)

    def tip_enrichment(self, coordinates: torch.Tensor) -> torch.Tensor:
        enriched = torch.zeros(
            (len(coordinates), 2), dtype=coordinates.dtype, device=coordinates.device
        )
        for index, tip in enumerate(self.problem.cracks.tips):
            relative = coordinates - self.tip_points[index]
            local_x = relative @ self.tip_extensions[index]
            local_y = relative @ self.tip_normals[index]
            radial = torch.sqrt(torch.clamp(local_x.square() + local_y.square(), min=1.0e-24))
            support_radius = self.tip_support_radii[index]
            radial_cut = 1.0 - self._smooth_step(
                (radial - support_radius) / support_radius
            )
            length = self.tip_crack_lengths[index]
            backward_distance = torch.relu(-local_x)
            branch_cut = 1.0 - self._smooth_step(
                (backward_distance - 0.75 * length) / (0.25 * length)
            )
            amplitudes = self.tip_amplitudes[index] * self.tip_sif_scales[index]
            selected = williams_vector_field(
                coordinates,
                young_modulus=self.problem.material.summary()["young_modulus"],
                poisson_ratio=self.problem.material.summary()["poisson_ratio"],
                assumption=self.problem.material.summary()["assumption"],
                k_i=amplitudes[0],
                k_ii=amplitudes[1],
                tip=tip.point,
                crack_angle=self.tip_angles[index],
            )
            enriched = enriched + (radial_cut * branch_cut)[:, None] * selected
        return enriched

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        regular = self.displacement_scale * self.network(self.features(coordinates))
        return regular + self.tip_enrichment(coordinates)


@dataclass(frozen=True)
class FiniteDomainTrainingOutcome:
    model: FiniteDomainVectorNetwork
    epochs: np.ndarray
    losses: np.ndarray
    coordinates: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    crack_trace_coordinates: np.ndarray
    crack_trace_side: np.ndarray
    crack_trace_id: np.ndarray
    crack_trace_displacement: np.ndarray
    metrics: dict[str, float]
    stress_intensity: object
    device: str
    dtype: str

    def write_field(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            coordinates=self.coordinates,
            displacement=self.displacement,
            stress=self.stress,
            crack_trace_coordinates=self.crack_trace_coordinates,
            crack_trace_side=self.crack_trace_side,
            crack_trace_id=self.crack_trace_id,
            crack_trace_displacement=self.crack_trace_displacement,
        )
        return output


def train_finite_domain(
    spec,
    options: ReferenceTrainingOptions,
) -> FiniteDomainTrainingOutcome:
    """Train one finite-domain predefined-crack problem."""

    problem = problem_from_spec(spec)
    parameters = _material_parameters(problem)
    dtype = torch.float64 if options.dtype == "float64" else torch.float32
    device = _resolve_device(options.device, dtype=options.dtype)
    torch.manual_seed(int(options.seed))
    displacement_scale = _displacement_scale(problem)
    model = FiniteDomainVectorNetwork(
        problem,
        displacement_scale=displacement_scale,
        hidden_layers=options.hidden_layers,
        dtype=dtype,
    ).to(device)
    rules = spec.integration
    training_points = _random_domain_points(
        problem,
        rules.training.count,
        seed=rules.training.seed,
        dtype=dtype,
        device=device,
    )
    validation_points = _random_domain_points(
        problem,
        rules.validation.count,
        seed=rules.validation.seed,
        dtype=dtype,
        device=device,
    )
    refinement = rules.refinements[0]
    refined_points = _random_domain_points(
        problem,
        refinement.count,
        seed=refinement.seed,
        dtype=dtype,
        device=device,
    )
    boundary_data = _boundary_data(problem, dtype=dtype, device=device)
    energy_scale = parameters["young_modulus"] * displacement_scale**2

    def objective(points, *, create_graph):
        displacement = model(points)
        _, _, density = _elastic_state(
            displacement, points, parameters, create_graph=create_graph
        )
        internal = problem.domain.area * density.mean()
        external, condition_error = _boundary_terms(model, boundary_data)
        total = internal - external
        selected = total / energy_scale + options.boundary_penalty * condition_error / (
            displacement_scale * displacement_scale
        )
        return selected, internal, external, condition_error

    epochs: list[float] = []
    losses: list[float] = []
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    for epoch in range(1, options.adam_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = objective(training_points, create_graph=True)
        loss.backward()
        optimizer.step()
        epochs.append(float(epoch))
        losses.append(float(loss.detach().cpu()))
        if options.progress and (epoch == 1 or epoch % 100 == 0):
            print(
                f"[agentfem-learning.xdem] finite-domain Adam {epoch}/"
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
                selected, _, _, _ = objective(training_points, create_graph=True)
                selected.backward()
                return selected

            optimizer_lbfgs.step(closure)
            selected, _, _, _ = objective(training_points, create_graph=True)
            epochs.append(float(options.adam_epochs + step))
            losses.append(float(selected.detach().cpu()))

    evaluation = _evaluation_grid(problem, rules.validation.count, dtype=dtype, device=device)
    prediction = model(evaluation)
    _, stress, _ = _elastic_state(prediction, evaluation, parameters, create_graph=False)
    training_values = objective(training_points, create_graph=False)
    validation_values = objective(validation_points, create_graph=False)
    refined_values = objective(refined_points, create_graph=False)
    field = TorchVectorFractureField(model, parameters, dtype=dtype, device=device)
    plan = tip_integration_plan(
        problem.cracks,
        bounds=problem.domain.bounds,
        relative_path_tolerance=0.08,
        metadata={"problem": problem.name},
    )
    reports = stress_intensity_reports(
        field,
        cracks=problem.cracks,
        material=problem.material,
        plan=plan,
        metadata={"problem": problem.name, "solver": "finite_domain_xdem_d"},
    )
    trace_coordinates, trace_side, trace_id = _crack_trace(problem)
    trace_tensor = torch.as_tensor(trace_coordinates, dtype=dtype, device=device)
    trace_prediction = model(trace_tensor)
    energy_gap = abs(
        float(validation_values[1].detach().cpu()) - float(refined_values[1].detach().cpu())
    ) / max(abs(float(refined_values[1].detach().cpu())), energy_scale * 1.0e-12)
    metrics = {
        "training_internal_energy": float(training_values[1].detach().cpu()),
        "validation_internal_energy": float(validation_values[1].detach().cpu()),
        "refined_internal_energy": float(refined_values[1].detach().cpu()),
        "validation_external_work": float(validation_values[2].detach().cpu()),
        "relative_boundary_error": float(
            torch.sqrt(validation_values[3]).detach().cpu() / displacement_scale
        ),
        "validation_refinement_energy_gap": float(energy_gap),
        "maximum_stress_intensity_path_variation": float(
            max(item.path_variation for item in reports.reports)
        ),
    }
    return FiniteDomainTrainingOutcome(
        model=model,
        epochs=np.asarray(epochs, dtype=float),
        losses=np.asarray(losses, dtype=float),
        coordinates=evaluation.detach().cpu().numpy(),
        displacement=prediction.detach().cpu().numpy(),
        stress=stress.detach().cpu().numpy(),
        crack_trace_coordinates=trace_coordinates,
        crack_trace_side=trace_side,
        crack_trace_id=trace_id,
        crack_trace_displacement=trace_prediction.detach().cpu().numpy(),
        metrics=metrics,
        stress_intensity=reports,
        device=str(device),
        dtype=options.dtype,
    )


def problem_from_spec(spec) -> StaticXDEMProblem2D:
    metadata = dict(spec.metadata)
    if metadata.get("provider") != "agentfem-learning.xdem":
        raise ValueError("Finite-domain trainer requires the AgentFEM-Learning provider.")
    if metadata.get("problem") != "finite_domain_static_xdem_d":
        raise NotImplementedError("Finite-domain trainer received another problem type.")
    summary = dict(metadata["scientific_problem"])
    domain_summary = dict(summary["domain"])
    domain = rectangular_domain(
        domain_summary["bounds"],
        name=domain_summary["name"],
        metadata=domain_summary.get("metadata", {}),
    )
    material_summary = dict(summary["material"])
    material = fracture.linear_elastic_fracture_material(
        young_modulus=material_summary["young_modulus"],
        poisson_ratio=material_summary["poisson_ratio"],
        assumption=material_summary["assumption"],
    )
    crack_summary = dict(summary["cracks"])
    cracks = fracture.crack_set(
        *(
            fracture.segment(
                item["crack_id"],
                start=item["start"],
                end=item["end"],
                metadata=item.get("metadata", {}),
            )
            for item in crack_summary["cracks"]
        ),
        name=crack_summary["name"],
        tolerance=crack_summary["tolerance"],
        metadata=crack_summary.get("metadata", {}),
    )
    conditions = []
    for item in summary["conditions"]:
        constructor = displacement_bc if item["kind"] == "displacement" else traction_bc
        conditions.append(
            constructor(
                item["name"],
                item["boundary"],
                item["value"],
                metadata=item.get("metadata", {}),
            )
        )
    return static_crack_problem(
        domain=domain,
        material=material,
        cracks=cracks,
        conditions=conditions,
        name=summary["name"],
        metadata=summary.get("metadata", {}),
    )


def _material_parameters(problem):
    selected = problem.material.summary()
    return {
        "young_modulus": float(selected["young_modulus"]),
        "poisson_ratio": float(selected["poisson_ratio"]),
        "assumption": str(selected["assumption"]),
    }


def _displacement_scale(problem):
    prescribed = max(
        (
            abs(value)
            for condition in problem.conditions
            if condition.kind == "displacement"
            for value in condition.value
            if value is not None
        ),
        default=0.0,
    )
    traction = max(
        (
            float(np.linalg.norm(condition.value))
            for condition in problem.conditions
            if condition.kind == "traction"
        ),
        default=0.0,
    )
    xmin, xmax, ymin, ymax = problem.domain.bounds
    length = max(xmax - xmin, ymax - ymin)
    modulus = float(problem.material.summary()["young_modulus"])
    scale = max(prescribed, traction * length / modulus)
    if scale <= 0.0:
        raise ValueError("The finite-domain problem has no nonzero mechanical loading.")
    return scale


def _random_domain_points(problem, count, *, seed, dtype, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.rand((int(count), 2), generator=generator, dtype=dtype)
    xmin, xmax, ymin, ymax = problem.domain.bounds
    values[:, 0] = xmin + (xmax - xmin) * values[:, 0]
    values[:, 1] = ymin + (ymax - ymin) * values[:, 1]
    return values.to(device).detach().requires_grad_(True)


def _evaluation_grid(problem, count, *, dtype, device):
    side = max(16, int(np.sqrt(int(count))))
    xmin, xmax, ymin, ymax = problem.domain.bounds
    x = torch.linspace(xmin, xmax, side, dtype=dtype, device=device)
    y = torch.linspace(ymin, ymax, side, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).detach().requires_grad_(True)


def _boundary_data(problem, *, dtype, device):
    selected = []
    count = 128
    xmin, xmax, ymin, ymax = problem.domain.bounds
    for condition in problem.conditions:
        fraction = (torch.arange(count, dtype=dtype, device=device) + 0.5) / count
        if condition.boundary in {"left", "right"}:
            x = torch.full_like(fraction, xmin if condition.boundary == "left" else xmax)
            y = ymin + (ymax - ymin) * fraction
            length = ymax - ymin
        else:
            x = xmin + (xmax - xmin) * fraction
            y = torch.full_like(fraction, ymin if condition.boundary == "bottom" else ymax)
            length = xmax - xmin
        selected.append((condition, torch.stack((x, y), dim=1), length))
    return tuple(selected)


def _boundary_terms(model, boundary_data):
    external = None
    errors = []
    for condition, coordinates, length in boundary_data:
        prediction = model(coordinates)
        if condition.kind == "traction":
            traction = torch.as_tensor(
                condition.value, dtype=prediction.dtype, device=prediction.device
            )
            contribution = length * torch.mean(prediction @ traction)
            external = contribution if external is None else external + contribution
        else:
            for component in condition.constrained_components:
                errors.append(
                    (prediction[:, component] - condition.value[component]).square().mean()
                )
    if external is None:
        sample = next(iter(boundary_data))[1]
        external = torch.zeros((), dtype=sample.dtype, device=sample.device)
    condition_error = torch.stack(errors).mean() if errors else torch.zeros_like(external)
    return external, condition_error


def _crack_trace(problem, *, count_per_crack: int = 64):
    coordinates = []
    sides = []
    identifiers = []
    for crack in problem.cracks.cracks:
        fraction = (np.arange(count_per_crack, dtype=float) + 0.5) / count_per_crack
        centerline = (
            np.asarray(crack.start)[None, :] * (1.0 - fraction[:, None])
            + np.asarray(crack.end)[None, :] * fraction[:, None]
        )
        epsilon = 1.0e-7 * crack.length
        normal = np.asarray(crack.normal)
        for side in (1, -1):
            coordinates.append(centerline + side * epsilon * normal)
            sides.extend([side] * count_per_crack)
            identifiers.extend([crack.crack_id] * count_per_crack)
    return (
        np.vstack(coordinates),
        np.asarray(sides, dtype=int),
        np.asarray(identifiers, dtype="U"),
    )


__all__ = [
    "FiniteDomainTrainingOutcome",
    "FiniteDomainVectorNetwork",
    "problem_from_spec",
    "train_finite_domain",
]
