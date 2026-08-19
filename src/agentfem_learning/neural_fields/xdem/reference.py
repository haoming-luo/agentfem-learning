"""A compact, independently verifiable XDEM-style PyTorch reference solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import pi
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ReferenceTrainingOptions:
    """Numerical controls for the small Williams-field reference provider."""

    adam_epochs: int = 500
    lbfgs_steps: int = 6
    learning_rate: float = 5.0e-3
    boundary_penalty: float = 400.0
    hidden_layers: tuple[int, ...] = (24, 24)
    seed: int = 2026
    device: str = "cpu"
    dtype: str = "float64"
    progress: bool = False

    def __post_init__(self) -> None:
        if int(self.adam_epochs) <= 0:
            raise ValueError("adam_epochs must be positive.")
        if int(self.lbfgs_steps) < 0:
            raise ValueError("lbfgs_steps must be nonnegative.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if float(self.boundary_penalty) <= 0.0:
            raise ValueError("boundary_penalty must be positive.")
        if not self.hidden_layers or any(int(width) <= 0 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths.")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")


class WilliamsModeIIINetwork(torch.nn.Module):
    """Regular MLP plus an explicit Mode-III Williams crack-tip basis."""

    def __init__(
        self,
        *,
        radius: float,
        tip_core_radius: float,
        boundary_displacement: float,
        hidden_layers: tuple[int, ...] = (24, 24),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.tip_core_radius = float(tip_core_radius)
        widths = (4, *(int(item) for item in hidden_layers), 1)
        layers: list[torch.nn.Module] = []
        for input_width, output_width in pairwise(widths):
            linear = torch.nn.Linear(input_width, output_width, dtype=dtype)
            torch.nn.init.xavier_uniform_(linear.weight, gain=0.08)
            torch.nn.init.zeros_(linear.bias)
            layers.append(linear)
            if output_width != 1:
                layers.append(torch.nn.Tanh())
        self.regular = torch.nn.Sequential(*layers)
        self.tip_amplitude = torch.nn.Parameter(
            torch.tensor(0.5 * float(boundary_displacement), dtype=dtype)
        )

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        radius = torch.sqrt(torch.clamp(x.square() + y.square(), min=1.0e-24))
        angle = torch.atan2(y, x)
        tip = torch.sqrt(radius / self.radius) * torch.sin(0.5 * angle)
        # The Williams feature already has the intended branch cut on the
        # negative x-axis. A separate piecewise branch flag would introduce a
        # second, nonphysical discontinuity at x=0 that autograd cannot price.
        return torch.cat(
            (x / self.radius, y / self.radius, radius / self.radius, tip),
            dim=1,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        features = self.features(coordinates)
        tip = features[:, 3:4]
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        radial = torch.sqrt(torch.clamp(x.square() + y.square(), min=1.0e-24))
        span = self.radius - self.tip_core_radius
        envelope = (
            (radial - self.tip_core_radius)
            * (self.radius - radial)
            / (span * span)
        )
        return envelope * self.regular(features) + self.tip_amplitude * tip


@dataclass(frozen=True)
class ReferenceTrainingOutcome:
    """Numerical payload consumed by the AgentFEM Step wrapper."""

    model: WilliamsModeIIINetwork
    epochs: np.ndarray
    losses: np.ndarray
    coordinates: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray
    metrics: Mapping[str, float]
    device: str
    dtype: str

    def write_field(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            coordinates=self.coordinates,
            prediction=self.prediction,
            reference=self.reference,
        )
        return output


def train_mode_iii_reference(
    spec,
    options: ReferenceTrainingOptions,
) -> ReferenceTrainingOutcome:
    """Optimize the declared slit-annulus energy problem with PyTorch."""

    parameters = _problem_parameters(spec)
    radius = parameters["radius"]
    core = parameters["tip_core_radius"]
    shear = parameters["shear_modulus"]
    displacement = parameters["boundary_displacement"]
    domain_count, boundary_count = _sample_counts(spec)
    dtype = torch.float64 if options.dtype == "float64" else torch.float32
    device = _resolve_device(options.device, dtype=options.dtype)

    torch.manual_seed(int(options.seed))
    model = WilliamsModeIIINetwork(
        radius=radius,
        tip_core_radius=core,
        boundary_displacement=displacement,
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
    boundary_target = williams_mode_iii(
        boundary,
        radius=radius,
        displacement=displacement,
    ).detach()
    area = pi * (radius * radius - core * core)
    exact_energy = analytical_mode_iii_energy(
        radius=radius,
        core=core,
        shear_modulus=shear,
        boundary_displacement=displacement,
    )
    scale = max(displacement * displacement, 1.0e-24)

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = model(domain)
        gradient = torch.autograd.grad(
            prediction,
            domain,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
        )[0]
        internal_energy = 0.5 * shear * area * gradient.square().sum(dim=1).mean()
        boundary_error = (model(boundary) - boundary_target).square().mean()
        loss = (
            internal_energy / exact_energy
            + options.boundary_penalty * boundary_error / scale
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
                f"[agentfem-learning.xdem] Adam {epoch}/{options.adam_epochs}: "
                f"{losses[-1]:.6e}"
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
            coordinate = options.adam_epochs + step
            epochs.append(float(coordinate))
            losses.append(float(selected.detach().cpu()))
            if options.progress:
                print(
                    f"[agentfem-learning.xdem] L-BFGS {step}/{options.lbfgs_steps}: "
                    f"{losses[-1]:.6e}"
                )

    evaluation = _evaluation_grid(
        radius=radius,
        core=core,
        dtype=dtype,
        device=device,
    )
    prediction = model(evaluation)
    reference = williams_mode_iii(
        evaluation,
        radius=radius,
        displacement=displacement,
    )
    metrics = _verification_metrics(
        model,
        evaluation=evaluation,
        prediction=prediction,
        reference=reference,
        radius=radius,
        core=core,
        shear_modulus=shear,
        boundary_displacement=displacement,
        exact_energy=exact_energy,
        boundary_count=boundary_count,
    )
    return ReferenceTrainingOutcome(
        model=model,
        epochs=np.asarray(epochs, dtype=float),
        losses=np.asarray(losses, dtype=float),
        coordinates=evaluation.detach().cpu().numpy(),
        prediction=prediction.detach().cpu().numpy(),
        reference=reference.detach().cpu().numpy(),
        metrics=metrics,
        device=str(device),
        dtype=options.dtype,
    )


def williams_mode_iii(
    coordinates: torch.Tensor,
    *,
    radius: float,
    displacement: float,
) -> torch.Tensor:
    """Return the normalized leading Williams anti-plane displacement."""

    x = coordinates[:, 0:1]
    y = coordinates[:, 1:2]
    radial = torch.sqrt(torch.clamp(x.square() + y.square(), min=1.0e-24))
    angle = torch.atan2(y, x)
    return float(displacement) * torch.sqrt(radial / float(radius)) * torch.sin(
        0.5 * angle
    )


def analytical_mode_iii_energy(
    *,
    radius: float,
    core: float,
    shear_modulus: float,
    boundary_displacement: float,
) -> float:
    """Return strain energy per thickness for the slit-annulus field."""

    return (
        pi
        * float(shear_modulus)
        * float(boundary_displacement) ** 2
        * (float(radius) - float(core))
        / (4.0 * float(radius))
    )


def _problem_parameters(spec) -> dict[str, float]:
    metadata = dict(spec.metadata)
    if metadata.get("provider") != "agentfem-learning.xdem":
        raise ValueError(
            "The reference trainer requires provider='agentfem-learning.xdem'."
        )
    if metadata.get("problem") != "williams_mode_iii_tip":
        raise NotImplementedError(
            "The first AgentFEM-Learning XDEM provider supports only "
            "williams_mode_iii_tip."
        )
    geometry = dict(metadata.get("geometry", {}))
    material = dict(metadata.get("material", {}))
    loading = dict(metadata.get("loading", {}))
    return {
        "radius": float(geometry["radius"]),
        "tip_core_radius": float(geometry["tip_core_radius"]),
        "shear_modulus": float(material["shear_modulus"]),
        "boundary_displacement": float(loading["boundary_displacement"]),
    }


def _sample_counts(spec) -> tuple[int, int]:
    plans = {item.name: item for item in spec.sampling}
    return (
        int(plans["slit_annulus_energy_points"].count),
        int(plans["circular_boundary_points"].count),
    )


def _resolve_device(selected: str, *, dtype: str) -> torch.device:
    normalized = str(selected).strip().lower()
    if normalized == "auto":
        # PyTorch's MPS backend does not support float64.  Scientific
        # reference runs therefore remain on CPU by default, while users can
        # opt into MPS explicitly with dtype="float32".
        normalized = (
            "mps"
            if dtype == "float32" and torch.backends.mps.is_available()
            else "cpu"
        )
    if normalized == "mps" and dtype == "float64":
        raise ValueError(
            "PyTorch MPS does not support float64; use dtype='float32' or "
            "device='cpu'."
        )
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' was requested but Apple MPS is unavailable.")
    return torch.device(normalized)


def _sample_annulus(count, *, radius, core, seed, dtype, device):
    # A tensor-product midpoint rule is deliberately used instead of a small
    # random cloud. Energy methods are vulnerable to networks exploiting gaps
    # between random collocation points and reporting spuriously low energy.
    selected_count = int(count)
    radial_count = max(4, int(np.sqrt(selected_count / 2.0)))
    while radial_count > 4 and selected_count % radial_count:
        radial_count -= 1
    if selected_count % radial_count:
        raise ValueError(
            "domain sample count must admit at least four radial strata."
        )
    angular_count = selected_count // radial_count
    radial_fraction = (
        torch.arange(radial_count, dtype=dtype) + 0.5
    ) / radial_count
    radial = torch.sqrt(
        core * core + radial_fraction * (radius * radius - core * core)
    )
    angular_shift = (int(seed) % angular_count) / angular_count
    angle = -pi + (
        torch.arange(angular_count, dtype=dtype) + 0.5 + angular_shift
    ) * (2.0 * pi / angular_count)
    rr, tt = torch.meshgrid(radial, angle, indexing="ij")
    coordinates = torch.stack((rr * torch.cos(tt), rr * torch.sin(tt)), dim=-1)
    coordinates = coordinates.reshape(-1, 2)
    return coordinates.to(device).requires_grad_(True)


def _sample_boundaries(count, *, radius, core, dtype, device):
    half = max(4, int(count) // 2)
    angle = torch.linspace(-pi, pi, half + 1, dtype=dtype)[:-1].reshape(-1, 1)
    outer = torch.cat((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
    inner = torch.cat((core * torch.cos(angle), core * torch.sin(angle)), dim=1)
    return torch.cat((outer, inner), dim=0).to(device)


def _evaluation_grid(*, radius, core, dtype, device):
    radial_count = 48
    angular_count = 96
    fractions = (torch.arange(radial_count, dtype=dtype) + 0.5) / radial_count
    radial = torch.sqrt(core * core + fractions * (radius * radius - core * core))
    angle = -pi + (torch.arange(angular_count, dtype=dtype) + 0.5) * (
        2.0 * pi / angular_count
    )
    rr, tt = torch.meshgrid(radial, angle, indexing="ij")
    coordinates = torch.stack((rr * torch.cos(tt), rr * torch.sin(tt)), dim=-1)
    return coordinates.reshape(-1, 2).to(device).requires_grad_(True)


def _verification_metrics(
    model,
    *,
    evaluation,
    prediction,
    reference,
    radius,
    core,
    shear_modulus,
    boundary_displacement,
    exact_energy,
    boundary_count,
) -> dict[str, float]:
    difference = prediction - reference
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
    gradient = torch.autograd.grad(
        prediction,
        evaluation,
        grad_outputs=torch.ones_like(prediction),
        create_graph=False,
        retain_graph=True,
    )[0]
    area = pi * (radius * radius - core * core)
    predicted_energy = 0.5 * shear_modulus * area * gradient.square().sum(dim=1).mean()
    energy_error = torch.abs(predicted_energy - exact_energy) / exact_energy

    boundary = _sample_boundaries(
        boundary_count,
        radius=radius,
        core=core,
        dtype=evaluation.dtype,
        device=evaluation.device,
    )
    boundary_prediction = model(boundary)
    boundary_reference = williams_mode_iii(
        boundary,
        radius=radius,
        displacement=boundary_displacement,
    )
    boundary_error = torch.linalg.vector_norm(
        boundary_prediction - boundary_reference
    ) / torch.linalg.vector_norm(boundary_reference)

    crack_radius = 0.5 * (radius + core)
    face_offset = 1.0e-6 * radius
    face = torch.tensor(
        [[-crack_radius, face_offset], [-crack_radius, -face_offset]],
        dtype=evaluation.dtype,
        device=evaluation.device,
        requires_grad=True,
    )
    face_prediction = model(face)
    face_reference = williams_mode_iii(
        face,
        radius=radius,
        displacement=boundary_displacement,
    )
    jump = face_prediction[0] - face_prediction[1]
    exact_jump = face_reference[0] - face_reference[1]
    jump_error = torch.abs(jump - exact_jump) / torch.abs(exact_jump)
    predicted_gradient = torch.autograd.grad(
        face_prediction,
        face,
        grad_outputs=torch.ones_like(face_prediction),
        create_graph=False,
        retain_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        face_reference,
        face,
        grad_outputs=torch.ones_like(face_reference),
        create_graph=False,
    )[0]
    traction_scale = max(
        abs(shear_modulus * boundary_displacement / radius),
        1.0e-24,
    )
    traction_error = (
        shear_modulus
        * torch.max(torch.abs(predicted_gradient[:, 1] - reference_gradient[:, 1]))
        / traction_scale
    )
    return {
        "relative_l2_error": float(relative_l2.detach().cpu()),
        "relative_boundary_error": float(boundary_error.detach().cpu()),
        "relative_energy_error": float(energy_error.detach().cpu()),
        "crack_jump_relative_error": float(jump_error.detach().cpu()),
        "crack_face_traction_relative_error": float(traction_error.detach().cpu()),
        "predicted_energy": float(predicted_energy.detach().cpu()),
        "reference_energy": float(exact_energy),
        "learned_tip_amplitude": float(model.tip_amplitude.detach().cpu()),
    }


__all__ = [
    "ReferenceTrainingOptions",
    "ReferenceTrainingOutcome",
    "WilliamsModeIIINetwork",
    "analytical_mode_iii_energy",
    "train_mode_iii_reference",
    "williams_mode_iii",
]
