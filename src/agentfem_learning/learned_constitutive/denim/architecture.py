# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""DENIM v1 discrete-energy J2 architecture.

The mechanics remain explicit while compact neural functions represent the
unknown hardening and recovery laws.  This module is runtime implementation;
no DENIM-specific name or state layout enters AgentFEM core.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

_VOIGT_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def double_contract(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    weights = left.new_tensor(_VOIGT_WEIGHTS)
    return (left * right * weights).sum(dim=-1)


def deviatoric(value: torch.Tensor) -> torch.Tensor:
    mean = value[..., :3].mean(dim=-1, keepdim=True)
    return torch.cat((value[..., :3] - mean, value[..., 3:]), dim=-1)


def von_mises(value: torch.Tensor) -> torch.Tensor:
    selected = deviatoric(value)
    squared = 1.5 * double_contract(selected, selected)
    # sqrt has an unbounded derivative at exactly zero.  A numerically tiny
    # floor leaves stresses unchanged while keeping the elastic-origin
    # automatic-differentiation tangent finite.
    return torch.sqrt(torch.clamp(squared, min=torch.finfo(value.dtype).tiny))


def elastic_stress(strain, plastic_strain, young, poisson):
    elastic = strain - plastic_strain
    shear = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    trace = elastic[..., :3].sum(dim=-1)
    mean = trace / 3.0
    normal = (
        2.0 * shear[..., None] * (elastic[..., :3] - mean[..., None])
        + bulk[..., None] * trace[..., None]
    )
    return torch.cat((normal, 2.0 * shear[..., None] * elastic[..., 3:]), dim=-1)


class MonotoneIsotropicHardening(nn.Module):
    def __init__(self, neurons: int = 12, plastic_scale: float = 0.01):
        super().__init__()
        self.plastic_scale = float(plastic_scale)
        self.raw_weight = nn.Parameter(torch.full((neurons,), -5.0))
        self.raw_slope = nn.Parameter(torch.linspace(-1.5, 1.5, neurons))
        self.raw_linear = nn.Parameter(torch.tensor(-5.0))

    def _positive_parameters(self):
        return (
            F.softplus(self.raw_weight),
            F.softplus(self.raw_slope) + 1.0e-6,
            F.softplus(self.raw_linear),
        )

    def forward(self, peeq, stress_scale):
        weight, slope, linear = self._positive_parameters()
        coordinate = peeq[..., None] / self.plastic_scale
        saturation = -torch.expm1(-slope * coordinate)
        dimensionless = linear * coordinate.squeeze(-1) + (weight * saturation).sum(-1)
        return stress_scale * dimensionless

    def stored_energy(self, peeq, stress_scale):
        coordinate = torch.linspace(0.0, 1.0, 65, dtype=peeq.dtype, device=peeq.device)
        points = peeq[..., None] * coordinate
        return torch.trapezoid(self.forward(points, stress_scale[..., None]), points, dim=-1)


class ObjectiveRecoveryNetwork(nn.Module):
    def __init__(self, channels: int = 2, hidden: int = 24):
        super().__init__()
        self.channels = int(channels)
        size = 3 + 3 * self.channels
        self.network = nn.Sequential(
            nn.Linear(size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.channels),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -1.0)

    def forward(self, peeq, isotropic_radius, flow_direction, memories, stress_scale, reversal):
        scale = stress_scale.clamp_min(1.0)
        norms = torch.sqrt(
            torch.clamp(
                double_contract(memories, memories),
                min=torch.finfo(memories.dtype).tiny,
            )
        )
        norms = norms / scale[..., None]
        projections = double_contract(memories, flow_direction[..., None, :])
        projections = projections / scale[..., None]
        cross = torch.zeros_like(norms)
        if self.channels > 1:
            cross = double_contract(memories, torch.roll(memories, -1, dims=-2))
            cross = cross / scale.square()[..., None]
        scalars = torch.stack((peeq / 0.02, isotropic_radius / scale, reversal), dim=-1)
        features = torch.cat((scalars, norms, projections, cross), dim=-1)
        return 200.0 * torch.sigmoid(self.network(features)) + 1.0e-6


@dataclass
class GrayboxState:
    plastic_strain: torch.Tensor
    peeq: torch.Tensor
    memories: torch.Tensor
    previous_flow: torch.Tensor

    @property
    def backstress(self):
        return self.memories.sum(dim=-2)


class NeuralHardeningLaw(nn.Module):
    def __init__(self, channels: int = 2, hidden: int = 24, neurons: int = 12):
        super().__init__()
        self.channels = int(channels)
        self.raw_moduli = nn.Parameter(torch.linspace(2.0, 0.0, channels))
        self.log_total_modulus_ratio = nn.Parameter(torch.tensor(5.0))
        self.isotropic = MonotoneIsotropicHardening(neurons=neurons)
        self.recovery = ObjectiveRecoveryNetwork(channels=channels, hidden=hidden)

    def moduli(self, stress_scale):
        fractions = torch.softmax(self.raw_moduli, dim=0)
        total = torch.exp(self.log_total_modulus_ratio).clamp(max=500.0)
        return total * stress_scale[..., None] * fractions

    def update_memories(self, state, flow_direction, increment, stress_scale):
        isotropic = self.isotropic(state.peeq, stress_scale)
        tiny = torch.finfo(flow_direction.dtype).tiny
        old_norm = torch.sqrt(
            torch.clamp(
                double_contract(state.previous_flow, state.previous_flow),
                min=tiny,
            )
        )
        current_norm = torch.sqrt(
            torch.clamp(double_contract(flow_direction, flow_direction), min=tiny)
        )
        reversal = double_contract(state.previous_flow, flow_direction) / (old_norm * current_norm).clamp_min(1.0e-12)
        recovery = self.recovery(state.peeq, isotropic, flow_direction, state.memories, stress_scale, reversal)
        moduli = self.moduli(stress_scale)
        numerator = state.memories + (2.0 / 3.0) * moduli[..., :, None] * increment[..., None, None] * flow_direction[..., None, :]
        denominator = 1.0 + recovery * increment[..., None]
        return numerator / denominator[..., None], recovery, moduli


def _candidate_update(trial_deviatoric, state, increment, shear, yield_stress, law, *, direction_iterations):
    shifted = trial_deviatoric - state.backstress
    direction = 1.5 * shifted / von_mises(shifted).clamp_min(1.0)[..., None]
    for _ in range(direction_iterations):
        memories, recovery, moduli = law.update_memories(state, direction, increment, yield_stress)
        denominator = 1.0 + recovery * increment[..., None]
        effective = trial_deviatoric - (state.memories / denominator[..., None]).sum(-2)
        direction = 1.5 * effective / von_mises(effective).clamp_min(1.0)[..., None]
    memories, recovery, moduli = law.update_memories(state, direction, increment, yield_stress)
    denominator = 1.0 + recovery * increment[..., None]
    effective = trial_deviatoric - (state.memories / denominator[..., None]).sum(-2)
    radius = yield_stress + law.isotropic(state.peeq + increment, yield_stress)
    modulus = 3.0 * shear + (moduli / denominator).sum(-1)
    return von_mises(effective) - modulus * increment - radius, direction, memories, recovery, moduli


def advance(strain, state, young, poisson, yield_stress, law, *, bisection_iterations=28, direction_iterations=6):
    """Advance a vectorized batch by one strain-controlled increment."""

    trial = elastic_stress(strain, state.plastic_strain, young, poisson)
    trial_dev = deviatoric(trial)
    shifted = trial_dev - state.backstress
    radius = yield_stress + law.isotropic(state.peeq, yield_stress)
    trial_function = von_mises(shifted) - radius
    plastic = trial_function > yield_stress.clamp_min(1.0) * 1.0e-12
    shear = young / (2.0 * (1.0 + poisson))
    lower = torch.zeros_like(trial_function)
    upper = 2.0 * F.relu(trial_function) / (3.0 * shear).clamp_min(1.0) + 1.0e-14
    for _ in range(16):
        residual, *_ = _candidate_update(trial_dev, state, upper, shear, yield_stress, law, direction_iterations=direction_iterations)
        upper = torch.where(plastic & (residual > 0.0), 2.0 * upper, upper)
    for _ in range(bisection_iterations):
        middle = 0.5 * (lower + upper)
        residual, *_ = _candidate_update(trial_dev, state, middle, shear, yield_stress, law, direction_iterations=direction_iterations)
        lower = torch.where(plastic & (residual > 0.0), middle, lower)
        upper = torch.where(plastic & (residual <= 0.0), middle, upper)
    increment = torch.where(plastic, 0.5 * (lower + upper), torch.zeros_like(lower))
    residual, direction, memories, recovery, moduli = _candidate_update(trial_dev, state, increment, shear, yield_stress, law, direction_iterations=direction_iterations)
    direction = torch.where(plastic[..., None], direction, torch.zeros_like(direction))
    memories = torch.where(plastic[..., None, None], memories, state.memories)
    updated = GrayboxState(
        plastic_strain=state.plastic_strain + increment[..., None] * direction,
        peeq=state.peeq + increment,
        memories=memories,
        previous_flow=torch.where(plastic[..., None], direction, state.previous_flow),
    )
    stress = elastic_stress(strain, updated.plastic_strain, young, poisson)
    return stress, updated, {
        "plastic_increment": increment,
        "yield_residual": torch.where(plastic, residual, torch.zeros_like(residual)),
        "recovery": recovery,
        "moduli": moduli,
        "plastic": plastic,
    }


DENIM = NeuralHardeningLaw


__all__ = [
    "DENIM",
    "GrayboxState",
    "NeuralHardeningLaw",
    "advance",
    "deviatoric",
    "double_contract",
    "elastic_stress",
    "von_mises",
]
