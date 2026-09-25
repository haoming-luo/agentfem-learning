# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""AgentFEM material adapter for a prepared DENIM v1 bundle."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from agentfem.constitutive import (
    MaterialParameter,
    MaterialParameterSchema,
    SmallStrainMaterialPointBatchInput,
    SmallStrainMaterialPointBatchOutput,
    SmallStrainMaterialPointInput,
    SmallStrainMaterialPointOutput,
    small_strain_tangent_convention,
)

from ..artifacts import ModelBundle
from ..torch_runtime import TorchRuntime
from .architecture import GrayboxState, advance, double_contract
from .state_codec import pack_state, state_schema, unpack_state


def _required_parameter(parameters, name: str) -> float:
    try:
        value = float(parameters[name])
    except KeyError as exc:
        raise ValueError(f"DENIM requires material parameter {name!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Material parameter {name!r} must be finite.")
    return value


def _tensor_to_voigt(values) -> np.ndarray:
    selected = np.asarray(values, dtype=float)
    return selected[..., (0, 1, 2, 0, 1, 0), (0, 1, 2, 1, 2, 2)]


def _voigt_to_tensor(values) -> np.ndarray:
    selected = np.asarray(values, dtype=float)
    result = np.zeros((*selected.shape[:-1], 3, 3), dtype=float)
    result[..., 0, 0] = selected[..., 0]
    result[..., 1, 1] = selected[..., 1]
    result[..., 2, 2] = selected[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = selected[..., 3]
    result[..., 1, 2] = result[..., 2, 1] = selected[..., 4]
    result[..., 0, 2] = result[..., 2, 0] = selected[..., 5]
    return result


@dataclass
class DenimMaterial:
    """Pure, cached, batched material update produced by the generic provider."""

    bundle: ModelBundle
    law: object
    runtime: TorchRuntime
    channels: int
    tangent_mode: str = "autodiff_consistent"
    name: str = "learned_constitutive"

    def __post_init__(self) -> None:
        self.channels = int(self.channels)
        self.state_schema = state_schema(
            self.channels,
            version=str(self.bundle.manifest.get("state_schema_version", "1.0.0")),
        )
        self.parameter_schema = MaterialParameterSchema(
            name="denim_v1",
            version=str(self.bundle.manifest.get("schema_version", "1.0.0")),
            parameters=tuple(
                MaterialParameter(
                    name=item["name"],
                    unit=item.get("unit"),
                    lower=item.get("minimum"),
                    upper=item.get("maximum"),
                    default=item.get("default"),
                    description=item.get("description", "DENIM material parameter."),
                )
                for item in self.bundle.manifest["parameter_schema"]
            ),
        )
        self.tangent_convention = small_strain_tangent_convention()
        if self.tangent_mode not in {"autodiff_consistent", "none"}:
            raise ValueError("DENIM tangent mode must be autodiff_consistent or none.")
        self._torch, self._device, self._dtype = self.runtime.resolve()
        self.law.to(device=self._device, dtype=self._dtype)
        self.law.eval()

    def _parameters(self, request):
        count = request.point_count
        torch = self._torch
        values = []
        for name in ("young", "poisson", "yield_stress"):
            selected = _required_parameter(request.parameters, name)
            values.append(
                torch.full((count,), selected, dtype=self._dtype, device=self._device)
            )
        if torch.any(values[0] <= 0) or torch.any((values[1] <= -1) | (values[1] >= 0.5)):
            raise ValueError("DENIM requires young > 0 and -1 < poisson < 0.5.")
        if torch.any(values[2] <= 0):
            raise ValueError("DENIM requires yield_stress > 0.")
        return tuple(values)

    def _single_stress(self, strain, packed, young, poisson, yield_stress):
        state = GrayboxState(
            packed[:6][None, :],
            packed[6][None],
            packed[7 : 7 + 6 * self.channels].reshape(1, self.channels, 6),
            packed[7 + 6 * self.channels :][None, :],
        )
        stress, _, _ = advance(
            strain[None, :],
            state,
            young[None],
            poisson[None],
            yield_stress[None],
            self.law,
        )
        return stress[0]

    def _tangent(self, strain, packed, young, poisson, yield_stress):
        if self.tangent_mode == "none":
            raise RuntimeError(
                "This artifact does not provide a consistent tangent and cannot "
                "be used by an implicit global solve."
            )
        torch = self._torch
        jacobian = torch.func.jacrev(self._single_stress, argnums=0)
        return torch.func.vmap(jacobian)(strain, packed, young, poisson, yield_stress)

    def _domain_status(self, request, peeq):
        domain = dict(self.bundle.manifest.get("applicability_domain", {}))
        status = np.full(request.point_count, "in_domain", dtype=object)
        strain_limit = domain.get("maximum_absolute_strain")
        if strain_limit is not None:
            outside = np.max(np.abs(request.strain_new), axis=(1, 2)) > float(
                strain_limit
            )
            status[outside] = "out_of_domain"
        peeq_limit = domain.get("maximum_peeq")
        if peeq_limit is not None:
            status[np.asarray(peeq) > float(peeq_limit)] = "out_of_domain"
        return tuple(str(value) for value in status)

    def update_batch(self, request: SmallStrainMaterialPointBatchInput):
        if request.state_schema.summary() != self.state_schema.summary():
            raise ValueError("DENIM request state schema differs from the model bundle.")
        if request.parameter_schema.summary() != self.parameter_schema.summary():
            raise ValueError("DENIM parameter schema differs from the model bundle.")
        started = time.perf_counter()
        torch = self._torch
        strain = torch.as_tensor(
            _tensor_to_voigt(request.strain_new),
            dtype=self._dtype,
            device=self._device,
        )
        old_state = unpack_state(
            request.state_old,
            channels=self.channels,
            torch=torch,
            dtype=self._dtype,
            device=self._device,
        )
        packed_old = pack_state(old_state)
        young, poisson, yield_stress = self._parameters(request)
        with torch.enable_grad():
            strain_for_tangent = strain.detach().requires_grad_(True)
            tangent = self._tangent(
                strain_for_tangent,
                packed_old.detach(),
                young,
                poisson,
                yield_stress,
            )
        with torch.inference_mode():
            stress, new_state, diagnostics = advance(
                strain,
                old_state,
                young,
                poisson,
                yield_stress,
                self.law,
            )
            packed_new = pack_state(new_state)
            elastic_strain = strain - new_state.plastic_strain
            elastic_energy = 0.5 * double_contract(stress, elastic_strain)
            isotropic_energy = self.law.isotropic.stored_energy(
                new_state.peeq,
                yield_stress,
            )
            moduli = self.law.moduli(yield_stress)
            kinematic_energy = (
                3.0 * double_contract(new_state.memories, new_state.memories) / (4.0 * moduli)
            ).sum(dim=-1)
            plastic_work = double_contract(
                stress,
                new_state.plastic_strain - old_state.plastic_strain,
            )
            old_isotropic = self.law.isotropic.stored_energy(old_state.peeq, yield_stress)
            old_kinematic = (
                3.0 * double_contract(old_state.memories, old_state.memories) / (4.0 * moduli)
            ).sum(dim=-1)
            hardening_change = (
                isotropic_energy + kinematic_energy - old_isotropic - old_kinematic
            )
            dissipation = torch.clamp(plastic_work - hardening_change, min=0.0)
            yield_residual = diagnostics["yield_residual"]
            finite = torch.isfinite(packed_new).all(dim=1) & torch.isfinite(stress).all(dim=1)
            peeq_monotone = new_state.peeq + 1.0e-14 >= old_state.peeq
            plastic_trace = new_state.plastic_strain[:, :3].sum(dim=-1).abs()
        stress_np = _voigt_to_tensor(stress.detach().cpu().numpy())
        tangent_np = tangent.detach().cpu().numpy()
        state_np = packed_new.detach().cpu().numpy()
        if not bool(torch.all(finite)):
            raise FloatingPointError("DENIM produced a non-finite stress or state.")
        statuses = list(self._domain_status(request, new_state.peeq.cpu().numpy()))
        for index, valid in enumerate(peeq_monotone.cpu().numpy()):
            if not bool(valid):
                statuses[index] = "invalid_state"
        elapsed = time.perf_counter() - started
        count = request.point_count
        diagnostic_arrays = {
            "plastic_increment": diagnostics["plastic_increment"].cpu().numpy(),
            "yield_residual": yield_residual.cpu().numpy(),
            "plastic": diagnostics["plastic"].cpu().numpy(),
            "maximum_plastic_trace": plastic_trace.cpu().numpy(),
            "peeq_monotone": peeq_monotone.cpu().numpy(),
            "plastic_work_increment": plastic_work.cpu().numpy(),
            "hardening_storage_increment": hardening_change.cpu().numpy(),
            "modeled_dissipation_increment": dissipation.cpu().numpy(),
            "inference_seconds_per_point": np.full(count, elapsed / max(count, 1)),
        }
        point_diagnostics = tuple(
            {
                name: values[index].item()
                for name, values in diagnostic_arrays.items()
            }
            for index in range(count)
        )
        return SmallStrainMaterialPointBatchOutput(
            cauchy_stress=stress_np,
            consistent_tangent=tangent_np,
            state_new=state_np,
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
            stored_energy_density=(elastic_energy + isotropic_energy + kinematic_energy)
            .cpu()
            .numpy(),
            dissipation_density_increment=dissipation.cpu().numpy(),
            suggested_time_scale=np.ones(count),
            applicability=tuple(statuses),
            stored_energy_density_components={
                "elastic_storage": elastic_energy.cpu().numpy(),
                "isotropic_hardening_storage": isotropic_energy.cpu().numpy(),
                "kinematic_hardening_storage": kinematic_energy.cpu().numpy(),
            },
            diagnostics=point_diagnostics,
        )

    def update(self, point: SmallStrainMaterialPointInput):
        request = SmallStrainMaterialPointBatchInput(
            strain_old=point.strain_old[None, :],
            strain_new=point.strain_new[None, :],
            time=point.time,
            time_increment=point.time_increment,
            parameters=point.parameters,
            state_old=point.state_old[None, :],
            state_schema=point.state_schema,
            parameter_schema=point.parameter_schema,
            temperature=None if point.temperature is None else np.asarray([point.temperature]),
            temperature_increment=(
                None
                if point.temperature_increment is None
                else np.asarray([point.temperature_increment])
            ),
            field_variables={
                name: np.asarray([value])
                for name, value in point.field_variables.items()
            },
        )
        response = self.update_batch(request)
        return SmallStrainMaterialPointOutput(
            cauchy_stress=response.cauchy_stress[0],
            consistent_tangent=response.consistent_tangent[0],
            state_new=response.state_new[0],
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
            stored_energy_density=float(response.stored_energy_density[0]),
            dissipation_density_increment=float(
                response.dissipation_density_increment[0]
            ),
            stored_energy_density_components={
                name: float(values[0])
                for name, values in response.stored_energy_density_components.items()
            },
            diagnostics=response.diagnostics[0],
            suggested_time_scale=float(response.suggested_time_scale[0]),
            applicability=response.applicability[0],
        )


__all__ = ["DenimMaterial"]
