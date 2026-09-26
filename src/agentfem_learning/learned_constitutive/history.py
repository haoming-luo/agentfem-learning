# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral material-history execution for learned constitutive laws."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from agentfem.constitutive import MaterialLoadingPath, SmallStrainMaterialPointInput


@dataclass(frozen=True)
class LearnedMaterialHistory:
    """One accepted strain-controlled path and its typed provider evidence."""

    path: MaterialLoadingPath
    stress: np.ndarray
    state: np.ndarray
    stored_energy_density: np.ndarray | None
    dissipation_density_increment: np.ndarray | None
    applicability: tuple[str, ...]
    diagnostics: tuple[Mapping[str, object], ...]
    state_schema: object

    def state_variable(self, name: str) -> np.ndarray:
        """Return one named state variable without exposing packed offsets."""

        values = [self.state_schema.unpack(item)[name] for item in self.state]
        return np.asarray(values)

    @property
    def accepted(self) -> bool:
        return all(value in {"in_domain", "warning"} for value in self.applicability)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem-learning.learned-material-history.v1",
            "path": self.path.summary(),
            "accepted": self.accepted,
            "point_count": self.path.point_count,
            "applicability_counts": {
                status: self.applicability.count(status)
                for status in sorted(set(self.applicability))
            },
            "state_schema": self.state_schema.summary(),
        }


def run_small_strain_material_history(
    material,
    path: MaterialLoadingPath,
    *,
    initial_state=None,
) -> LearnedMaterialHistory:
    """Run a learned material on an explicit path with atomic state commits.

    The path contract, tensor conventions and named state schema come from
    AgentFEM.  The companion package only executes the provider.  A failed or
    unusable response never replaces the last accepted state.
    """

    if not isinstance(path, MaterialLoadingPath):
        raise TypeError("path must be an AgentFEM MaterialLoadingPath.")
    if path.control != "strain":
        raise NotImplementedError("Learned-material history currently requires strain control.")
    if path.temperature is not None:
        raise NotImplementedError(
            "Temperature histories require a provider declaring temperature input."
        )
    if not np.allclose(path.strain[0], 0.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("A fresh learned-material history must start at zero strain.")
    if not bool(getattr(material, "rate_independent", False)):
        raise NotImplementedError(
            "This history runner currently requires a rate-independent material."
        )

    schema = material.state_schema
    state = (
        schema.initial_state()
        if initial_state is None
        else schema.validate(initial_state, label="initial_state")
    )
    initial_increment = float(path.coordinate[1] - path.coordinate[0])
    initial_response = material.update(
        SmallStrainMaterialPointInput(
            strain_old=path.strain[0],
            strain_new=path.strain[0],
            time=float(path.coordinate[0]),
            time_increment=initial_increment,
            parameters=material.parameters,
            state_old=state,
            state_schema=schema,
            parameter_schema=material.parameter_schema,
        )
    )
    initial_response.require_usable()
    state = np.asarray(initial_response.state_new, dtype=float).copy()
    stress = [np.asarray(initial_response.cauchy_stress, dtype=float).copy()]
    states = [state]
    stored = [initial_response.stored_energy_density]
    dissipation = [initial_response.dissipation_density_increment]
    applicability = [initial_response.applicability]
    diagnostics: list[Mapping[str, object]] = [dict(initial_response.diagnostics)]
    old_strain = np.asarray(path.strain[0], dtype=float)

    for index in range(1, path.point_count):
        increment = float(path.coordinate[index] - path.coordinate[index - 1])
        response = material.update(
            SmallStrainMaterialPointInput(
                strain_old=old_strain,
                strain_new=path.strain[index],
                time=float(path.coordinate[index]),
                time_increment=increment,
                parameters=material.parameters,
                state_old=state,
                state_schema=schema,
                parameter_schema=material.parameter_schema,
            )
        )
        response.require_usable()
        state = np.asarray(response.state_new, dtype=float).copy()
        old_strain = np.asarray(path.strain[index], dtype=float)
        stress.append(np.asarray(response.cauchy_stress, dtype=float).copy())
        states.append(state)
        stored.append(response.stored_energy_density)
        dissipation.append(response.dissipation_density_increment)
        applicability.append(response.applicability)
        diagnostics.append(dict(response.diagnostics))

    result = LearnedMaterialHistory(
        path=path,
        stress=np.asarray(stress),
        state=np.asarray(states),
        stored_energy_density=(
            None if any(value is None for value in stored) else np.asarray(stored)
        ),
        dissipation_density_increment=(
            None
            if any(value is None for value in dissipation)
            else np.asarray(dissipation)
        ),
        applicability=tuple(applicability),
        diagnostics=tuple(diagnostics),
        state_schema=schema,
    )
    return result


__all__ = ["LearnedMaterialHistory", "run_small_strain_material_history"]
