# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral material-history execution for learned constitutive laws."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from agentfem.constitutive import (
    MaterialApplicabilityError,
    MaterialLoadingPath,
    SmallStrainMaterialPointInput,
)

_SYMMETRIC_COMPONENTS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (1, 2),
    (0, 2),
)


def _component_value(tensor: np.ndarray, component: tuple[int, int]) -> float:
    row, column = component
    return float(tensor[row, column])


def _set_component(
    tensor: np.ndarray,
    component: tuple[int, int],
    value: float,
) -> None:
    row, column = component
    tensor[row, column] = value
    tensor[column, row] = value


def _point_request(
    material,
    *,
    strain_old,
    strain_new,
    state_old,
    time,
    time_increment,
):
    return SmallStrainMaterialPointInput(
        strain_old=strain_old,
        strain_new=strain_new,
        time=float(time),
        time_increment=float(time_increment),
        parameters=material.parameters,
        state_old=state_old,
        state_schema=material.state_schema,
        parameter_schema=material.parameter_schema,
    )


def _mixed_control_update(
    material,
    *,
    strain_old,
    state_old,
    prescribed_strain,
    target_stress,
    strain_control,
    initial_guess,
    time,
    time_increment,
    tolerance,
    maximum_iterations,
):
    """Solve stress-controlled components without committing trial state."""

    unknown = [
        component
        for component in _SYMMETRIC_COMPONENTS
        if not bool(strain_control[component])
    ]
    candidate = np.asarray(initial_guess, dtype=float).copy()
    for component in _SYMMETRIC_COMPONENTS:
        if bool(strain_control[component]):
            _set_component(candidate, component, _component_value(prescribed_strain, component))
    reference_scale = max(
        1.0,
        float(material.parameters.get("yield_stress", 1.0)),
        float(np.max(np.abs(target_stress), initial=0.0)),
    )
    component_index = {component: index for index, component in enumerate(_SYMMETRIC_COMPONENTS)}

    def evaluate(strain):
        return material.update(
            _point_request(
                material,
                strain_old=strain_old,
                strain_new=strain,
                state_old=state_old,
                time=time,
                time_increment=time_increment,
            )
        ).require_usable()

    for _ in range(maximum_iterations):
        response = evaluate(candidate)
        scale = max(
            reference_scale,
            float(np.max(np.abs(response.cauchy_stress), initial=0.0)),
        )
        residual = np.asarray(
            [
                response.cauchy_stress[component] - target_stress[component]
                for component in unknown
            ],
            dtype=float,
        )
        residual_norm = float(np.max(np.abs(residual), initial=0.0))
        if residual_norm <= tolerance * scale:
            return candidate.copy(), response, residual_norm
        indices = [component_index[component] for component in unknown]
        jacobian = np.asarray(response.consistent_tangent)[np.ix_(indices, indices)]
        try:
            correction = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Learned-material mixed control produced a singular local Jacobian."
            ) from exc
        for backtrack in range(36):
            trial = candidate.copy()
            factor = 0.5**backtrack
            for component, increment in zip(unknown, correction, strict=True):
                _set_component(
                    trial,
                    component,
                    _component_value(trial, component) + factor * float(increment),
                )
            trial_response = evaluate(trial)
            trial_residual = np.asarray(
                [
                    trial_response.cauchy_stress[component]
                    - target_stress[component]
                    for component in unknown
                ],
                dtype=float,
            )
            if float(np.max(np.abs(trial_residual), initial=0.0)) < residual_norm:
                candidate = trial
                break
        else:
            raise RuntimeError(
                "Learned-material mixed-control line search could not reduce the "
                "stress residual. Refine the loading path or review the model scope."
            )
    raise RuntimeError(
        "Learned-material mixed control did not converge within "
        f"{maximum_iterations} iterations."
    )


@dataclass(frozen=True)
class LearnedMaterialHistory:
    """One transactional material path and its typed provider evidence."""

    path: MaterialLoadingPath
    stress: np.ndarray
    state: np.ndarray
    stored_energy_density: np.ndarray | None
    dissipation_density_increment: np.ndarray | None
    applicability: tuple[str, ...]
    diagnostics: tuple[Mapping[str, object], ...]
    state_schema: object
    completed: bool = True
    failure: Mapping[str, object] | None = None

    def state_variable(self, name: str) -> np.ndarray:
        """Return one named state variable without exposing packed offsets."""

        values = [self.state_schema.unpack(item)[name] for item in self.state]
        return np.asarray(values)

    @property
    def accepted(self) -> bool:
        return self.completed and all(
            value in {"in_domain", "warning"} for value in self.applicability
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem-learning.learned-material-history.v1",
            "path": self.path.summary(),
            "accepted": self.accepted,
            "completed": self.completed,
            "point_count": self.path.point_count,
            "applicability_counts": {
                status: self.applicability.count(status)
                for status in sorted(set(self.applicability))
            },
            "state_schema": self.state_schema.summary(),
            "failure": None if self.failure is None else dict(self.failure),
        }


def run_small_strain_material_history(
    material,
    path: MaterialLoadingPath,
    *,
    initial_state=None,
    control_tolerance: float = 1.0e-10,
    control_maximum_iterations: int = 30,
    failure_policy: str = "raise",
) -> LearnedMaterialHistory:
    """Run a learned material on an explicit path with atomic state commits.

    The path contract, tensor conventions and named state schema come from
    AgentFEM.  The companion package only executes the provider.  A failed or
    unusable response never replaces the last accepted state.
    """

    if not isinstance(path, MaterialLoadingPath):
        raise TypeError("path must be an AgentFEM MaterialLoadingPath.")
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
    failure_policy = str(failure_policy).strip().lower()
    if failure_policy not in {"raise", "return_partial"}:
        raise ValueError("failure_policy must be 'raise' or 'return_partial'.")

    schema = material.state_schema
    state = (
        schema.initial_state()
        if initial_state is None
        else schema.validate(initial_state, label="initial_state")
    )
    tolerance = float(control_tolerance)
    maximum_iterations = int(control_maximum_iterations)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("control_tolerance must be finite and positive.")
    if maximum_iterations < 1:
        raise ValueError("control_maximum_iterations must be positive.")

    initial_increment = float(path.coordinate[1] - path.coordinate[0])
    initial_response = material.update(
        _point_request(
            material,
            strain_old=path.strain[0],
            strain_new=path.strain[0],
            time=path.coordinate[0],
            time_increment=initial_increment,
            state_old=state,
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
    accepted_strain = [old_strain.copy()]
    maximum_control_residual = 0.0
    failure = None

    for index in range(1, path.point_count):
        increment = float(path.coordinate[index] - path.coordinate[index - 1])
        try:
            if path.control == "strain":
                new_strain = np.asarray(path.strain[index], dtype=float)
                response = material.update(
                    _point_request(
                        material,
                        strain_old=old_strain,
                        strain_new=new_strain,
                        time=path.coordinate[index],
                        time_increment=increment,
                        state_old=state,
                    )
                )
            else:
                new_strain, response, control_residual = _mixed_control_update(
                    material,
                    strain_old=old_strain,
                    state_old=state,
                    prescribed_strain=path.strain[index],
                    target_stress=path.stress[index],
                    strain_control=path.strain_control,
                    initial_guess=old_strain,
                    time=path.coordinate[index],
                    time_increment=increment,
                    tolerance=tolerance,
                    maximum_iterations=maximum_iterations,
                )
                maximum_control_residual = max(maximum_control_residual, control_residual)
            response.require_usable()
        except MaterialApplicabilityError as exc:
            if failure_policy == "raise" or len(accepted_strain) < 2:
                raise
            failure = {
                "kind": "material_applicability",
                "step_index": index,
                "coordinate": float(path.coordinate[index]),
                "error": str(exc),
                "state_commit": "failed_trial_not_committed",
            }
            break
        state = np.asarray(response.state_new, dtype=float).copy()
        old_strain = np.asarray(new_strain, dtype=float).copy()
        accepted_strain.append(old_strain)
        stress.append(np.asarray(response.cauchy_stress, dtype=float).copy())
        states.append(state)
        stored.append(response.stored_energy_density)
        dissipation.append(response.dissipation_density_increment)
        applicability.append(response.applicability)
        diagnostics.append(dict(response.diagnostics))

    accepted_count = len(accepted_strain)
    accepted_path = MaterialLoadingPath(
        coordinate=path.coordinate[:accepted_count],
        strain=np.asarray(accepted_strain),
        temperature=(
            None if path.temperature is None else path.temperature[:accepted_count]
        ),
        stress=None if path.stress is None else path.stress[:accepted_count],
        strain_control=path.strain_control,
        name=path.name,
        coordinate_name=path.coordinate_name,
        coordinate_unit=path.coordinate_unit,
    )
    if path.control != "strain":
        diagnostics[-1] = {
            **diagnostics[-1],
            "maximum_stress_control_residual": maximum_control_residual,
        }
    result = LearnedMaterialHistory(
        path=accepted_path,
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
        completed=failure is None,
        failure=failure,
    )
    return result


__all__ = ["LearnedMaterialHistory", "run_small_strain_material_history"]
