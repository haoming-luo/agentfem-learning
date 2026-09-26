# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible non-proportional material-path evidence for DENIM v1."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
from agentfem.constitutive import material_strain_path

from ..history import run_small_strain_material_history

PATH_IDENTITY = "abaqus-inspired-tension-torsion-circle@1"
PATH_REFERENCE = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMACAEBMKRefMap/simabmk-c-cyclictests.htm"
)


def tension_torsion_circle(
    *,
    amplitude: float = 0.004,
    cycles: int = 1,
    segments_per_cycle: int = 16,
):
    """Return a zero-ramp plus isochoric tension-torsion circular path.

    The path follows the public Abaqus non-proportional test topology but uses
    the fixed DENIM material parameters.  Consequently the Abaqus copper
    stresses are not treated as numerical reference values.
    """

    amplitude = float(amplitude)
    cycles = int(cycles)
    segments = int(segments_per_cycle)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive.")
    if cycles < 1 or segments < 8 or segments % 4:
        raise ValueError("cycles must be positive and segments_per_cycle a multiple of 4 >= 8.")

    coordinate = [0.0, 1.0]
    strain = [np.zeros((3, 3)), _strain_on_circle(amplitude, 0.0)]
    for index in range(1, cycles * segments + 1):
        theta = 2.0 * pi * index / segments
        coordinate.append(1.0 + index / segments)
        strain.append(_strain_on_circle(amplitude, theta))
    return material_strain_path(
        np.asarray(coordinate),
        np.asarray(strain),
        name=PATH_IDENTITY,
        coordinate_name="normalized_time",
        coordinate_unit="1",
    )


def _strain_on_circle(amplitude: float, theta: float) -> np.ndarray:
    axial = amplitude * np.cos(theta)
    tensor_shear = 0.5 * sqrt(3.0) * amplitude * np.sin(theta)
    return np.asarray(
        (
            (axial, tensor_shear, 0.0),
            (tensor_shear, -0.5 * axial, 0.0),
            (0.0, 0.0, -0.5 * axial),
        )
    )


def rotate_path(path, rotation: np.ndarray):
    selected = np.asarray(rotation, dtype=float)
    if selected.shape != (3, 3) or not np.allclose(
        selected.T @ selected, np.eye(3), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("rotation must be an orthogonal 3x3 matrix.")
    rotated = np.einsum("ij,njk,lk->nil", selected, path.strain, selected)
    return material_strain_path(
        path.coordinate,
        rotated,
        name=f"{path.name}_rotated",
        coordinate_name=path.coordinate_name,
        coordinate_unit=path.coordinate_unit,
    )


def _relative_history_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return float(np.linalg.norm(candidate - reference) / scale)


def run_nonproportional_validation(material) -> dict[str, object]:
    """Run nested increment refinement and rigid-rotation covariance checks."""

    base_path = tension_torsion_circle()
    medium_path = base_path.refine(2)
    fine_path = base_path.refine(4)
    base = run_small_strain_material_history(material, base_path)
    medium = run_small_strain_material_history(material, medium_path)
    fine = run_small_strain_material_history(material, fine_path)

    # Both refined paths describe the same piecewise-linear continuous path.
    # Every other fine point is therefore exactly one medium physical knot.
    fine_at_medium = fine.stress[::2]
    fine_peeq_at_medium = fine.state_variable("peeq")[::2]
    medium_peeq = medium.state_variable("peeq")

    rotation = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    rotated_path = rotate_path(medium_path, rotation)
    rotated = run_small_strain_material_history(material, rotated_path)
    rotated_back = np.einsum("ji,njk,kl->nil", rotation, rotated.stress, rotation)

    yield_residual = max(
        (abs(float(item.get("yield_residual", 0.0))) for item in fine.diagnostics),
        default=0.0,
    )
    plastic_trace = max(
        (abs(float(item.get("maximum_plastic_trace", 0.0))) for item in fine.diagnostics),
        default=0.0,
    )
    fine_peeq = fine.state_variable("peeq")
    return {
        "schema": "agentfem-learning.denim-nonproportional-validation.v1",
        "path": {
            "identity": PATH_IDENTITY,
            "reference": PATH_REFERENCE,
            "reference_role": "path_topology_only_not_numerical_calibration",
            "amplitude": 0.004,
            "cycles": 1,
            "base_segments_per_cycle": 16,
            "base_fingerprint": base_path.fingerprint,
            "refinement_factors": [1, 2, 4],
            "point_counts": [
                base.path.point_count,
                medium.path.point_count,
                fine.path.point_count,
            ],
        },
        "maximum_absolute_stress_mpa": float(np.max(np.abs(fine.stress)) / 1.0e6),
        "final_peeq": float(fine_peeq[-1]),
        "minimum_peeq_increment": float(np.min(np.diff(fine_peeq))),
        "refinement": {
            "stress_relative_l2": _relative_history_error(fine_at_medium, medium.stress),
            "peeq_relative_l2": _relative_history_error(fine_peeq_at_medium, medium_peeq),
        },
        "rotation_covariance": {
            "stress_relative_l2": _relative_history_error(medium.stress, rotated_back),
            "peeq_relative_l2": _relative_history_error(
                medium_peeq, rotated.state_variable("peeq")
            ),
        },
        "physics": {
            "maximum_yield_residual_pa": yield_residual,
            "maximum_plastic_strain_trace": plastic_trace,
            "all_refinements_usable": base.accepted and medium.accepted and fine.accepted,
            "rotated_path_usable": rotated.accepted,
        },
    }


__all__ = [
    "PATH_IDENTITY",
    "PATH_REFERENCE",
    "rotate_path",
    "run_nonproportional_validation",
    "tension_torsion_circle",
]
