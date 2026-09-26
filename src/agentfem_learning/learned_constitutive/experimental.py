# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental uniaxial histories and model-to-data comparison contracts."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
from agentfem.constitutive import MaterialLoadingPath


def _readonly(values, *, ndim: int, name: str) -> np.ndarray:
    selected = np.asarray(values, dtype=float)
    if selected.ndim != ndim or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    selected = selected.copy()
    selected.setflags(write=False)
    return selected


def file_sha256(path: str | Path) -> str:
    """Return the immutable SHA-256 identity of a local experimental file."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentalUniaxialHistory:
    """One measured stress--strain history with explicit source identity."""

    sample_id: str
    coordinate: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    source_identifier: str
    source_sha256: str
    license: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        source = str(self.source_identifier).strip()
        license_name = str(self.license).strip()
        digest = str(self.source_sha256).strip().lower()
        if not sample_id or not source or not license_name:
            raise ValueError("Sample ID, source identifier and license must be nonempty.")
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("source_sha256 must contain 64 lowercase hexadecimal characters.")
        coordinate = _readonly(self.coordinate, ndim=1, name="coordinate")
        strain = _readonly(self.strain, ndim=1, name="strain")
        stress = _readonly(self.stress, ndim=1, name="stress")
        if coordinate.size < 2 or strain.size != coordinate.size or stress.size != coordinate.size:
            raise ValueError("Experimental histories require matching arrays with at least 2 points.")
        if np.any(np.diff(coordinate) <= 0.0):
            raise ValueError("Experimental coordinates must increase strictly.")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "strain", strain)
        object.__setattr__(self, "stress", stress)
        object.__setattr__(self, "source_identifier", source)
        object.__setattr__(self, "source_sha256", digest)
        object.__setattr__(self, "license", license_name)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def point_count(self) -> int:
        return int(self.coordinate.size)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem-learning.experimental-uniaxial-history.v1",
            "sample_id": self.sample_id,
            "point_count": self.point_count,
            "source_identifier": self.source_identifier,
            "source_sha256": self.source_sha256,
            "license": self.license,
            "strain_range": [float(np.min(self.strain)), float(np.max(self.strain))],
            "stress_range_pa": [float(np.min(self.stress)), float(np.max(self.stress))],
            "metadata": dict(self.metadata),
        }


def read_uniaxial_csv(
    path: str | Path,
    *,
    sample_id: str,
    source_identifier: str,
    license: str,
    coordinate_column: str,
    strain_column: str,
    stress_column: str,
    strain_scale: float = 1.0,
    stress_scale: float = 1.0,
    zero_origin: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> ExperimentalUniaxialHistory:
    """Read a declared CSV mapping without guessing columns or units."""

    selected = Path(path).expanduser().resolve()
    rows = list(csv.DictReader(selected.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("Experimental CSV contains no data rows.")
    required = (coordinate_column, strain_column, stress_column)
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"Experimental CSV is missing columns: {missing!r}.")
    try:
        coordinate = np.asarray([float(row[coordinate_column]) for row in rows])
        strain = float(strain_scale) * np.asarray(
            [float(row[strain_column]) for row in rows]
        )
        stress = float(stress_scale) * np.asarray(
            [float(row[stress_column]) for row in rows]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Experimental CSV contains a missing or non-numeric value.") from exc
    offsets = {"strain": 0.0, "stress_pa": 0.0}
    if zero_origin:
        offsets = {"strain": float(strain[0]), "stress_pa": float(stress[0])}
        strain = strain - strain[0]
        stress = stress - stress[0]
    return ExperimentalUniaxialHistory(
        sample_id=sample_id,
        coordinate=coordinate,
        strain=strain,
        stress=stress,
        source_identifier=source_identifier,
        source_sha256=file_sha256(selected),
        license=license,
        metadata={**dict(metadata or {}), "zero_origin_offsets": offsets},
    )


def truncate_to_strain_domain(
    history: ExperimentalUniaxialHistory,
    maximum_absolute_strain: float,
) -> ExperimentalUniaxialHistory:
    """Take the longest chronological prefix inside a declared strain domain."""

    limit = float(maximum_absolute_strain)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum_absolute_strain must be finite and positive.")
    outside = np.flatnonzero(np.abs(history.strain) > limit)
    count = int(outside[0]) if outside.size else history.point_count
    if count < 2:
        raise ValueError("The experimental history leaves the strain domain immediately.")
    return ExperimentalUniaxialHistory(
        sample_id=history.sample_id,
        coordinate=history.coordinate[:count],
        strain=history.strain[:count],
        stress=history.stress[:count],
        source_identifier=history.source_identifier,
        source_sha256=history.source_sha256,
        license=history.license,
        metadata={
            **dict(history.metadata),
            "domain_truncation": {
                "maximum_absolute_strain": limit,
                "original_point_count": history.point_count,
            },
        },
    )


def prefix_uniaxial_history(
    history: ExperimentalUniaxialHistory,
    point_count: int,
) -> ExperimentalUniaxialHistory:
    """Return an explicit chronological prefix without changing source identity."""

    count = int(point_count)
    if count < 2 or count > history.point_count:
        raise ValueError("point_count must select at least 2 existing history points.")
    return ExperimentalUniaxialHistory(
        sample_id=history.sample_id,
        coordinate=history.coordinate[:count],
        strain=history.strain[:count],
        stress=history.stress[:count],
        source_identifier=history.source_identifier,
        source_sha256=history.source_sha256,
        license=history.license,
        metadata={
            **dict(history.metadata),
            "chronological_prefix": {
                "point_count": count,
                "original_point_count": history.point_count,
            },
        },
    )


def _rdp_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        norm = float(np.linalg.norm(segment))
        candidates = points[start + 1 : end]
        if norm <= np.finfo(float).eps:
            distances = np.linalg.norm(candidates - points[start], axis=1)
        else:
            relative = candidates - points[start]
            projection = np.outer(relative @ segment / (norm * norm), segment)
            distances = np.linalg.norm(relative - projection, axis=1)
        offset = int(np.argmax(distances))
        distance = float(distances[offset])
        if distance > tolerance:
            index = start + 1 + offset
            keep.add(index)
            stack.extend(((start, index), (index, end)))
    return np.asarray(sorted(keep), dtype=int)


def simplify_uniaxial_history(
    history: ExperimentalUniaxialHistory,
    *,
    maximum_points: int = 401,
    geometric_tolerance: float = 1.0e-4,
) -> ExperimentalUniaxialHistory:
    """Reduce a loop in normalized strain--stress space without smoothing it."""

    count = int(maximum_points)
    tolerance = float(geometric_tolerance)
    if count < 3:
        raise ValueError("maximum_points must be at least 3.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("geometric_tolerance must be finite and positive.")
    if history.point_count <= count:
        return history
    strain_scale = max(float(np.ptp(history.strain)), np.finfo(float).eps)
    stress_scale = max(float(np.ptp(history.stress)), np.finfo(float).eps)
    points = np.column_stack(
        (
            (history.strain - history.strain[0]) / strain_scale,
            (history.stress - history.stress[0]) / stress_scale,
        )
    )
    lower = tolerance
    indices = _rdp_indices(points, lower)
    if indices.size > count:
        upper = math.sqrt(2.0)
        for _ in range(40):
            trial = 0.5 * (lower + upper)
            selected = _rdp_indices(points, trial)
            if selected.size > count:
                lower = trial
            else:
                upper = trial
                indices = selected
    return ExperimentalUniaxialHistory(
        sample_id=history.sample_id,
        coordinate=history.coordinate[indices],
        strain=history.strain[indices],
        stress=history.stress[indices],
        source_identifier=history.source_identifier,
        source_sha256=history.source_sha256,
        license=history.license,
        metadata={
            **dict(history.metadata),
            "path_reduction": {
                "method": "normalized_strain_stress_rdp",
                "original_point_count": history.point_count,
                "maximum_points": count,
                "minimum_geometric_tolerance": tolerance,
            },
        },
    )


def uniaxial_stress_free_path(
    history: ExperimentalUniaxialHistory,
    *,
    axis: int = 0,
) -> MaterialLoadingPath:
    """Translate a lab coupon history to axial-strain/free-traction control."""

    axis = int(axis)
    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1 or 2.")
    strain = np.zeros((history.point_count, 3, 3), dtype=float)
    strain[:, axis, axis] = history.strain
    stress = np.zeros_like(strain)
    strain_control = np.zeros((3, 3), dtype=bool)
    strain_control[axis, axis] = True
    return MaterialLoadingPath(
        coordinate=history.coordinate,
        strain=strain,
        stress=stress,
        strain_control=strain_control,
        name=f"experimental:{history.sample_id}",
        coordinate_name="time",
        coordinate_unit="s",
    )


def uniaxial_stress_metrics(
    history: ExperimentalUniaxialHistory,
    predicted_stress_pa,
) -> dict[str, float | int | str]:
    """Compare stress histories and loop work without hiding scale choices."""

    predicted = np.asarray(predicted_stress_pa, dtype=float)
    if predicted.shape != history.stress.shape or not np.all(np.isfinite(predicted)):
        raise ValueError("predicted_stress_pa must match the finite measured history.")
    error = predicted - history.stress
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    stress_range = float(np.ptp(history.stress))
    measured_work = float(np.trapezoid(history.stress, history.strain))
    predicted_work = float(np.trapezoid(predicted, history.strain))
    work_scale = max(abs(measured_work), np.finfo(float).eps)
    return {
        "schema": "agentfem-learning.uniaxial-stress-metrics.v1",
        "sample_count": history.point_count,
        "stress_rmse_pa": rmse,
        "stress_mae_pa": mae,
        "stress_range_normalized_rmse": rmse / max(stress_range, np.finfo(float).eps),
        "maximum_absolute_stress_error_pa": float(np.max(np.abs(error))),
        "measured_signed_work_density_j_per_m3": measured_work,
        "predicted_signed_work_density_j_per_m3": predicted_work,
        "signed_work_relative_error": abs(predicted_work - measured_work) / work_scale,
    }


def validate_sample_partitions(
    available_sample_ids,
    *,
    fit,
    validation,
    held_out,
) -> dict[str, list[str]]:
    """Require nonempty, disjoint, known sample-level partitions."""

    available = {str(value) for value in available_sample_ids}
    result = {
        "fit": [str(value) for value in fit],
        "validation": [str(value) for value in validation],
        "held_out": [str(value) for value in held_out],
    }
    sets = {name: set(values) for name, values in result.items()}
    if any(not values or len(values) != len(sets[name]) for name, values in result.items()):
        raise ValueError("Every partition must contain unique sample IDs.")
    overlap = (sets["fit"] & sets["validation"]) | (sets["fit"] & sets["held_out"]) | (
        sets["validation"] & sets["held_out"]
    )
    if overlap:
        raise ValueError(f"Experimental sample partitions overlap: {sorted(overlap)!r}.")
    unknown = set().union(*sets.values()) - available
    if unknown:
        raise ValueError(f"Experimental partitions contain unknown samples: {sorted(unknown)!r}.")
    return result


__all__ = [
    "ExperimentalUniaxialHistory",
    "file_sha256",
    "prefix_uniaxial_history",
    "read_uniaxial_csv",
    "simplify_uniaxial_history",
    "truncate_to_strain_domain",
    "uniaxial_stress_free_path",
    "uniaxial_stress_metrics",
    "validate_sample_partitions",
]
