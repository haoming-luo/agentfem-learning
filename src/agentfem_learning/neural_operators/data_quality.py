"""Scientific information audits for supervised operator datasets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorDatasetAudit:
    """Identity and label-consistency evidence for one dataset partition."""

    case_count: int
    unique_input_count: int
    duplicate_input_count: int
    duplicate_input_fraction: float
    largest_duplicate_group: int
    conflicting_input_count: int
    maximum_duplicate_output_relative_l2: float
    input_fingerprints: tuple[str, ...]

    def summary(self) -> dict[str, int | float]:
        """Return bounded metrics without exposing case-level fingerprints."""

        return {
            "case_count": self.case_count,
            "unique_input_count": self.unique_input_count,
            "duplicate_input_count": self.duplicate_input_count,
            "duplicate_input_fraction": self.duplicate_input_fraction,
            "largest_duplicate_group": self.largest_duplicate_group,
            "conflicting_input_count": self.conflicting_input_count,
            "maximum_duplicate_output_relative_l2": (
                self.maximum_duplicate_output_relative_l2
            ),
        }


def audit_operator_dataset(
    *,
    inputs: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    case_ids: Sequence[str],
    conflict_tolerance: float = 1.0e-10,
) -> OperatorDatasetAudit:
    """Audit independent input mappings and contradictory duplicate labels.

    Arrays must have one leading entry per case. Input identity is exact after
    numerical arrays have been represented canonically. Output consistency is
    measured with a scale-aware relative L2 norm so harmless roundoff is not
    mistaken for a contradictory deterministic operator mapping.
    """

    identifiers = tuple(str(item) for item in case_ids)
    if not identifiers:
        raise ValueError("Operator dataset audits require at least one case.")
    tolerance = float(conflict_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("conflict_tolerance must be finite and non-negative.")
    input_arrays = _case_arrays(inputs, len(identifiers), label="inputs")
    output_arrays = _case_arrays(outputs, len(identifiers), label="outputs")
    fingerprints = tuple(
        _case_fingerprint(input_arrays, case) for case in range(len(identifiers))
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        groups[fingerprint].append(index)

    conflicts = 0
    maximum_difference = 0.0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        reference = indices[0]
        group_conflict = False
        for candidate in indices[1:]:
            difference = _relative_output_difference(
                output_arrays, reference, candidate
            )
            maximum_difference = max(maximum_difference, difference)
            group_conflict = group_conflict or difference > tolerance
        conflicts += int(group_conflict)

    unique_count = len(groups)
    duplicate_count = len(identifiers) - unique_count
    return OperatorDatasetAudit(
        case_count=len(identifiers),
        unique_input_count=unique_count,
        duplicate_input_count=duplicate_count,
        duplicate_input_fraction=duplicate_count / len(identifiers),
        largest_duplicate_group=max(len(indices) for indices in groups.values()),
        conflicting_input_count=conflicts,
        maximum_duplicate_output_relative_l2=maximum_difference,
        input_fingerprints=fingerprints,
    )


def partition_input_overlap(
    left: OperatorDatasetAudit, right: OperatorDatasetAudit
) -> int:
    """Count exact declared operator inputs shared by two partitions."""

    return len(set(left.input_fingerprints).intersection(right.input_fingerprints))


def require_consistent_operator_labels(
    audit: OperatorDatasetAudit, *, partition: str, provider: str
) -> None:
    """Reject one deterministic operator input mapped to conflicting labels."""

    if audit.conflicting_input_count:
        raise ValueError(
            f"{provider} {partition} data contain {audit.conflicting_input_count} declared "
            "input mapping(s) with contradictory outputs. Declare the missing physical "
            "input or parameter instead of asking one deterministic operator input to map "
            "to multiple labels."
        )


def require_independent_operator_partitions(
    training: OperatorDatasetAudit,
    other: OperatorDatasetAudit,
    *,
    right_name: str,
    provider: str,
) -> None:
    """Reject exact physical-input replicas across evidence partitions."""

    overlap = partition_input_overlap(training, other)
    if overlap:
        raise ValueError(
            f"{provider} training and {right_name} partitions share {overlap} exact "
            "declared operator input(s). Independent evidence must not contain "
            "training-input replicas."
        )


def operator_audit_metrics(
    prefix: str, audit: OperatorDatasetAudit
) -> dict[str, float]:
    """Flatten one audit into SimulationResult-compatible scalar quantities."""

    return {
        f"{prefix}_case_count": float(audit.case_count),
        f"{prefix}_unique_operator_input_count": float(audit.unique_input_count),
        f"{prefix}_duplicate_operator_input_count": float(audit.duplicate_input_count),
        f"{prefix}_duplicate_operator_input_fraction": audit.duplicate_input_fraction,
        f"{prefix}_largest_duplicate_operator_input_group": float(
            audit.largest_duplicate_group
        ),
        f"{prefix}_maximum_duplicate_output_relative_l2": (
            audit.maximum_duplicate_output_relative_l2
        ),
    }


def grouped_validation_indices(
    fingerprints: Sequence[str], *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split cases while keeping repeated physical inputs in one partition."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        groups[str(fingerprint)].append(index)
    if len(groups) < 2:
        raise ValueError(
            "Automatic validation requires at least two unique declared operator inputs; "
            "repeated files do not create independent validation evidence."
        )
    validation = float(validation_fraction)
    if not 0.0 < validation < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    keys = tuple(groups)
    order = np.random.default_rng(int(seed)).permutation(len(keys))
    validation_group_count = min(
        max(1, round(len(keys) * validation)), len(keys) - 1
    )
    selected = {keys[int(index)] for index in order[:validation_group_count]}
    validation_indices = np.asarray(
        [index for key in keys if key in selected for index in groups[key]], dtype=int
    )
    training_indices = np.asarray(
        [index for key in keys if key not in selected for index in groups[key]], dtype=int
    )
    return training_indices, validation_indices


def _case_arrays(values, case_count: int, *, label: str) -> tuple[tuple[str, np.ndarray], ...]:
    if not values:
        raise ValueError(f"Operator dataset {label} must not be empty.")
    arrays = []
    for name in sorted(values):
        array = np.asarray(values[name])
        if array.ndim < 1 or array.shape[0] != case_count:
            raise ValueError(
                f"Operator dataset {label} array {name!r} must have {case_count} cases."
            )
        if array.dtype.kind not in "biufc":
            raise TypeError(
                f"Operator dataset {label} array {name!r} must be numerical."
            )
        canonical = _canonical_numeric_array(array)
        if not np.isfinite(canonical).all():
            raise ValueError(
                f"Operator dataset {label} array {name!r} contains non-finite values."
            )
        arrays.append((str(name), canonical))
    return tuple(arrays)


def _canonical_numeric_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind == "b":
        dtype = np.uint8
    elif array.dtype.kind == "u":
        dtype = np.uint64
    elif array.dtype.kind == "i":
        dtype = np.int64
    elif array.dtype.kind == "c":
        dtype = np.complex128
    else:
        dtype = np.float64
    return np.asarray(array, dtype=dtype)


def _case_fingerprint(arrays, case: int) -> str:
    digest = hashlib.sha256()
    for name, array in arrays:
        values = np.ascontiguousarray(array[case])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _relative_output_difference(arrays, reference: int, candidate: int) -> float:
    numerator_squared = 0.0
    scale_squared = 0.0
    for _, array in arrays:
        left = array[reference].reshape(-1)
        right = array[candidate].reshape(-1)
        numerator_squared += float(np.vdot(left - right, left - right).real)
        scale_squared += max(
            float(np.vdot(left, left).real), float(np.vdot(right, right).real)
        )
    denominator = max(np.sqrt(scale_squared), np.finfo(float).eps)
    return float(np.sqrt(numerator_squared) / denominator)


__all__ = [
    "OperatorDatasetAudit",
    "audit_operator_dataset",
    "grouped_validation_indices",
    "operator_audit_metrics",
    "partition_input_overlap",
    "require_consistent_operator_labels",
    "require_independent_operator_partitions",
]
