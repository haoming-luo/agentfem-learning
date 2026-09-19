"""Auditable dataset refinement for learned scientific operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from agentfem import datasets


@dataclass(frozen=True)
class ParameterPathRefinementPlan:
    """Bounded candidate set selected from failed independent path evidence."""

    parameter: str
    values: tuple[float, ...]
    case_ids: tuple[str, ...]
    risks: tuple[float, ...]
    strategy: str = "risk_diversity"
    reason: str = "failed_path_evidence"

    def summary(self) -> dict[str, object]:
        return {
            "kind": "parameter_path_refinement_plan",
            "parameter": self.parameter,
            "values": self.values,
            "case_ids": self.case_ids,
            "risks": self.risks,
            "strategy": self.strategy,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ParameterPathDatasetRefinement:
    """One bounded promotion of failed path cases into the training partition."""

    plan: ParameterPathRefinementPlan
    training_dataset: object
    validation_dataset: object
    added_case_ids: tuple[str, ...]
    training_fingerprint_before: str
    validation_fingerprint_before: str
    validation_parameter_range_before: tuple[float, float]
    validation_parameter_range_after: tuple[float, float]

    def summary(self) -> dict[str, object]:
        return {
            "kind": "parameter_path_dataset_refinement",
            "plan": self.plan.summary(),
            "added_case_ids": self.added_case_ids,
            "training_fingerprint_before": self.training_fingerprint_before,
            "training_fingerprint_after": self.training_dataset.fingerprint,
            "validation_fingerprint_before": self.validation_fingerprint_before,
            "validation_fingerprint_after": self.validation_dataset.fingerprint,
            "validation_parameter_range_before": self.validation_parameter_range_before,
            "validation_parameter_range_after": self.validation_parameter_range_after,
            "training_case_count": self.training_dataset.case_count,
            "remaining_validation_case_count": self.validation_dataset.case_count,
        }


def merge_operator_datasets(base, additions, *, name: str | None = None):
    """Append compatible field cases without weakening their scientific schema."""

    if not isinstance(base, datasets.ScientificFieldDataset) or not isinstance(
        additions, datasets.ScientificFieldDataset
    ):
        raise TypeError("Operator refinement requires ScientificFieldDataset inputs.")
    _require_compatible_dataset_schema(base, additions)
    overlap = set(base.case_ids).intersection(additions.case_ids)
    if overlap:
        raise ValueError(
            "Operator datasets cannot merge duplicate case IDs; "
            f"overlap={tuple(sorted(overlap))!r}."
        )
    return datasets.ScientificFieldDataset(
        case_ids=(*base.case_ids, *additions.case_ids),
        encodings=base.encodings,
        fields={
            key: np.concatenate((base.fields[key], additions.fields[key]), axis=0)
            for key in base.fields
        },
        coordinates={
            key: np.concatenate(
                (base.coordinates[key], additions.coordinates[key]), axis=0
            )
            for key in base.coordinates
        },
        parameters={
            key: np.concatenate((base.parameters[key], additions.parameters[key]), axis=0)
            for key in base.parameters
        },
        masks={
            key: np.concatenate((base.masks[key], additions.masks[key]), axis=0)
            for key in base.masks
        },
        case_metadata=(*base.case_metadata, *additions.case_metadata),
        name=str(name or f"{base.name}_refined"),
        metadata=base.metadata,
    )


def apply_parameter_path_refinement(
    training_dataset,
    validation_dataset,
    plan: ParameterPathRefinementPlan,
    *,
    minimum_remaining_cases: int = 3,
    preserve_path_bounds: bool = True,
) -> ParameterPathDatasetRefinement:
    """Move selected trusted path cases into training and preserve a held-out path.

    The selected cases already carry independent reference fields. They are
    removed from validation before being appended to training, so the next
    training run cannot evaluate itself on promoted cases. A separate trusted
    solver can produce an additions dataset and use :func:`merge_operator_datasets`
    directly when the requested values have not yet been evaluated.
    """

    if not isinstance(plan, ParameterPathRefinementPlan):
        raise TypeError("plan must be a ParameterPathRefinementPlan.")
    if not isinstance(training_dataset, datasets.ScientificFieldDataset) or not isinstance(
        validation_dataset, datasets.ScientificFieldDataset
    ):
        raise TypeError("Path refinement requires ScientificFieldDataset partitions.")
    minimum = int(minimum_remaining_cases)
    if minimum < 3:
        raise ValueError("minimum_remaining_cases must be at least three for a path audit.")
    if plan.parameter not in validation_dataset.parameters:
        raise ValueError(
            f"Validation dataset has no refinement parameter {plan.parameter!r}."
        )
    parameter_values = np.asarray(
        validation_dataset.parameters[plan.parameter], dtype=float
    ).reshape(validation_dataset.case_count, -1)
    if parameter_values.shape[1] != 1:
        raise ValueError("Parameter-path refinement requires one scalar value per case.")
    if not np.isfinite(parameter_values).all():
        raise ValueError("Parameter-path refinement values must be finite.")
    if np.unique(parameter_values[:, 0]).size != validation_dataset.case_count:
        raise ValueError("Validation path parameter values must be unique.")
    path_range_before = (
        float(np.min(parameter_values[:, 0])),
        float(np.max(parameter_values[:, 0])),
    )
    if not plan.case_ids:
        return ParameterPathDatasetRefinement(
            plan=plan,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            added_case_ids=(),
            training_fingerprint_before=training_dataset.fingerprint,
            validation_fingerprint_before=validation_dataset.fingerprint,
            validation_parameter_range_before=path_range_before,
            validation_parameter_range_after=path_range_before,
        )
    if len(plan.case_ids) != len(plan.values) or len(set(plan.case_ids)) != len(
        plan.case_ids
    ):
        raise ValueError("Refinement plan case IDs and values must be unique and aligned.")

    locations = {case_id: index for index, case_id in enumerate(validation_dataset.case_ids)}
    missing = tuple(case_id for case_id in plan.case_ids if case_id not in locations)
    if missing:
        raise ValueError(f"Refinement cases are absent from validation: {missing!r}.")
    selected = np.asarray([locations[case_id] for case_id in plan.case_ids], dtype=int)
    actual_values = parameter_values[selected, 0]
    expected_values = np.asarray(plan.values, dtype=float)
    scale = max(float(np.max(np.abs(expected_values))), 1.0)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps * scale,
    ):
        raise ValueError("Refinement plan values do not match their validation case IDs.")

    selected_set = {int(index) for index in selected}
    remaining = np.asarray(
        [
            index
            for index in range(validation_dataset.case_count)
            if index not in selected_set
        ],
        dtype=int,
    )
    if remaining.size < minimum:
        raise ValueError(
            "Refinement would leave too few independent path cases; reduce the candidate "
            "count or evaluate a denser path."
        )
    remaining_values = parameter_values[remaining, 0]
    if np.unique(remaining_values).size != remaining_values.size:
        raise ValueError("Remaining path parameter values must stay unique.")
    path_range_after = (
        float(np.min(remaining_values)),
        float(np.max(remaining_values)),
    )
    if preserve_path_bounds and not np.allclose(
        path_range_after,
        path_range_before,
        rtol=0.0,
        atol=64.0
        * np.finfo(float).eps
        * max(*(abs(item) for item in path_range_before), 1.0),
    ):
        raise ValueError(
            "Refinement would remove a path endpoint and shrink the audited domain; "
            "evaluate a denser path or keep boundary cases independent."
        )

    additions = validation_dataset.subset(
        selected, name=f"{validation_dataset.name}_promoted"
    )
    remaining_validation = validation_dataset.subset(
        remaining, name=f"{validation_dataset.name}_remaining"
    )
    refined_training = merge_operator_datasets(
        training_dataset,
        additions,
        name=f"{training_dataset.name}_refined",
    )
    return ParameterPathDatasetRefinement(
        plan=plan,
        training_dataset=refined_training,
        validation_dataset=remaining_validation,
        added_case_ids=tuple(plan.case_ids),
        training_fingerprint_before=training_dataset.fingerprint,
        validation_fingerprint_before=validation_dataset.fingerprint,
        validation_parameter_range_before=path_range_before,
        validation_parameter_range_after=path_range_after,
    )


def _require_compatible_dataset_schema(left, right) -> None:
    if tuple(left.encodings) != tuple(right.encodings):
        raise ValueError("Operator datasets must have identical field encodings.")
    if dict(left.metadata) != dict(right.metadata):
        raise ValueError("Operator datasets must have identical scientific metadata.")
    for label, left_values, right_values in (
        ("field", left.fields, right.fields),
        ("coordinate", left.coordinates, right.coordinates),
        ("parameter", left.parameters, right.parameters),
        ("mask", left.masks, right.masks),
    ):
        if set(left_values) != set(right_values):
            raise ValueError(f"Operator dataset {label} names must match exactly.")
        for key in left_values:
            if left_values[key].shape[1:] != right_values[key].shape[1:]:
                raise ValueError(
                    f"Operator dataset {label} {key!r} has incompatible per-case shapes."
                )
            if left_values[key].dtype != right_values[key].dtype:
                raise ValueError(
                    f"Operator dataset {label} {key!r} has incompatible dtypes."
                )


__all__ = [
    "ParameterPathDatasetRefinement",
    "ParameterPathRefinementPlan",
    "apply_parameter_path_refinement",
    "merge_operator_datasets",
]
