"""Auditable dataset refinement for learned scientific operators."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from agentfem import campaigns, datasets


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


@dataclass(frozen=True)
class ParameterPathAcquisitionPlan:
    """New parameter values selected for trusted high-fidelity evaluation."""

    parameter: str
    values: tuple[float, ...]
    scores: tuple[float, ...]
    source_intervals: tuple[tuple[float, float], ...] = ()
    strategy: str = "risk_diverse_candidates"
    reason: str = "reference_or_model_disagreement"
    source_claim: str | None = None

    def __post_init__(self) -> None:
        parameter = str(self.parameter).strip()
        if not parameter:
            raise ValueError("Acquisition parameter must not be empty.")
        if len(self.values) != len(self.scores):
            raise ValueError("Acquisition values and scores must align.")
        if self.source_intervals and len(self.source_intervals) != len(self.values):
            raise ValueError("Acquisition source intervals must align with values.")
        if len(set(self.values)) != len(self.values):
            raise ValueError("Acquisition values must be unique.")
        if not np.isfinite(self.values).all() or not np.isfinite(self.scores).all():
            raise ValueError("Acquisition values and scores must be finite.")
        if any(score < 0.0 for score in self.scores):
            raise ValueError("Acquisition scores must be non-negative.")
        object.__setattr__(self, "parameter", parameter)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.summary(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def summary(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result = {
            "kind": "parameter_path_acquisition_plan",
            "parameter": self.parameter,
            "values": self.values,
            "scores": self.scores,
            "source_intervals": self.source_intervals,
            "strategy": self.strategy,
            "reason": self.reason,
            "source_claim": self.source_claim,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def sampling_plan(
        self,
        parameter_space,
        *,
        fixed_parameters: Mapping[str, object] | None = None,
    ):
        """Lower this plan to AgentFEM's ordinary explicit Campaign sampling."""

        if not isinstance(parameter_space, campaigns.ParameterSpace):
            raise TypeError("parameter_space must be an AgentFEM ParameterSpace.")
        if self.parameter not in parameter_space.names:
            raise ValueError(
                f"Parameter space has no acquisition parameter {self.parameter!r}."
            )
        fixed = dict(fixed_parameters or {})
        expected_fixed = set(parameter_space.names).difference({self.parameter})
        if set(fixed) != expected_fixed:
            raise ValueError(
                "fixed_parameters must define every non-acquired parameter exactly; "
                f"expected={tuple(sorted(expected_fixed))!r}."
            )
        if not self.values:
            raise ValueError("An empty acquisition plan cannot create a SamplingPlan.")
        samples = tuple({**fixed, self.parameter: value} for value in self.values)
        return campaigns.explicit(
            parameter_space,
            samples,
            metadata={
                "source": "agentfem-learning",
                "acquisition_plan": self.summary(),
            },
        )


@dataclass(frozen=True)
class ParameterPathDatasetAcquisition:
    """Validated high-fidelity acquisition merged into an operator dataset."""

    plan: ParameterPathAcquisitionPlan
    training_dataset: object
    acquired_case_ids: tuple[str, ...]
    training_fingerprint_before: str
    acquisition_fingerprint: str

    def summary(self) -> dict[str, object]:
        return {
            "kind": "parameter_path_dataset_acquisition",
            "plan": self.plan.summary(),
            "acquired_case_ids": self.acquired_case_ids,
            "training_fingerprint_before": self.training_fingerprint_before,
            "training_fingerprint_after": self.training_dataset.fingerprint,
            "acquisition_fingerprint": self.acquisition_fingerprint,
            "training_case_count": self.training_dataset.case_count,
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
            key: np.concatenate((base.coordinates[key], additions.coordinates[key]), axis=0)
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
        raise ValueError(f"Validation dataset has no refinement parameter {plan.parameter!r}.")
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
    if len(plan.case_ids) != len(plan.values) or len(set(plan.case_ids)) != len(plan.case_ids):
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
        [index for index in range(validation_dataset.case_count) if index not in selected_set],
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
        atol=64.0 * np.finfo(float).eps * max(*(abs(item) for item in path_range_before), 1.0),
    ):
        raise ValueError(
            "Refinement would remove a path endpoint and shrink the audited domain; "
            "evaluate a denser path or keep boundary cases independent."
        )

    additions = validation_dataset.subset(selected, name=f"{validation_dataset.name}_promoted")
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


def parameter_candidate_acquisition_plan(
    parameter: str,
    values,
    scores,
    *,
    existing_values=(),
    maximum_candidates: int = 3,
    minimum_spacing: float = 0.0,
    source_intervals=(),
    strategy: str = "risk_diverse_candidates",
    reason: str = "reference_or_model_disagreement",
    source_claim: str | None = None,
) -> ParameterPathAcquisitionPlan:
    """Rank unlabelled candidates by risk while retaining parameter diversity."""

    parameter_name = str(parameter).strip()
    candidates = np.asarray(tuple(values), dtype=float).reshape(-1)
    risks = np.asarray(tuple(scores), dtype=float).reshape(-1)
    existing = np.asarray(tuple(existing_values), dtype=float).reshape(-1)
    intervals = tuple(tuple(float(item) for item in pair) for pair in source_intervals)
    count = int(maximum_candidates)
    spacing = float(minimum_spacing)
    if not parameter_name:
        raise ValueError("parameter must not be empty.")
    if candidates.size != risks.size:
        raise ValueError("Candidate values and scores must align.")
    if intervals and len(intervals) != candidates.size:
        raise ValueError("Candidate source intervals must align with values.")
    if count < 1:
        raise ValueError("maximum_candidates must be positive.")
    if not np.isfinite(spacing) or spacing < 0.0:
        raise ValueError("minimum_spacing must be finite and non-negative.")
    if (
        not np.isfinite(candidates).all()
        or not np.isfinite(risks).all()
        or not np.isfinite(existing).all()
        or np.any(risks < 0.0)
    ):
        raise ValueError("Candidate values, scores, and existing values must be finite.")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("Candidate parameter values must be unique.")
    if not candidates.size:
        return ParameterPathAcquisitionPlan(
            parameter=parameter_name,
            values=(),
            scores=(),
            strategy=strategy,
            reason=reason,
            source_claim=source_claim,
        )

    span = max(float(np.ptp(np.concatenate((candidates, existing)))), np.finfo(float).eps)
    scale = max(float(np.max(np.abs(candidates))), 1.0)
    tolerance = max(spacing, 64.0 * np.finfo(float).eps * scale)
    remaining = [
        index
        for index, value in enumerate(candidates)
        if not existing.size or float(np.min(np.abs(existing - value))) > tolerance
    ]
    selected: list[int] = []
    anchors = [float(value) for value in existing]
    while remaining and len(selected) < count:
        weighted = []
        for index in remaining:
            distance = (
                min(abs(float(candidates[index]) - anchor) for anchor in anchors)
                if anchors
                else span
            )
            diversity = min(distance / span, 1.0)
            weighted.append(float(risks[index]) * np.sqrt(max(diversity, 0.05)))
        chosen = remaining.pop(int(np.argmax(weighted)))
        selected.append(chosen)
        anchors.append(float(candidates[chosen]))
        remaining = [
            index
            for index in remaining
            if abs(float(candidates[index]) - float(candidates[chosen])) > tolerance
        ]
    return ParameterPathAcquisitionPlan(
        parameter=parameter_name,
        values=tuple(float(candidates[index]) for index in selected),
        scores=tuple(float(risks[index]) for index in selected),
        source_intervals=tuple(intervals[index] for index in selected) if intervals else (),
        strategy=strategy,
        reason=reason,
        source_claim=source_claim,
    )


def merge_parameter_path_acquisition(
    training_dataset,
    acquired_dataset,
    plan: ParameterPathAcquisitionPlan,
    *,
    name: str | None = None,
) -> ParameterPathDatasetAcquisition:
    """Validate trusted acquired cases against a plan before merging them."""

    if not isinstance(plan, ParameterPathAcquisitionPlan):
        raise TypeError("plan must be a ParameterPathAcquisitionPlan.")
    if not isinstance(acquired_dataset, datasets.ScientificFieldDataset):
        raise TypeError("acquired_dataset must be a ScientificFieldDataset.")
    if acquired_dataset.case_count != len(plan.values):
        raise ValueError("Acquired case count does not match the acquisition plan.")
    if plan.parameter not in acquired_dataset.parameters:
        raise ValueError(f"Acquired dataset has no planned parameter {plan.parameter!r}.")
    values = np.asarray(acquired_dataset.parameters[plan.parameter], dtype=float).reshape(
        acquired_dataset.case_count, -1
    )
    if values.shape[1] != 1 or not np.isfinite(values).all():
        raise ValueError("Acquired parameter values must be finite scalars per case.")
    expected = np.sort(np.asarray(plan.values, dtype=float))
    actual = np.sort(values[:, 0])
    scale = max(float(np.max(np.abs(expected))), 1.0)
    if not np.allclose(
        actual,
        expected,
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps * scale,
    ):
        raise ValueError("Acquired parameter values do not match the acquisition plan.")
    merged = merge_operator_datasets(
        training_dataset,
        acquired_dataset,
        name=name or f"{training_dataset.name}_acquired",
    )
    return ParameterPathDatasetAcquisition(
        plan=plan,
        training_dataset=merged,
        acquired_case_ids=tuple(acquired_dataset.case_ids),
        training_fingerprint_before=training_dataset.fingerprint,
        acquisition_fingerprint=acquired_dataset.fingerprint,
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
                raise ValueError(f"Operator dataset {label} {key!r} has incompatible dtypes.")


__all__ = [
    "ParameterPathAcquisitionPlan",
    "ParameterPathDatasetAcquisition",
    "ParameterPathDatasetRefinement",
    "ParameterPathRefinementPlan",
    "apply_parameter_path_refinement",
    "merge_operator_datasets",
    "merge_parameter_path_acquisition",
    "parameter_candidate_acquisition_plan",
]
