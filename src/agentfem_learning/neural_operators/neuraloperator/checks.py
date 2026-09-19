"""Named scientific checks for learned field operators."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
from agentfem import verification


@dataclass(frozen=True)
class OperatorCheckContext:
    """Physical held-out data made available to one reviewed check."""

    specification: object
    training_dataset: object
    validation_dataset: object
    predictions: Mapping[str, object]
    references: Mapping[str, object]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class OperatorCheck:
    """A named, fingerprinted evaluator that returns one verification claim.

    The name is part of ``NeuralOperatorSpec.required_checks``.  Keeping the
    callable behind this explicit record lets a project provide physics-aware
    boundary, balance, or applicability checks without teaching the generic
    trainer the equations of every analysis family.
    """

    name: str
    evaluator: Callable[[OperatorCheckContext], verification.VerificationClaim]
    version: str = "1"
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        version = str(self.version).strip()
        if not name or not version:
            raise ValueError("OperatorCheck name and version must not be empty.")
        if not callable(self.evaluator):
            raise TypeError("OperatorCheck.evaluator must be callable.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def evaluate(self, context: OperatorCheckContext) -> verification.VerificationClaim:
        claim = self.evaluator(context)
        if not isinstance(claim, verification.VerificationClaim):
            raise TypeError(f"Operator check {self.name!r} must return VerificationClaim.")
        if claim.name != self.name:
            raise ValueError(f"Operator check {self.name!r} returned claim {claim.name!r}.")
        return claim

    def summary(self) -> dict[str, object]:
        return {
            "kind": "operator_check",
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "evaluator": _callable_identity(self.evaluator),
            "metadata": dict(self.metadata),
        }


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


def _callable_identity(function) -> dict[str, object]:
    identity = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(
            function,
            "__qualname__",
            getattr(function, "__name__", None),
        ),
    }
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError):
        source = None
    if source is not None:
        identity["source_sha256"] = sha256(source).hexdigest()
    return identity


def parameter_path_reliability_check(
    parameter: str,
    *,
    maximum_relative_l2: float = 0.10,
    output_tolerances: Mapping[str, float] | None = None,
    maximum_spike_ratio: float = 3.0,
    name: str = "parameter_path_reliability",
    version: str = "1",
) -> OperatorCheck:
    """Return a held-out check for continuous interpolation along one parameter.

    The validation dataset must describe one path with at least three distinct
    parameter values.  Every predicted output is compared with its independent
    reference case by case.  Acceptance requires both the declared worst-case
    field-error limits and the absence of an isolated interior error spike.

    This check deliberately consumes physical held-out fields rather than
    optimizer loss.  It is useful for geometry, loading, material, or other
    scalar paths and makes no assumption about the operator architecture.
    """

    parameter_name = str(parameter).strip()
    check_name = str(name).strip()
    check_version = str(version).strip()
    default_tolerance = float(maximum_relative_l2)
    spike_limit = float(maximum_spike_ratio)
    tolerances = {
        str(key): float(value) for key, value in dict(output_tolerances or {}).items()
    }
    if not parameter_name:
        raise ValueError("parameter must not be empty.")
    if not check_name or not check_version:
        raise ValueError("name and version must not be empty.")
    if not np.isfinite(default_tolerance) or default_tolerance <= 0.0:
        raise ValueError("maximum_relative_l2 must be finite and positive.")
    if not np.isfinite(spike_limit) or spike_limit <= 1.0:
        raise ValueError("maximum_spike_ratio must be finite and greater than one.")
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances.values()):
        raise ValueError("Every output tolerance must be finite and positive.")

    def evaluate(context: OperatorCheckContext) -> verification.VerificationClaim:
        dataset = context.validation_dataset
        if dataset is None:
            raise ValueError(f"{check_name!r} requires an explicit validation_dataset.")
        parameters = getattr(dataset, "parameters", {})
        if parameter_name not in parameters:
            raise ValueError(f"Validation dataset has no parameter {parameter_name!r}.")
        values = np.asarray(parameters[parameter_name], dtype=float).reshape(-1)
        case_ids = tuple(str(item) for item in getattr(dataset, "case_ids", ()))
        if values.size < 3:
            raise ValueError("A parameter-path check requires at least three cases.")
        if len(case_ids) != values.size:
            raise ValueError("Parameter values and validation case IDs must align.")
        if not np.isfinite(values).all():
            raise ValueError("Parameter-path values must be finite.")
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        if np.any(np.diff(sorted_values) <= 0.0):
            raise ValueError(
                "Parameter-path values must be unique so interior spikes are defined."
            )

        available = tuple(key for key in context.predictions if key in context.references)
        selected = tuple(tolerances) if tolerances else available
        if not selected:
            raise ValueError("No predicted/reference output pairs are available.")
        missing = tuple(key for key in selected if key not in available)
        if missing:
            raise ValueError(f"Parameter-path outputs are unavailable: {missing!r}.")

        output_errors: dict[str, np.ndarray] = {}
        normalized_errors = []
        for output_name in selected:
            predicted = np.asarray(context.predictions[output_name], dtype=float)
            reference = np.asarray(context.references[output_name], dtype=float)
            if predicted.shape != reference.shape or predicted.shape[0] != values.size:
                raise ValueError(
                    f"Output {output_name!r} prediction/reference arrays must match "
                    "and start with the validation case axis."
                )
            if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
                raise ValueError(f"Output {output_name!r} contains non-finite values.")
            delta = (predicted - reference).reshape(values.size, -1)
            reference_flat = reference.reshape(values.size, -1)
            numerator = np.linalg.norm(delta, axis=1)
            denominator = np.linalg.norm(reference_flat, axis=1)
            global_scale = max(float(np.linalg.norm(reference_flat)), 1.0)
            floor = np.finfo(float).eps * global_scale
            errors = numerator / np.maximum(denominator, floor)
            output_errors[output_name] = errors[order]
            tolerance = tolerances.get(output_name, default_tolerance)
            normalized_errors.append(errors[order] / tolerance)

        case_risk = np.max(np.stack(normalized_errors, axis=0), axis=0)
        interior_expected = case_risk[:-2] + (
            (case_risk[2:] - case_risk[:-2])
            * (sorted_values[1:-1] - sorted_values[:-2])
            / (sorted_values[2:] - sorted_values[:-2])
        )
        risk_floor = max(float(np.median(case_risk)) * 0.05, 1.0e-12)
        spike_ratios = case_risk[1:-1] / np.maximum(interior_expected, risk_floor)
        worst_index = int(np.argmax(case_risk))
        spike_offset = int(np.argmax(spike_ratios))
        spike_index = spike_offset + 1
        worst_spike = float(spike_ratios[spike_offset])
        normalized_contract = np.asarray(
            [float(case_risk[worst_index]), worst_spike / spike_limit]
        )
        sorted_case_ids = tuple(case_ids[index] for index in order)
        evidence = {
            "parameter": parameter_name,
            "parameter_values": sorted_values.tolist(),
            "case_ids": sorted_case_ids,
            "output_relative_l2": {
                key: values.tolist() for key, values in output_errors.items()
            },
            "output_tolerances": {
                key: tolerances.get(key, default_tolerance) for key in selected
            },
            "case_risk": case_risk.tolist(),
            "worst_case_id": sorted_case_ids[worst_index],
            "worst_parameter_value": float(sorted_values[worst_index]),
            "maximum_spike_ratio": worst_spike,
            "spike_case_id": sorted_case_ids[spike_index],
            "spike_parameter_value": float(sorted_values[spike_index]),
            "spike_ratio_limit": spike_limit,
        }
        return verification.VerificationClaim.compare(
            name=check_name,
            observable="normalized worst field error and interior spike ratio",
            actual=normalized_contract,
            expected=np.zeros(2),
            reference="independent physical fields sampled along one parameter path",
            absolute_tolerance=1.0,
            validity_domain=(
                f"held-out path in parameter {parameter_name!r}; "
                "does not establish behavior outside the sampled interval"
            ),
            evidence=evidence,
        )

    return OperatorCheck(
        name=check_name,
        evaluator=evaluate,
        version=check_version,
        description=(
            "Worst-case held-out field error and isolated interpolation-spike "
            f"audit along parameter {parameter_name!r}."
        ),
        metadata={
            "parameter": parameter_name,
            "maximum_relative_l2": default_tolerance,
            "output_tolerances": tolerances,
            "maximum_spike_ratio": spike_limit,
            "minimum_case_count": 3,
        },
    )


def parameter_path_refinement_plan(
    claim: verification.VerificationClaim,
    *,
    existing_values=(),
    maximum_candidates: int = 3,
    minimum_spacing: float = 0.0,
) -> ParameterPathRefinementPlan:
    """Select diverse high-risk samples from a failed parameter-path claim.

    The function does not generate fields, mutate a dataset, or retrain a
    model. It translates independent failure evidence into a bounded set of
    parameter values that a project can evaluate with its trusted simulator.
    """

    count = int(maximum_candidates)
    spacing = float(minimum_spacing)
    if count < 1:
        raise ValueError("maximum_candidates must be positive.")
    if not np.isfinite(spacing) or spacing < 0.0:
        raise ValueError("minimum_spacing must be finite and non-negative.")
    evidence = dict(getattr(claim, "evidence", {}) or {})
    required = {"parameter", "parameter_values", "case_ids", "case_risk"}
    missing = required.difference(evidence)
    if missing:
        raise ValueError(
            f"Refinement requires parameter-path evidence; missing {tuple(sorted(missing))!r}."
        )
    parameter = str(evidence["parameter"])
    values = np.asarray(evidence["parameter_values"], dtype=float).reshape(-1)
    risks = np.asarray(evidence["case_risk"], dtype=float).reshape(-1)
    case_ids = tuple(str(item) for item in evidence["case_ids"])
    if values.size != risks.size or len(case_ids) != values.size:
        raise ValueError("Parameter-path values, risks, and case IDs must align.")
    if not np.isfinite(values).all() or not np.isfinite(risks).all():
        raise ValueError("Parameter-path refinement evidence must be finite.")
    if getattr(claim, "status", None) != "failed":
        return ParameterPathRefinementPlan(
            parameter=parameter,
            values=(),
            case_ids=(),
            risks=(),
            reason="path_claim_not_failed",
        )

    existing = np.asarray(tuple(existing_values), dtype=float).reshape(-1)
    if existing.size and not np.isfinite(existing).all():
        raise ValueError("existing_values must be finite.")
    span = max(float(np.ptp(values)), np.finfo(float).eps)
    tolerance = max(spacing, 64.0 * np.finfo(float).eps * max(np.max(np.abs(values)), 1.0))
    remaining = [
        index
        for index, value in enumerate(values)
        if not existing.size or float(np.min(np.abs(existing - value))) > tolerance
    ]
    selected = []
    anchors = list(existing)
    while remaining and len(selected) < count:
        scores = []
        for index in remaining:
            if anchors:
                distance = min(abs(float(values[index]) - float(item)) for item in anchors)
            else:
                distance = span
            diversity = min(distance / span, 1.0)
            scores.append(float(risks[index]) * np.sqrt(max(diversity, 0.05)))
        chosen_offset = int(np.argmax(scores))
        chosen = remaining.pop(chosen_offset)
        selected.append(chosen)
        anchors.append(float(values[chosen]))
        remaining = [
            index
            for index in remaining
            if abs(float(values[index]) - float(values[chosen])) > tolerance
        ]
    return ParameterPathRefinementPlan(
        parameter=parameter,
        values=tuple(float(values[index]) for index in selected),
        case_ids=tuple(case_ids[index] for index in selected),
        risks=tuple(float(risks[index]) for index in selected),
    )


__all__ = [
    "OperatorCheck",
    "OperatorCheckContext",
    "ParameterPathRefinementPlan",
    "parameter_path_refinement_plan",
    "parameter_path_reliability_check",
]
