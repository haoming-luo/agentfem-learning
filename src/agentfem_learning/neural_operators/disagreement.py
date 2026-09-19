"""Model-disagreement evidence for learned scientific operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorEnsembleDisagreement:
    """Per-case disagreement across independently trained operator members.

    Disagreement is an acquisition signal, not a calibrated error bar.  It can
    identify inputs on which independently initialized models do not agree,
    but agreement alone cannot establish physical accuracy.
    """

    member_count: int
    case_count: int
    output_relative_l2: Mapping[str, tuple[float, ...]]
    case_risk: tuple[float, ...]

    def summary(self) -> dict[str, object]:
        return {
            "kind": "operator_ensemble_disagreement",
            "semantics": "epistemic_disagreement_proxy_not_calibrated_uncertainty",
            "member_count": self.member_count,
            "case_count": self.case_count,
            "output_relative_l2": dict(self.output_relative_l2),
            "case_risk": self.case_risk,
        }


def operator_ensemble_disagreement(
    members: Sequence[Mapping[str, object]],
) -> OperatorEnsembleDisagreement:
    """Measure normalized per-case field disagreement for an operator ensemble.

    Every member must predict the same named arrays with the case axis first.
    For each output and case, the root-mean-square member deviation is divided
    by the norm of the ensemble mean.  The maximum normalized output value is
    retained as the case acquisition risk.
    """

    selected = tuple(members)
    if len(selected) < 2:
        raise ValueError("Operator disagreement requires at least two members.")
    names = tuple(selected[0])
    if not names:
        raise ValueError("Operator members must contain at least one output field.")
    if any(set(member) != set(names) for member in selected[1:]):
        raise ValueError("Operator ensemble members must have identical output names.")

    output_risk: dict[str, tuple[float, ...]] = {}
    case_count: int | None = None
    for name in names:
        arrays = [np.asarray(member[name], dtype=float) for member in selected]
        shape = arrays[0].shape
        if not shape or shape[0] < 1:
            raise ValueError(f"Operator output {name!r} must start with a case axis.")
        if any(array.shape != shape for array in arrays[1:]):
            raise ValueError(f"Operator output {name!r} shapes must match across members.")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Operator output {name!r} contains non-finite predictions.")
        if case_count is None:
            case_count = shape[0]
        elif shape[0] != case_count:
            raise ValueError("Every ensemble output must use the same case count.")

        stacked = np.stack(arrays, axis=0)
        mean = np.mean(stacked, axis=0)
        centered = (stacked - mean[None, ...]).reshape(len(selected), shape[0], -1)
        numerator = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=0))
        mean_flat = mean.reshape(shape[0], -1)
        denominator = np.linalg.norm(mean_flat, axis=1)
        global_scale = max(float(np.linalg.norm(mean_flat)), 1.0)
        floor = np.finfo(float).eps * global_scale
        output_risk[name] = tuple(
            float(value) for value in numerator / np.maximum(denominator, floor)
        )

    assert case_count is not None
    case_risk = np.max(
        np.stack([np.asarray(values) for values in output_risk.values()], axis=0),
        axis=0,
    )
    return OperatorEnsembleDisagreement(
        member_count=len(selected),
        case_count=case_count,
        output_relative_l2=output_risk,
        case_risk=tuple(float(value) for value in case_risk),
    )


__all__ = [
    "OperatorEnsembleDisagreement",
    "operator_ensemble_disagreement",
]
