# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed evidence contract for learned-material calibration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CalibrationCheck:
    name: str
    accepted: bool
    observed: object
    expected: object

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "accepted": self.accepted,
            "observed": self.observed,
            "expected": self.expected,
        }


@dataclass(frozen=True)
class CalibrationEvidenceReport:
    checks: tuple[CalibrationCheck, ...]

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.accepted for check in self.checks)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem-learning.calibration-evidence-report.v1",
            "accepted": self.accepted,
            "checks": [check.summary() for check in self.checks],
        }


def _check(name: str, accepted: bool, observed: object, expected: object):
    return CalibrationCheck(name, bool(accepted), observed, expected)


def evaluate_calibration_evidence(
    record: Mapping[str, object],
) -> CalibrationEvidenceReport:
    """Validate provenance, split independence and declared held-out metrics.

    This checks the evidence contract, not the truth of an external paper or
    experiment. Raw-data authentication and reproduction remain mandatory
    scientific work outside this structural validator.
    """

    source = record.get("source", {})
    model = record.get("model", {})
    partitions = record.get("partitions", {})
    metrics = record.get("held_out_metrics", ())
    objective = record.get("calibration_objective", {})
    checks = [
        _check(
            "schema",
            record.get("schema")
            == "agentfem-learning.constitutive-calibration-evidence.v1",
            record.get("schema"),
            "agentfem-learning.constitutive-calibration-evidence.v1",
        ),
        _check("status", record.get("status") == "accepted", record.get("status"), "accepted"),
        _check(
            "scope",
            isinstance(record.get("calibration_scope"), str)
            and bool(record["calibration_scope"].strip()),
            record.get("calibration_scope"),
            "nonempty explicit scope",
        ),
    ]
    source_mapping = source if isinstance(source, Mapping) else {}
    model_mapping = model if isinstance(model, Mapping) else {}
    checks.extend(
        (
            _check(
                "source.kind",
                source_mapping.get("kind") in {"experiment", "published_experiment"},
                source_mapping.get("kind"),
                ["experiment", "published_experiment"],
            ),
            _check(
                "source.identifier",
                isinstance(source_mapping.get("identifier"), str)
                and bool(source_mapping["identifier"].strip()),
                source_mapping.get("identifier"),
                "DOI, repository record, or stable publication URL",
            ),
            _check(
                "source.license",
                isinstance(source_mapping.get("license"), str)
                and bool(source_mapping["license"].strip()),
                source_mapping.get("license"),
                "declared reuse terms",
            ),
            _check(
                "source.raw_data_sha256",
                isinstance(source_mapping.get("raw_data_sha256"), str)
                and bool(_SHA256.fullmatch(source_mapping["raw_data_sha256"])),
                source_mapping.get("raw_data_sha256"),
                "64 lowercase hexadecimal characters",
            ),
            _check(
                "model.identity",
                all(
                    isinstance(model_mapping.get(name), str)
                    and bool(model_mapping[name].strip())
                    for name in ("architecture", "revision", "weights_sha256")
                )
                and bool(_SHA256.fullmatch(str(model_mapping.get("weights_sha256", "")))),
                dict(model_mapping),
                "architecture, immutable revision, and weights SHA-256",
            ),
        )
    )
    checks.append(
        _check(
            "calibration_objective",
            isinstance(objective, Mapping)
            and all(
                isinstance(objective.get(name), str) and bool(objective[name].strip())
                for name in ("name", "loss", "weighting")
            ),
            dict(objective) if isinstance(objective, Mapping) else objective,
            "named objective, loss, and weighting policy",
        )
    )

    partition_mapping = partitions if isinstance(partitions, Mapping) else {}
    sets = {}
    for name in ("fit", "validation", "held_out"):
        values = partition_mapping.get(name, ())
        valid = (
            isinstance(values, list)
            and bool(values)
            and all(isinstance(value, str) and value for value in values)
            and len(values) == len(set(values))
        )
        sets[name] = set(values) if valid else set()
        checks.append(
            _check(f"partitions.{name}", valid, values, "nonempty unique sample IDs")
        )
    overlap = (sets["fit"] & sets["validation"]) | (
        sets["fit"] & sets["held_out"]
    ) | (sets["validation"] & sets["held_out"])
    checks.append(_check("partitions.disjoint", not overlap, sorted(overlap), []))

    units = record.get("units", {})
    checks.append(
        _check(
            "units",
            isinstance(units, Mapping)
            and bool(units)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in units.items()),
            dict(units) if isinstance(units, Mapping) else units,
            "nonempty quantity-to-unit mapping",
        )
    )
    valid_metrics = isinstance(metrics, list) and bool(metrics)
    if valid_metrics:
        for index, metric in enumerate(metrics):
            if not isinstance(metric, Mapping):
                checks.append(_check(f"metric.{index}", False, metric, "metric mapping"))
                continue
            value = metric.get("value")
            maximum = metric.get("maximum")
            name = metric.get("name")
            unit = metric.get("unit")
            sample_count = metric.get("sample_count")
            finite = isinstance(value, (int, float)) and math.isfinite(float(value))
            finite_maximum = isinstance(maximum, (int, float)) and math.isfinite(
                float(maximum)
            )
            checks.append(
                _check(
                    f"metric.{index}.{metric.get('name', 'unnamed')}",
                    isinstance(name, str)
                    and bool(name.strip())
                    and isinstance(unit, str)
                    and bool(unit.strip())
                    and isinstance(sample_count, int)
                    and sample_count > 0
                    and finite
                    and finite_maximum
                    and float(value) <= float(maximum),
                    value,
                    {
                        "maximum": maximum,
                        "partition": "held_out",
                        "unit": unit,
                        "minimum_sample_count": 1,
                    },
                )
            )
    else:
        checks.append(_check("held_out_metrics", False, metrics, "nonempty metric list"))
    return CalibrationEvidenceReport(tuple(checks))


__all__ = [
    "CalibrationCheck",
    "CalibrationEvidenceReport",
    "evaluate_calibration_evidence",
]
