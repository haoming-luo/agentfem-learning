# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed acceptance for the fixed DENIM v1 scientific asset."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256


@dataclass(frozen=True)
class AcceptanceCheck:
    """One machine-readable acceptance assertion."""

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
class AcceptanceReport:
    """Aggregate evidence without promoting scientific maturity implicitly."""

    subject: str
    checks: tuple[AcceptanceCheck, ...]
    maturity: Mapping[str, object]

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.accepted for check in self.checks)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem-learning.acceptance-report.v1",
            "subject": self.subject,
            "accepted": self.accepted,
            "maturity": dict(self.maturity),
            "checks": [check.summary() for check in self.checks],
        }


def load_denim_v1_acceptance() -> dict[str, object]:
    """Load the acceptance contract shipped inside the installed wheel."""

    resource = files(__package__).joinpath("denim_v1_acceptance.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _path(record: Mapping[str, Any], *keys: str) -> object:
    value: object = record
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _equal(name: str, observed: object, expected: object) -> AcceptanceCheck:
    return AcceptanceCheck(name, observed == expected, observed, expected)


def _close(name: str, observed: object, rule: Mapping[str, object]) -> AcceptanceCheck:
    reference = float(rule["reference"])
    absolute = float(rule.get("absolute_tolerance", 0.0))
    relative = float(rule.get("relative_tolerance", 0.0))
    finite = isinstance(observed, (int, float)) and math.isfinite(float(observed))
    accepted = finite and math.isclose(
        float(observed), reference, rel_tol=relative, abs_tol=absolute
    )
    return AcceptanceCheck(
        name,
        accepted,
        observed,
        {
            "reference": reference,
            "absolute_tolerance": absolute,
            "relative_tolerance": relative,
        },
    )


def _bounded(
    name: str,
    observed: object,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> AcceptanceCheck:
    finite = isinstance(observed, (int, float)) and math.isfinite(float(observed))
    accepted = finite
    if finite and lower is not None:
        accepted = accepted and float(observed) >= float(lower)
    if finite and upper is not None:
        accepted = accepted and float(observed) <= float(upper)
    return AcceptanceCheck(
        name,
        bool(accepted),
        observed,
        {"lower": lower, "upper": upper},
    )


def _identity_checks(runtime: Mapping[str, object], contract: Mapping[str, object]):
    identity = contract["identity"]
    return (
        _equal("runtime.model_name", runtime.get("model_name"), identity["model_name"]),
        _equal(
            "runtime.model_version",
            runtime.get("model_version"),
            identity["model_version"],
        ),
        _equal(
            "runtime.model_revision",
            runtime.get("model_revision"),
            identity["model_revision"],
        ),
        _equal(
            "runtime.source_checkpoint_sha256",
            runtime.get("source_checkpoint_sha256"),
            identity["legacy_checkpoint_sha256"],
        ),
        _equal(
            "runtime.weights_sha256",
            runtime.get("weights_sha256"),
            identity["runtime_weights_sha256"],
        ),
        _equal(
            "runtime.dataset_id",
            runtime.get("dataset_id"),
            identity["dataset_repository"],
        ),
        _equal(
            "runtime.dataset_revision",
            runtime.get("dataset_revision"),
            identity["dataset_revision"],
        ),
        _equal("runtime.dtype", runtime.get("dtype"), "float64"),
        _equal("runtime.offline_runtime", runtime.get("offline_runtime"), True),
    )


def evaluate_material_point(
    record: Mapping[str, object],
    contract: Mapping[str, object] | None = None,
) -> AcceptanceReport:
    """Evaluate the fixed 121-step material-point software Golden."""

    contract = contract or load_denim_v1_acceptance()
    golden = contract["material_point_golden"]
    runtime = record.get("runtime", {})
    loading_path = record.get("loading_path", {})
    try:
        stress = np.asarray(record.get("stress", ()), dtype=float)
        peeq = np.asarray(record.get("peeq", ()), dtype=float)
        histories_valid = (
            stress.shape == (int(golden["steps"]), 3, 3)
            and peeq.shape == (int(golden["steps"]),)
            and np.all(np.isfinite(stress))
            and np.all(np.isfinite(peeq))
        )
    except (TypeError, ValueError):
        stress = np.asarray(())
        peeq = np.asarray(())
        histories_valid = False
    checks = [
        _equal("record.schema", record.get("schema"), golden["schema"]),
        _equal(
            "loading_path.identity",
            loading_path.get("identity") if isinstance(loading_path, Mapping) else None,
            golden["loading_path_identity"],
        ),
        AcceptanceCheck(
            "history_shape_and_finiteness",
            bool(histories_valid),
            {"stress_shape": list(stress.shape), "peeq_shape": list(peeq.shape)},
            {
                "stress_shape": [int(golden["steps"]), 3, 3],
                "peeq_shape": [int(golden["steps"])],
                "finite": True,
            },
        ),
        *_identity_checks(runtime if isinstance(runtime, Mapping) else {}, contract),
        _close(
            "maximum_absolute_stress_mpa",
            record.get("maximum_absolute_stress_mpa"),
            golden["maximum_absolute_stress_mpa"],
        ),
        _close("final_peeq", record.get("final_peeq"), golden["final_peeq"]),
    ]
    residual = record.get("maximum_yield_residual_pa")
    upper = float(golden["maximum_yield_residual_pa"]["upper_bound"])
    checks.append(
        AcceptanceCheck(
            "maximum_yield_residual_pa",
            isinstance(residual, (int, float))
            and math.isfinite(float(residual))
            and abs(float(residual)) <= upper,
            residual,
            {"absolute_upper_bound": upper},
        )
    )
    return AcceptanceReport(
        "denim_v1_material_point_software_golden",
        tuple(checks),
        contract["maturity"],
    )


def evaluate_global_bar(
    record: Mapping[str, object],
    contract: Mapping[str, object] | None = None,
) -> AcceptanceReport:
    """Evaluate the fixed plastic global-Newton regression."""

    contract = contract or load_denim_v1_acceptance()
    golden = contract["global_bar_golden"]
    runtime = _path(record, "learned_constitutive", "runtime")
    diagnostics = _path(record, "learned_constitutive", "diagnostics")
    counts = (
        diagnostics.get("applicability_counts", {}) if isinstance(diagnostics, Mapping) else {}
    )
    try:
        count_values = [int(value) for value in counts.values()]
    except (AttributeError, TypeError, ValueError):
        count_values = []
    required_domain = golden["required_applicability_status"]
    maximum_peeq = record.get("maximum_peeq")
    minimum_peeq = float(golden["minimum_maximum_peeq"])
    checks = [
        _equal("record.schema", record.get("schema"), golden["schema"]),
        _equal(
            "problem_identity",
            record.get("problem_identity"),
            golden["problem_identity"],
        ),
        _equal("status", record.get("status"), "completed"),
        _equal("converged", record.get("converged"), True),
        _equal(
            "accepted_increments",
            record.get("accepted_increments"),
            golden["increments"],
        ),
        _close(
            "maximum_displacement",
            record.get("maximum_displacement"),
            {
                "reference": golden["target_displacement"],
                "absolute_tolerance": 1e-12,
                "relative_tolerance": 1e-10,
            },
        ),
        _close(
            "maximum_absolute_stress_pa",
            record.get("maximum_absolute_stress_pa"),
            golden["maximum_absolute_stress_pa"],
        ),
        AcceptanceCheck(
            "maximum_peeq_is_plastic",
            isinstance(maximum_peeq, (int, float))
            and math.isfinite(float(maximum_peeq))
            and float(maximum_peeq) >= minimum_peeq,
            maximum_peeq,
            {"minimum": minimum_peeq},
        ),
        AcceptanceCheck(
            "applicability_status",
            isinstance(counts, Mapping)
            and bool(count_values)
            and int(counts.get(required_domain, 0)) > 0
            and sum(count_values) == int(counts.get(required_domain, 0)),
            dict(counts) if isinstance(counts, Mapping) else counts,
            {"only": required_domain, "minimum_count": 1},
        ),
        *_identity_checks(runtime if isinstance(runtime, Mapping) else {}, contract),
    ]
    return AcceptanceReport(
        "denim_v1_global_plastic_software_golden",
        tuple(checks),
        contract["maturity"],
    )


def evaluate_nonproportional_path(
    record: Mapping[str, object],
    contract: Mapping[str, object] | None = None,
) -> AcceptanceReport:
    """Evaluate a non-proportional path without claiming alloy calibration."""

    contract = contract or load_denim_v1_acceptance()
    gate = contract["nonproportional_path_gate"]
    runtime = record.get("runtime", {})
    checks = [
        _equal("record.schema", record.get("schema"), gate["schema"]),
        _equal(
            "path.identity",
            _path(record, "path", "identity"),
            gate["path_identity"],
        ),
        _equal(
            "path.reference_role",
            _path(record, "path", "reference_role"),
            "path_topology_only_not_numerical_calibration",
        ),
        _equal(
            "path.refinement_factors",
            _path(record, "path", "refinement_factors"),
            gate["refinement_factors"],
        ),
        _bounded(
            "final_peeq_is_plastic",
            record.get("final_peeq"),
            lower=float(gate["minimum_final_peeq"]),
        ),
        _bounded(
            "peeq_monotonicity",
            record.get("minimum_peeq_increment"),
            lower=-float(gate["maximum_peeq_decrease"]),
        ),
        _bounded(
            "refinement.stress_relative_l2",
            _path(record, "refinement", "stress_relative_l2"),
            upper=float(gate["maximum_refined_stress_relative_l2"]),
        ),
        _bounded(
            "refinement.peeq_relative_l2",
            _path(record, "refinement", "peeq_relative_l2"),
            upper=float(gate["maximum_refined_peeq_relative_l2"]),
        ),
        _bounded(
            "rotation_covariance.stress_relative_l2",
            _path(record, "rotation_covariance", "stress_relative_l2"),
            upper=float(gate["maximum_rotation_relative_l2"]),
        ),
        _bounded(
            "rotation_covariance.peeq_relative_l2",
            _path(record, "rotation_covariance", "peeq_relative_l2"),
            upper=float(gate["maximum_rotation_relative_l2"]),
        ),
        _bounded(
            "physics.maximum_yield_residual_pa",
            _path(record, "physics", "maximum_yield_residual_pa"),
            upper=float(gate["maximum_yield_residual_pa"]),
        ),
        _bounded(
            "physics.maximum_plastic_strain_trace",
            _path(record, "physics", "maximum_plastic_strain_trace"),
            upper=float(gate["maximum_plastic_strain_trace"]),
        ),
        _equal(
            "physics.all_refinements_usable",
            _path(record, "physics", "all_refinements_usable"),
            True,
        ),
        _equal(
            "physics.rotated_path_usable",
            _path(record, "physics", "rotated_path_usable"),
            True,
        ),
        *_identity_checks(runtime if isinstance(runtime, Mapping) else {}, contract),
    ]
    return AcceptanceReport(
        "denim_v1_nonproportional_path_gate",
        tuple(checks),
        contract["maturity"],
    )


def evaluate_structural_convergence(
    record: Mapping[str, object],
    contract: Mapping[str, object] | None = None,
) -> AcceptanceReport:
    """Evaluate two independent structural convergence axes."""

    contract = contract or load_denim_v1_acceptance()
    gate = contract["structural_convergence_gate"]
    runtime = record.get("runtime", {})
    axes = record.get("axes", {})
    all_cases = []
    case_identities: set[tuple[tuple[int, ...], int]] = set()
    if isinstance(axes, Mapping):
        for name in ("mesh", "increment"):
            values = axes.get(name, ())
            if isinstance(values, list):
                for case in values:
                    if not isinstance(case, Mapping):
                        continue
                    identity = (
                        tuple(int(value) for value in case.get("cells_per_axis", ())),
                        int(case.get("increments", -1)),
                    )
                    if identity not in case_identities:
                        case_identities.add(identity)
                        all_cases.append(case)
    case_checks = []
    for index, case in enumerate(all_cases):
        usable_counts = case.get("applicability_counts", {})
        only_in_domain = (
            isinstance(usable_counts, Mapping)
            and int(usable_counts.get("in_domain", 0)) > 0
            and sum(int(value) for value in usable_counts.values())
            == int(usable_counts.get("in_domain", 0))
        )
        case_checks.extend(
            (
                _equal(f"case.{index}.status", case.get("status"), "completed"),
                _equal(f"case.{index}.converged", case.get("converged"), True),
                _equal(f"case.{index}.in_domain", only_in_domain, True),
            )
        )
    checks = [
        _equal("record.schema", record.get("schema"), gate["schema"]),
        _equal(
            "problem_identity",
            record.get("problem_identity"),
            gate["problem_identity"],
        ),
        _equal(
            "execution.unique_case_count",
            _path(record, "execution_policy", "unique_case_count"),
            5,
        ),
        _equal("execution.observed_unique_case_count", len(all_cases), 5),
        *case_checks,
        _bounded(
            "mesh.reaction_last_two_relative_change",
            _path(record, "convergence", "mesh_reaction_last_two_relative_change"),
            upper=float(gate["maximum_mesh_reaction_relative_change"]),
        ),
        _bounded(
            "increment.reaction_last_two_relative_change",
            _path(
                record,
                "convergence",
                "increment_reaction_last_two_relative_change",
            ),
            upper=float(gate["maximum_increment_reaction_relative_change"]),
        ),
        _bounded(
            "increment.peeq_last_two_relative_change",
            _path(record, "convergence", "increment_peeq_last_two_relative_change"),
            upper=float(gate["maximum_increment_peeq_relative_change"]),
        ),
        *_identity_checks(runtime if isinstance(runtime, Mapping) else {}, contract),
    ]
    return AcceptanceReport(
        "denim_v1_structural_mesh_increment_convergence_gate",
        tuple(checks),
        contract["maturity"],
    )


def verify_published_evidence(
    source: str | Path,
    contract: Mapping[str, object] | None = None,
) -> AcceptanceReport:
    """Authenticate and inspect the separately published comparison receipt."""

    contract = contract or load_denim_v1_acceptance()
    expected = contract["published_scientific_evidence"]
    path = Path(source).expanduser().resolve()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        digest = file_sha256(path)
    except (OSError, json.JSONDecodeError):
        record = {}
        digest = None
    checks = [
        _equal("source.sha256", digest, expected["source_sha256"]),
        _equal("source.scope", record.get("scope"), expected["scope"]),
        _close(
            "time_discretization.121.rmse_mpa",
            _path(record, "time_discretization", "121", "rmse_mpa"),
            {
                "reference": expected["metrics"]["material_path_121_rmse_mpa"],
                "absolute_tolerance": 1e-12,
            },
        ),
        _close(
            "structure.mild_cyclic.reaction_relative_l2",
            _path(record, "structure", "mild_cyclic", "reaction_relative_l2"),
            {
                "reference": expected["metrics"]["mild_cyclic_reaction_relative_l2"],
                "absolute_tolerance": 1e-15,
            },
        ),
        _close(
            "structure.severe_monotonic.reaction_relative_l2",
            _path(record, "structure", "severe_monotonic", "reaction_relative_l2"),
            {
                "reference": expected["metrics"]["severe_monotonic_reaction_relative_l2"],
                "absolute_tolerance": 1e-15,
            },
        ),
        _equal(
            "structure.severe_cyclic_stress_test.passed",
            _path(record, "structure", "severe_cyclic_stress_test", "passed"),
            True,
        ),
    ]
    return AcceptanceReport(
        "denim_v1_published_scientific_evidence_receipt",
        tuple(checks),
        contract["maturity"],
    )


__all__ = [
    "AcceptanceCheck",
    "AcceptanceReport",
    "evaluate_global_bar",
    "evaluate_material_point",
    "load_denim_v1_acceptance",
    "verify_published_evidence",
]
