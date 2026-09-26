# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from agentfem_learning.learned_constitutive.denim import (
    evaluate_global_bar,
    evaluate_material_point,
    evaluate_nonproportional_path,
    evaluate_structural_convergence,
    load_denim_v1_acceptance,
    tension_torsion_circle,
    verify_published_evidence,
)


def _runtime(contract):
    identity = contract["identity"]
    return {
        "model_name": identity["model_name"],
        "model_version": identity["model_version"],
        "model_revision": identity["model_revision"],
        "source_checkpoint_sha256": identity["legacy_checkpoint_sha256"],
        "weights_sha256": identity["runtime_weights_sha256"],
        "dataset_id": identity["dataset_repository"],
        "dataset_revision": identity["dataset_revision"],
        "dtype": "float64",
        "offline_runtime": True,
    }


def test_material_point_golden_accepts_only_fixed_identity_and_response():
    contract = load_denim_v1_acceptance()
    golden = contract["material_point_golden"]
    record = {
        "schema": golden["schema"],
        "loading_path": {"identity": golden["loading_path_identity"]},
        "runtime": _runtime(contract),
        "maximum_absolute_stress_mpa": golden["maximum_absolute_stress_mpa"]["reference"],
        "final_peeq": golden["final_peeq"]["reference"],
        "maximum_yield_residual_pa": 0.5,
        "stress": [[[0.0] * 3 for _ in range(3)] for _ in range(golden["steps"])],
        "peeq": [0.0] * golden["steps"],
    }
    assert evaluate_material_point(record, contract).accepted
    altered = deepcopy(record)
    altered["runtime"]["dataset_revision"] = "latest"
    assert not evaluate_material_point(altered, contract).accepted


def test_global_bar_golden_requires_plasticity_and_in_domain_state():
    contract = load_denim_v1_acceptance()
    golden = contract["global_bar_golden"]
    record = {
        "schema": golden["schema"],
        "problem_identity": golden["problem_identity"],
        "status": "completed",
        "converged": True,
        "accepted_increments": golden["increments"],
        "maximum_displacement": golden["target_displacement"],
        "maximum_absolute_stress_pa": golden["maximum_absolute_stress_pa"]["reference"],
        "maximum_peeq": 1.0e-3,
        "learned_constitutive": {
            "runtime": _runtime(contract),
            "diagnostics": {"applicability_counts": {"in_domain": 48}},
        },
    }
    assert evaluate_global_bar(record, contract).accepted
    elastic = deepcopy(record)
    elastic["maximum_peeq"] = 0.0
    assert not evaluate_global_bar(elastic, contract).accepted
    out_of_domain = deepcopy(record)
    out_of_domain["learned_constitutive"]["diagnostics"]["applicability_counts"] = {
        "in_domain": 47,
        "out_of_domain": 1,
    }
    assert not evaluate_global_bar(out_of_domain, contract).accepted


def test_published_receipt_is_authenticated_and_not_a_software_golden(tmp_path):
    source = (
        Path(__file__).parents[1]
        / "evidence"
        / "denim_v1"
        / "published_deployment_validation.json"
    )
    report = verify_published_evidence(source)
    assert report.accepted
    assert report.subject == "denim_v1_published_scientific_evidence_receipt"
    altered = tmp_path / "altered.json"
    altered.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not verify_published_evidence(altered).accepted


def test_nonproportional_gate_requires_refinement_covariance_and_scope():
    contract = load_denim_v1_acceptance()
    gate = contract["nonproportional_path_gate"]
    record = {
        "schema": gate["schema"],
        "path": {
            "identity": gate["path_identity"],
            "reference_role": "path_topology_only_not_numerical_calibration",
            "refinement_factors": gate["refinement_factors"],
        },
        "runtime": _runtime(contract),
        "final_peeq": 0.002,
        "minimum_peeq_increment": 0.0,
        "refinement": {
            "stress_relative_l2": 0.01,
            "peeq_relative_l2": 0.01,
        },
        "rotation_covariance": {
            "stress_relative_l2": 1.0e-12,
            "peeq_relative_l2": 1.0e-12,
        },
        "physics": {
            "maximum_yield_residual_pa": 20.0,
            "maximum_plastic_strain_trace": 1.0e-12,
            "all_refinements_usable": True,
            "rotated_path_usable": True,
        },
    }
    assert evaluate_nonproportional_path(record, contract).accepted
    misrepresented = deepcopy(record)
    misrepresented["path"]["reference_role"] = "abaqus_numerical_golden"
    assert not evaluate_nonproportional_path(misrepresented, contract).accepted
    unresolved = deepcopy(record)
    unresolved["refinement"]["stress_relative_l2"] = 0.2
    assert not evaluate_nonproportional_path(unresolved, contract).accepted


def test_tension_torsion_path_preserves_exact_knots_under_refinement():
    path = tension_torsion_circle(cycles=1, segments_per_cycle=8)
    refined = path.refine(4)
    assert path.point_count == 10
    assert refined.point_count == 37
    assert path.fingerprint != refined.fingerprint
    np.testing.assert_allclose(refined.strain[::4], path.strain)
    np.testing.assert_allclose(np.trace(path.strain, axis1=1, axis2=2), 0.0)


def test_structural_convergence_gate_uses_reaction_not_singular_peak():
    contract = load_denim_v1_acceptance()
    gate = contract["structural_convergence_gate"]
    case = {
        "cells_per_axis": [4, 1, 1],
        "increments": 16,
        "status": "completed",
        "converged": True,
        "applicability_counts": {"in_domain": 24},
    }
    record = {
        "schema": gate["schema"],
        "problem_identity": gate["problem_identity"],
        "runtime": _runtime(contract),
        "axes": {
            "mesh": [
                {**deepcopy(case), "cells_per_axis": [4, 1, 1]},
                {**deepcopy(case), "cells_per_axis": [8, 2, 2]},
                {**deepcopy(case), "cells_per_axis": [12, 3, 3]},
            ],
            "increment": [
                {
                    **deepcopy(case),
                    "cells_per_axis": [12, 3, 3],
                    "increments": 4,
                },
                {
                    **deepcopy(case),
                    "cells_per_axis": [12, 3, 3],
                    "increments": 8,
                },
                {**deepcopy(case), "cells_per_axis": [12, 3, 3]},
            ],
        },
        "execution_policy": {"unique_case_count": 5},
        "convergence": {
            "mesh_reaction_last_two_relative_change": 0.01,
            "mesh_peeq_last_two_relative_change": 0.5,
            "increment_reaction_last_two_relative_change": 0.001,
            "increment_peeq_last_two_relative_change": 0.002,
        },
    }
    assert evaluate_structural_convergence(record, contract).accepted
    unconverged = deepcopy(record)
    unconverged["convergence"]["mesh_reaction_last_two_relative_change"] = 0.2
    assert not evaluate_structural_convergence(unconverged, contract).accepted
