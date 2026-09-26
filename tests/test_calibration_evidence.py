# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from agentfem_learning.learned_constitutive import evaluate_calibration_evidence


def _record():
    return {
        "schema": "agentfem-learning.constitutive-calibration-evidence.v1",
        "status": "accepted",
        "calibration_scope": "Alloy X, room temperature, small-strain cyclic loading.",
        "source": {
            "kind": "published_experiment",
            "identifier": "https://example.org/stable-record",
            "license": "CC-BY-4.0",
            "raw_data_sha256": "a" * 64,
        },
        "model": {
            "architecture": "example.v1",
            "revision": "immutable-revision",
            "weights_sha256": "b" * 64,
        },
        "calibration_objective": {
            "name": "weighted_stress_history_fit",
            "loss": "mean squared Cauchy-stress error",
            "weighting": "equal trajectory weight",
        },
        "partitions": {
            "fit": ["fit-1"],
            "validation": ["validation-1"],
            "held_out": ["blind-1"],
        },
        "units": {"strain": "1", "stress": "Pa"},
        "held_out_metrics": [
            {
                "name": "stress_rmse_pa",
                "value": 2.0e6,
                "maximum": 5.0e6,
                "unit": "Pa",
                "sample_count": 12,
            }
        ],
    }


def test_calibration_evidence_accepts_independent_held_out_contract():
    assert evaluate_calibration_evidence(_record()).accepted


def test_calibration_evidence_rejects_leakage_and_placeholder_status():
    leaked = _record()
    leaked["partitions"]["held_out"] = ["fit-1"]
    assert not evaluate_calibration_evidence(leaked).accepted
    placeholder = deepcopy(_record())
    placeholder["status"] = "not_established"
    assert not evaluate_calibration_evidence(placeholder).accepted
