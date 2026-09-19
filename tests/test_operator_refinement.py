from __future__ import annotations

import numpy as np
import pytest
from agentfem import campaigns, datasets, learning, verification

from agentfem_learning.neural_operators import (
    ParameterPathAcquisitionPlan,
    ParameterPathRefinementPlan,
    apply_parameter_path_refinement,
    merge_operator_datasets,
    merge_parameter_path_acquisition,
    operator_ensemble_disagreement,
    parameter_candidate_acquisition_plan,
)
from agentfem_learning.neural_operators.neuraloperator import (
    parameter_path_acquisition_plan,
)


def _dataset(values, *, prefix):
    values = np.asarray(values, dtype=float)
    source = learning.FieldEncoding(
        name="source",
        role="input",
        unit="1",
        representation="point_samples",
        shape=(1, 2),
        mesh_policy="registered_mesh_family",
    )
    response = learning.FieldEncoding(
        name="response",
        role="output",
        unit="1",
        representation="point_samples",
        shape=(1, 2),
        mesh_policy="registered_mesh_family",
    )
    fields = values[:, None, None] * np.ones((len(values), 1, 2))
    coordinates = np.broadcast_to(
        np.asarray([[[0.0, 0.0], [1.0, 0.0]]]),
        (len(values), 2, 2),
    ).copy()
    return datasets.ScientificFieldDataset(
        case_ids=tuple(f"{prefix}-{index:03d}" for index in range(len(values))),
        encodings=(source, response),
        fields={"source": fields, "response": 2.0 * fields},
        coordinates={"nodes": coordinates},
        parameters={"design": values},
        name=f"{prefix}_operator",
    )


def test_path_refinement_promotes_cases_without_validation_leakage():
    training = _dataset([0.1, 0.9], prefix="train")
    validation = _dataset([0.2, 0.3, 0.4, 0.5, 0.6], prefix="path")
    plan = ParameterPathRefinementPlan(
        parameter="design",
        values=(0.3, 0.5),
        case_ids=("path-001", "path-003"),
        risks=(2.0, 1.5),
    )

    refinement = apply_parameter_path_refinement(training, validation, plan)

    assert refinement.added_case_ids == plan.case_ids
    assert refinement.training_dataset.case_count == 4
    assert refinement.validation_dataset.case_count == 3
    assert set(refinement.training_dataset.case_ids).isdisjoint(
        refinement.validation_dataset.case_ids
    )
    assert refinement.training_dataset.parameters["design"] == pytest.approx(
        [0.1, 0.9, 0.3, 0.5]
    )
    assert refinement.validation_dataset.parameters["design"] == pytest.approx([0.2, 0.4, 0.6])
    summary = refinement.summary()
    assert summary["training_fingerprint_before"] == training.fingerprint
    assert summary["training_fingerprint_after"] != training.fingerprint
    assert summary["validation_parameter_range_before"] == (0.2, 0.6)
    assert summary["validation_parameter_range_after"] == (0.2, 0.6)


def test_refinement_preserves_a_minimum_independent_path():
    training = _dataset([0.1, 0.9], prefix="train")
    validation = _dataset([0.2, 0.3, 0.4, 0.5], prefix="path")
    plan = ParameterPathRefinementPlan(
        parameter="design",
        values=(0.3, 0.4),
        case_ids=("path-001", "path-002"),
        risks=(2.0, 1.5),
    )

    with pytest.raises(ValueError, match="too few independent path cases"):
        apply_parameter_path_refinement(training, validation, plan)


def test_refinement_does_not_hide_failed_path_endpoints():
    training = _dataset([0.1, 0.9], prefix="train")
    validation = _dataset([0.2, 0.3, 0.4, 0.5], prefix="path")
    plan = ParameterPathRefinementPlan(
        parameter="design",
        values=(0.2,),
        case_ids=("path-000",),
        risks=(2.0,),
    )

    with pytest.raises(ValueError, match="remove a path endpoint"):
        apply_parameter_path_refinement(training, validation, plan)


def test_generic_dataset_merge_rejects_duplicate_case_identity():
    base = _dataset([0.1, 0.2], prefix="base")
    additions = base.subset((1,), name="duplicate")

    with pytest.raises(ValueError, match="duplicate case IDs"):
        merge_operator_datasets(base, additions)


def _failed_path_claim():
    return verification.VerificationClaim.compare(
        name="parameter_path_reliability",
        observable="normalized path risk",
        actual=2.0,
        expected=0.0,
        reference="independent held-out path",
        absolute_tolerance=1.0,
        evidence={
            "parameter": "design",
            "parameter_values": [0.1, 0.3, 0.5],
            "case_risk": [0.2, 2.0, 0.3],
        },
    )


def test_failed_path_proposes_new_campaign_samples_between_observations():
    plan = parameter_path_acquisition_plan(
        _failed_path_claim(),
        existing_values=(0.0, 0.6),
        maximum_candidates=2,
    )
    space = campaigns.ParameterSpace.create(campaigns.RealParameter("design", 0.0, 0.6))
    sampling = plan.sampling_plan(space)

    assert sorted(plan.values) == pytest.approx((0.2, 0.4))
    assert set(plan.source_intervals) == {(0.1, 0.3), (0.3, 0.5)}
    assert sampling.method == "explicit"
    assert tuple(sample["design"] for sample in sampling.samples) == pytest.approx(plan.values)
    assert sampling.metadata["acquisition_plan"]["fingerprint"] == plan.fingerprint


def test_acquisition_sampling_requires_explicit_non_acquired_parameters():
    plan = ParameterPathAcquisitionPlan(
        parameter="design",
        values=(0.25,),
        scores=(2.0,),
    )
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("design", 0.0, 1.0),
        campaigns.RealParameter("load", 1.0, 2.0),
    )

    with pytest.raises(ValueError, match="every non-acquired parameter"):
        plan.sampling_plan(space)

    sampling = plan.sampling_plan(space, fixed_parameters={"load": 1.5})
    assert sampling.samples == ({"load": 1.5, "design": 0.25},)


def test_high_fidelity_acquisition_must_match_plan_before_dataset_merge():
    training = _dataset([0.1, 0.9], prefix="train")
    acquired = _dataset([0.2, 0.4], prefix="agentfem")
    plan = ParameterPathAcquisitionPlan(
        parameter="design",
        values=(0.4, 0.2),
        scores=(2.0, 1.0),
    )

    result = merge_parameter_path_acquisition(training, acquired, plan)

    assert result.acquired_case_ids == acquired.case_ids
    assert result.training_dataset.case_count == 4
    assert result.acquisition_fingerprint == acquired.fingerprint
    assert result.summary()["training_fingerprint_after"] != training.fingerprint

    wrong = _dataset([0.2, 0.5], prefix="wrong")
    with pytest.raises(ValueError, match="do not match"):
        merge_parameter_path_acquisition(training, wrong, plan)


def test_candidate_acquisition_balances_risk_and_distance():
    plan = parameter_candidate_acquisition_plan(
        "design",
        values=(0.2, 0.4, 0.6),
        scores=(4.0, 3.0, 2.0),
        existing_values=(0.0, 0.2),
        maximum_candidates=2,
    )

    assert 0.2 not in plan.values
    assert len(plan.values) == 2
    assert plan.values[0] == pytest.approx(0.4)


def test_accepted_path_does_not_request_new_high_fidelity_cases():
    passed = verification.VerificationClaim.compare(
        name="parameter_path_reliability",
        observable="normalized path risk",
        actual=0.2,
        expected=0.0,
        reference="independent held-out path",
        absolute_tolerance=1.0,
        evidence={
            "parameter": "design",
            "parameter_values": [0.1, 0.3, 0.5],
            "case_risk": [0.1, 0.2, 0.1],
        },
    )

    plan = parameter_path_acquisition_plan(passed)

    assert plan.values == ()
    assert plan.reason == "path_claim_not_failed"
    with pytest.raises(ValueError, match="empty acquisition plan"):
        plan.sampling_plan(
            campaigns.ParameterSpace.create(campaigns.RealParameter("design", 0.0, 1.0))
        )


def test_ensemble_disagreement_is_per_case_and_not_labelled_as_calibrated_error():
    first = {
        "displacement": np.asarray([[[1.0, 0.0]], [[2.0, 0.0]]]),
        "stress": np.asarray([[[10.0]], [[20.0]]]),
    }
    second = {
        "displacement": np.asarray([[[1.0, 0.0]], [[4.0, 0.0]]]),
        "stress": np.asarray([[[10.0]], [[24.0]]]),
    }

    evidence = operator_ensemble_disagreement((first, second))

    assert evidence.member_count == 2
    assert evidence.case_count == 2
    assert evidence.case_risk[0] == pytest.approx(0.0)
    assert evidence.case_risk[1] > 0.0
    assert "not_calibrated_uncertainty" in evidence.summary()["semantics"]
