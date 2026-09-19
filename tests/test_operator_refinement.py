from __future__ import annotations

import numpy as np
import pytest
from agentfem import datasets, learning

from agentfem_learning.neural_operators import (
    ParameterPathRefinementPlan,
    apply_parameter_path_refinement,
    merge_operator_datasets,
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
    assert refinement.validation_dataset.parameters["design"] == pytest.approx(
        [0.2, 0.4, 0.6]
    )
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
