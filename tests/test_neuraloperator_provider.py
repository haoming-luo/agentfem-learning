from __future__ import annotations

import json

import numpy as np
import pytest
from agentfem import (
    datasets,
    extensions,
    learning,
    models,
    provenance,
    studies,
    verification,
)
from agentfem.step_providers import step_providers

from agentfem_learning.neural_operators.neuraloperator import (
    NeuralOperatorTrainingOptions,
    OperatorCheck,
    load_predictor,
    train_operator,
)
from agentfem_learning.neural_operators.neuraloperator.extension import extension


def _activate_extension():
    if any(item.name == "neuraloperator_field_operator" for item in step_providers()):
        return
    context = extensions.ExtensionContext(extension.spec)
    extension.register(context)
    context.commit()


def _operator_dataset(
    case_count: int = 18,
    spatial_size: int = 8,
    case_prefix: str = "heat",
    amplitude_bounds: tuple[float, float] = (0.5, 2.0),
):
    shape = (1, spatial_size, spatial_size)
    source = learning.FieldEncoding(
        name="source",
        role="input",
        unit="W/m^3",
        representation="structured_grid",
        shape=shape,
        mesh_policy="mesh_independent_coordinates",
    )
    temperature = learning.FieldEncoding(
        name="temperature_rise",
        role="output",
        unit="K",
        representation="structured_grid",
        shape=shape,
        mesh_policy="mesh_independent_coordinates",
    )
    x = np.linspace(0.0, 1.0, spatial_size)
    y = np.linspace(0.0, 1.0, spatial_size)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    spatial = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    amplitudes = np.linspace(*amplitude_bounds, case_count)
    inputs = np.asarray([(amplitude * spatial)[None, ...] for amplitude in amplitudes])
    outputs = 2.5 * inputs
    return (
        datasets.ScientificFieldDataset(
            case_ids=tuple(f"{case_prefix}-{index:03d}" for index in range(case_count)),
            encodings=(source, temperature),
            fields={"source": inputs, "temperature_rise": outputs},
            parameters={"gain": np.linspace(0.0, 1.0, case_count)},
            name="manufactured_heat_operator",
            metadata={"reference": "temperature_rise = 2.5 source"},
        ),
        learning.NeuralOperatorSpec(
            architecture="fno",
            inputs=(source,),
            outputs=(temperature,),
            boundary_encoding="structured_grid",
            parameter_inputs=("gain",),
            required_checks=("held_out_field_error",),
        ),
    )


def test_fno_step_trains_writes_evidence_and_reloads(tmp_path):
    _activate_extension()
    dataset, specification = _operator_dataset()
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="heat_field_operator",
    )
    output = tmp_path / "fno_heat"
    result = model.step(
        target=specification,
        dataset=dataset,
        n_modes=(3, 3),
        hidden_channels=8,
        n_layers=2,
        epochs=40,
        batch_size=4,
        learning_rate=5.0e-3,
        patience=15,
        relative_l2_tolerance=0.15,
        seed=22,
        device="cpu",
        output=output,
    ).solve_result()

    assert result.trust_level == "verified"
    assert result.quantity("validation_relative_l2_error") < 0.15
    assert (output / "operator_state.pt").is_file()
    assert (output / "held_out_fields.npz").is_file()
    assert (output / "training_ledger.json").is_file()
    manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["dataset"]["fingerprint"] == dataset.fingerprint
    assert manifest["metadata"]["data_quality"]["training"]["unique_input_count"] > 1
    assert result.quantity("training_duplicate_operator_input_count") == 0.0
    assert manifest["metadata"]["verification_coverage"]["missing"] == []
    assert provenance.verify_manifest(output / "result.json").verified is True

    predictor = load_predictor(output / "operator_state.pt")
    prediction = predictor.predict(
        {"source": dataset.fields["source"][:2]},
        parameters={"gain": dataset.parameters["gain"][:2]},
    )
    assert prediction["temperature_rise"].shape == (2, 1, 8, 8)
    relative_error = np.linalg.norm(
        prediction["temperature_rise"] - dataset.fields["temperature_rise"][:2]
    ) / np.linalg.norm(dataset.fields["temperature_rise"][:2])
    assert relative_error < 0.15

    with pytest.raises(ValueError, match="match the trained input names exactly"):
        predictor.predict(
            {"wrong_source": dataset.fields["source"][:2]},
            parameters={"gain": dataset.parameters["gain"][:2]},
        )
    with pytest.raises(ValueError, match="match the trained names exactly"):
        predictor.predict(
            {"source": dataset.fields["source"][:2]},
            parameters={"wrong_gain": dataset.parameters["gain"][:2]},
        )


def test_default_operator_checks_remain_explicitly_inconclusive(tmp_path):
    _activate_extension()
    dataset, base = _operator_dataset()
    specification = learning.NeuralOperatorSpec(
        architecture="fno",
        inputs=base.inputs,
        outputs=base.outputs,
        boundary_encoding="structured_grid",
    )
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="heat_field_operator_checks",
    )
    result = model.step(
        target=specification,
        dataset=dataset,
        n_modes=(2, 2),
        hidden_channels=4,
        n_layers=1,
        epochs=2,
        batch_size=6,
        patience=2,
        relative_l2_tolerance=10.0,
        seed=7,
        device="cpu",
        output=tmp_path / "checks",
    ).solve_result()

    missing = result.metadata["verification_coverage"]["missing"]
    assert "boundary_condition_error" in missing
    assert "conservation_or_balance_error" in missing
    assert "out_of_distribution_test" in missing
    assert any(claim.status == "inconclusive" for claim in result.verification.claims)
    assert result.trust_level in {"computed", "converged"}


def test_tfno_architecture_and_masked_geometry_fail_closed():
    dataset, base = _operator_dataset()
    specification = learning.NeuralOperatorSpec(
        architecture="tensorized_fno",
        inputs=base.inputs,
        outputs=base.outputs,
        parameter_inputs=base.parameter_inputs,
        boundary_encoding="structured_grid",
        required_checks=("held_out_field_error",),
    )
    outcome = train_operator(
        specification,
        dataset,
        NeuralOperatorTrainingOptions(
            n_modes=(2, 2),
            hidden_channels=4,
            n_layers=1,
            epochs=2,
            batch_size=6,
            patience=2,
            device="cpu",
        ),
    )
    assert outcome.model_configuration["architecture"] == "tfno"
    assert outcome.predictions["temperature_rise"].shape[0] > 0

    masked = datasets.ScientificFieldDataset(
        case_ids=dataset.case_ids,
        encodings=dataset.encodings,
        fields=dataset.fields,
        parameters=dataset.parameters,
        masks={
            "temperature_rise": np.ones(
                (dataset.case_count, 8, 8), dtype=bool
            )
        },
    )
    with pytest.raises(NotImplementedError, match="masked geometries"):
        train_operator(
            specification,
            masked,
            NeuralOperatorTrainingOptions(epochs=1, n_modes=(2, 2)),
        )


def _boundary_condition_check(context):
    prediction = context.predictions["temperature_rise"]
    boundary = np.concatenate(
        (
            prediction[:, :, 0, :].reshape(-1),
            prediction[:, :, -1, :].reshape(-1),
            prediction[:, :, :, 0].reshape(-1),
            prediction[:, :, :, -1].reshape(-1),
        )
    )
    return verification.VerificationClaim.compare(
        name="boundary_condition_error",
        observable="maximum_boundary_temperature_rise",
        actual=float(np.max(np.abs(boundary))),
        expected=0.0,
        reference="manufactured homogeneous Dirichlet boundary",
        absolute_tolerance=10.0,
        validity_domain="held-out manufactured heat fields",
    )


def test_named_physics_check_is_fingerprinted_and_satisfies_required_check(tmp_path):
    _activate_extension()
    dataset, base = _operator_dataset()
    specification = learning.NeuralOperatorSpec(
        architecture="fno",
        inputs=base.inputs,
        outputs=base.outputs,
        parameter_inputs=base.parameter_inputs,
        boundary_encoding="structured_grid",
        required_checks=("held_out_field_error", "boundary_condition_error"),
    )
    check = OperatorCheck(
        name="boundary_condition_error",
        evaluator=_boundary_condition_check,
        version="manufactured-v1",
    )
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="checked_heat_operator",
    )
    result = model.step(
        target=specification,
        dataset=dataset,
        check_evaluators=(check,),
        n_modes=(2, 2),
        hidden_channels=4,
        n_layers=1,
        epochs=2,
        relative_l2_tolerance=10.0,
        output=tmp_path / "checked",
    ).solve_result()

    assert result.metadata["verification_coverage"]["missing"] == ()
    evaluator = result.metadata["check_evaluators"][0]["evaluator"]
    assert evaluator["qualname"] == "_boundary_condition_check"
    assert "source_sha256" in evaluator
    assert result.verification.acceptable is True


def test_resolution_transfer_requires_a_different_spatial_resolution():
    dataset, base = _operator_dataset()
    same_resolution, _ = _operator_dataset(
        case_count=4,
        case_prefix="same-test",
        amplitude_bounds=(0.6, 1.9),
    )
    changed_resolution, _ = _operator_dataset(
        case_count=4,
        spatial_size=10,
        case_prefix="changed-test",
        amplitude_bounds=(0.6, 1.9),
    )
    options = NeuralOperatorTrainingOptions(
        n_modes=(2, 2),
        hidden_channels=4,
        n_layers=1,
        epochs=2,
        batch_size=6,
        patience=2,
        device="cpu",
    )

    same = train_operator(base, dataset, options, test_dataset=same_resolution)
    changed = train_operator(base, dataset, options, test_dataset=changed_resolution)

    assert same.resolution_transfer is False
    assert changed.resolution_transfer is True
    assert "test_relative_l2_error" in changed.metrics

    with pytest.raises(ValueError, match="case IDs must be disjoint"):
        train_operator(base, dataset, options, test_dataset=dataset.subset(range(4)))


def test_fno_split_keeps_replicas_atomic_and_rejects_conflicting_labels():
    base_dataset, specification = _operator_dataset(case_count=6)
    indices = np.asarray([0, 0, 2, 2, 5, 5])
    replicas = datasets.ScientificFieldDataset(
        case_ids=tuple(f"replica-{index}" for index in range(len(indices))),
        encodings=base_dataset.encodings,
        fields={
            name: np.asarray(values)[indices].copy()
            for name, values in base_dataset.fields.items()
        },
        parameters={
            name: np.asarray(values)[indices].copy()
            for name, values in base_dataset.parameters.items()
        },
        name="replicated_structured_operator",
    )
    options = NeuralOperatorTrainingOptions(
        n_modes=(2, 2),
        hidden_channels=4,
        n_layers=1,
        epochs=1,
        batch_size=2,
        validation_fraction=1.0 / 3.0,
        patience=1,
        device="cpu",
    )

    outcome = train_operator(specification, replicas, options)
    train_pairs = {int(case_id.rsplit("-", 1)[1]) // 2 for case_id in outcome.train_case_ids}
    validation_pairs = {
        int(case_id.rsplit("-", 1)[1]) // 2 for case_id in outcome.validation_case_ids
    }
    assert train_pairs.isdisjoint(validation_pairs)
    assert outcome.metrics["training_duplicate_operator_input_fraction"] == 0.5

    conflicting_fields = {
        name: np.asarray(values).copy() for name, values in replicas.fields.items()
    }
    conflicting_fields["temperature_rise"][1] *= 1.1
    conflicting = datasets.ScientificFieldDataset(
        case_ids=replicas.case_ids,
        encodings=replicas.encodings,
        fields=conflicting_fields,
        parameters=replicas.parameters,
        name=replicas.name,
    )
    with pytest.raises(ValueError, match="contradictory outputs"):
        train_operator(specification, conflicting, options)
