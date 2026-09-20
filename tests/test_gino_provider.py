from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from agentfem import datasets, extensions, learning, models, provenance, studies
from agentfem.step_providers import step_providers

from agentfem_learning.neural_operators import apply_parameter_path_refinement
from agentfem_learning.neural_operators.neuraloperator import (
    GINOTrainingOptions,
    load_predictor,
    parameter_path_refinement_plan,
    parameter_path_reliability_check,
    train_gino,
)
from agentfem_learning.neural_operators.neuraloperator.checks import (
    OperatorCheckContext,
)
from agentfem_learning.neural_operators.neuraloperator.extension import extension


def _activate_extension():
    if any(item.name == "neuraloperator_geometry_operator" for item in step_providers()):
        return
    context = extensions.ExtensionContext(extension.spec)
    extension.register(context)
    context.commit()


def _encodings(point_count=16):
    source = learning.FieldEncoding(
        name="source",
        role="input",
        unit="W/m^3",
        representation="point_samples",
        shape=(1, point_count),
        mesh_policy="registered_mesh_family",
    )
    response = learning.FieldEncoding(
        name="temperature_rise",
        role="output",
        unit="K",
        representation="point_samples",
        shape=(1, point_count),
        mesh_policy="registered_mesh_family",
    )
    return source, response


def _dataset(scales, *, prefix, point_count=16):
    source_encoding, response_encoding = _encodings(point_count)
    side = round(point_count**0.5)
    base = np.stack(
        np.meshgrid(np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    coordinates = np.asarray(
        [base * np.asarray([float(scale), 2.0 - 0.25 * float(scale)]) for scale in scales]
    )
    source = np.asarray(
        [
            (
                np.sin(np.pi * points[:, 0] / scale)
                * np.sin(np.pi * points[:, 1] / (2.0 - 0.25 * scale))
            )[None, :]
            for scale, points in zip(scales, coordinates, strict=True)
        ]
    )
    response = 2.0 * source
    return datasets.ScientificFieldDataset(
        case_ids=tuple(f"{prefix}-{index:03d}" for index in range(len(scales))),
        encodings=(source_encoding, response_encoding),
        fields={"source": source, "temperature_rise": response},
        parameters={"geometry_scale": np.asarray(scales)},
        coordinates={"nodes": coordinates, "queries": coordinates.copy()},
        name="registered_rectangle_family",
        metadata={
            "topology": "registered rectangular point family",
            "coordinate_system": "cartesian",
            "coordinate_unit": "1",
        },
    )


def _specification(point_count=16, *, parameter_inputs=()):
    source, response = _encodings(point_count)
    return learning.NeuralOperatorSpec(
        architecture="gino",
        inputs=(source,),
        outputs=(response,),
        boundary_encoding="explicit point coordinates",
        parameter_inputs=tuple(parameter_inputs),
        required_checks=("held_out_field_error", "geometry_transfer"),
    )


def _query_dataset(scales, *, prefix, output_side=5):
    dataset = _dataset(scales, prefix=prefix)
    base = np.stack(
        np.meshgrid(
            np.linspace(0.0, 1.0, output_side),
            np.linspace(0.0, 1.0, output_side),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    queries = np.asarray(
        [base * np.asarray([float(scale), 2.0 - 0.25 * float(scale)]) for scale in scales]
    )
    response = np.asarray(
        [
            (
                2.0
                * np.sin(np.pi * points[:, 0] / scale)
                * np.sin(np.pi * points[:, 1] / (2.0 - 0.25 * scale))
            )[None, :]
            for scale, points in zip(scales, queries, strict=True)
        ]
    )
    source_encoding = dataset.encodings[0]
    response_encoding = learning.FieldEncoding(
        name="temperature_rise",
        role="output",
        unit="K",
        representation="point_samples",
        shape=(1, output_side**2),
        mesh_policy="registered_mesh_family",
    )
    return datasets.ScientificFieldDataset(
        case_ids=dataset.case_ids,
        encodings=(source_encoding, response_encoding),
        fields={"source": dataset.fields["source"], "temperature_rise": response},
        coordinates={"nodes": dataset.coordinates["nodes"], "queries": queries},
        name="independent_output_query_family",
        metadata=dataset.metadata,
    )


def _options(**updates):
    values = {
        "input_geometry": "nodes",
        "output_queries": "queries",
        "latent_shape": (4, 4),
        "n_modes": (2, 2),
        "hidden_channels": 4,
        "n_layers": 1,
        "input_radius": 0.8,
        "output_radius": 0.8,
        "epochs": 2,
        "batch_size": 2,
        "patience": 2,
        "device": "cpu",
        "relative_l2_tolerance": 10.0,
        "seed": 17,
    }
    values.update(updates)
    return GINOTrainingOptions(**values)


def _parameter_path_context(errors):
    errors = np.asarray(errors, dtype=float)
    reference = np.ones((len(errors), 1, 4), dtype=float)
    prediction = reference * (1.0 + errors[:, None, None])
    dataset = SimpleNamespace(
        case_ids=tuple(f"radius-{value:.2f}" for value in np.linspace(0.1, 0.5, len(errors))),
        parameters={"hole_radius": np.linspace(0.1, 0.5, len(errors))},
    )
    return OperatorCheckContext(
        specification=object(),
        training_dataset=object(),
        validation_dataset=dataset,
        predictions={"stress": prediction},
        references={"stress": reference},
        metrics={},
    )


def test_parameter_path_check_accepts_smooth_held_out_errors():
    check = parameter_path_reliability_check(
        "hole_radius",
        output_tolerances={"stress": 0.10},
        maximum_spike_ratio=2.0,
    )
    claim = check.evaluate(_parameter_path_context([0.02, 0.03, 0.04, 0.03, 0.02]))

    assert claim.status == "passed"
    assert claim.evidence["worst_case_id"] == "radius-0.30"
    assert claim.evidence["output_relative_l2"]["stress"] == pytest.approx(
        [0.02, 0.03, 0.04, 0.03, 0.02]
    )
    assert check.summary()["metadata"]["parameter"] == "hole_radius"


def test_parameter_path_check_rejects_an_interior_error_spike():
    check = parameter_path_reliability_check(
        "hole_radius",
        output_tolerances={"stress": 0.10},
        maximum_spike_ratio=2.0,
    )
    claim = check.evaluate(_parameter_path_context([0.02, 0.025, 0.08, 0.025, 0.02]))

    assert claim.status == "failed"
    assert claim.evidence["spike_case_id"] == "radius-0.30"
    assert claim.evidence["maximum_spike_ratio"] > 3.0
    assert claim.actual[0] < 1.0
    assert claim.actual[1] > 1.0


def test_parameter_path_check_fails_closed_for_duplicate_coordinates():
    context = _parameter_path_context([0.01, 0.02, 0.03])
    context.validation_dataset.parameters["hole_radius"] = np.asarray([0.1, 0.1, 0.2])
    check = parameter_path_reliability_check("hole_radius")

    with pytest.raises(ValueError, match="must be unique"):
        check.evaluate(context)


def test_parameter_path_refinement_selects_diverse_failed_cases():
    check = parameter_path_reliability_check(
        "hole_radius",
        output_tolerances={"stress": 0.05},
        maximum_spike_ratio=2.0,
    )
    claim = check.evaluate(_parameter_path_context([0.01, 0.04, 0.10, 0.06, 0.01]))

    plan = parameter_path_refinement_plan(
        claim,
        existing_values=(0.1, 0.5),
        maximum_candidates=2,
    )

    assert plan.values[0] == pytest.approx(0.3)
    assert len(plan.values) == 2
    assert all(value not in {0.1, 0.5} for value in plan.values)
    assert plan.summary()["kind"] == "parameter_path_refinement_plan"


def test_parameter_path_refinement_is_empty_after_acceptance():
    check = parameter_path_reliability_check("hole_radius", maximum_relative_l2=0.20)
    claim = check.evaluate(_parameter_path_context([0.01, 0.02, 0.03]))

    plan = parameter_path_refinement_plan(claim)

    assert plan.values == ()
    assert plan.reason == "path_claim_not_failed"


def test_parameter_path_check_enters_gino_result_lifecycle(tmp_path):
    _activate_extension()
    training = _dataset([0.8, 1.0, 1.2], prefix="train-path")
    validation = _dataset([0.85, 0.90, 0.95], prefix="validation-path")
    source, response = _encodings()
    specification = learning.NeuralOperatorSpec(
        architecture="gino",
        inputs=(source,),
        outputs=(response,),
        boundary_encoding="explicit point coordinates",
        parameter_inputs=("geometry_scale",),
        required_checks=(
            "held_out_field_error",
            "geometry_transfer",
            "parameter_path_reliability",
        ),
    )
    check = parameter_path_reliability_check(
        "geometry_scale",
        output_tolerances={"temperature_rise": 10.0},
        maximum_spike_ratio=1.0e6,
    )
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="geometry_path_audit",
    )
    output = tmp_path / "path-audit"
    result = model.step(
        target=specification,
        dataset=training,
        validation_dataset=validation,
        input_geometry="nodes",
        output_queries="queries",
        latent_shape=(4, 4),
        n_modes=(2, 2),
        hidden_channels=4,
        n_layers=1,
        input_radius=0.8,
        output_radius=0.8,
        epochs=2,
        batch_size=2,
        patience=2,
        device="cpu",
        relative_l2_tolerance=10.0,
        check_evaluators=(check,),
        output=output,
    ).solve_result()

    claims = {item.name: item for item in result.verification.claims}
    assert claims["parameter_path_reliability"].status == "passed"
    assert claims["parameter_path_reliability"].evidence["parameter_values"] == [
        0.85,
        0.90,
        0.95,
    ]
    assert result.metadata["verification_coverage"]["missing"] == ()
    manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
    names = {item["name"] for item in manifest["verification"]["claims"]}
    assert "parameter_path_reliability" in names


def test_failed_path_plan_refines_retrains_and_rechecks_independent_cases():
    training = _dataset([0.8, 1.2], prefix="cycle-train")
    validation = _dataset([0.85, 0.90, 0.95, 1.00, 1.05], prefix="cycle-path")
    reference = validation.fields["temperature_rise"]
    synthetic_prediction = reference * np.asarray(
        [1.01, 1.04, 1.14, 1.08, 1.01]
    )[:, None, None]
    strict_check = parameter_path_reliability_check(
        "geometry_scale",
        output_tolerances={"temperature_rise": 0.05},
        maximum_spike_ratio=2.0,
    )
    failed = strict_check.evaluate(
        OperatorCheckContext(
            specification=_specification(parameter_inputs=("geometry_scale",)),
            training_dataset=training,
            validation_dataset=validation,
            predictions={"temperature_rise": synthetic_prediction},
            references={"temperature_rise": reference},
            metrics={},
        )
    )
    plan = parameter_path_refinement_plan(
        failed,
        existing_values=training.parameters["geometry_scale"],
        maximum_candidates=2,
    )
    refinement = apply_parameter_path_refinement(training, validation, plan)

    outcome = train_gino(
        _specification(parameter_inputs=("geometry_scale",)),
        refinement.training_dataset,
        _options(epochs=1),
        validation_dataset=refinement.validation_dataset,
    )
    permissive_check = parameter_path_reliability_check(
        "geometry_scale",
        output_tolerances={"temperature_rise": 10.0},
        maximum_spike_ratio=1.0e6,
    )
    rechecked = permissive_check.evaluate(
        OperatorCheckContext(
            specification=_specification(parameter_inputs=("geometry_scale",)),
            training_dataset=refinement.training_dataset,
            validation_dataset=refinement.validation_dataset,
            predictions=outcome.predictions,
            references=outcome.references,
            metrics=outcome.metrics,
        )
    )

    assert failed.status == "failed"
    assert len(refinement.added_case_ids) == 2
    assert set(outcome.train_case_ids).isdisjoint(outcome.validation_case_ids)
    assert rechecked.status == "passed"


def test_gino_trains_across_registered_geometries_and_reloads(tmp_path):
    _activate_extension()
    training = _dataset([0.8, 0.9, 1.0, 1.1], prefix="train")
    validation = _dataset([0.85, 1.05], prefix="validation")
    test = _query_dataset([1.0], prefix="query-test")
    specification = _specification()
    output = tmp_path / "gino"
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="geometry_operator",
    )
    result = model.step(
        target=specification,
        dataset=training,
        validation_dataset=validation,
        test_dataset=test,
        input_geometry="nodes",
        output_queries="queries",
        latent_shape=(4, 4),
        n_modes=(2, 2),
        hidden_channels=4,
        n_layers=1,
        input_radius=0.8,
        output_radius=0.8,
        epochs=2,
        batch_size=2,
        patience=2,
        device="cpu",
        relative_l2_tolerance=10.0,
        seed=17,
        output=output,
    ).solve_result()

    assert result.metadata["capability"]["supports_geometry_varying_meshes"] is True
    assert result.metadata["geometry"]["batching"].startswith("exact_geometry_groups")
    assert result.quantity("training_geometry_count") == 4.0
    assert result.quantity("validation_geometry_count") == 2.0
    assert result.quantity("validation_relative_l2_error") >= 0.0
    assert result.quantity("validation_maximum_case_relative_l2_error") >= result.quantity(
        "validation_median_case_relative_l2_error"
    )
    assert result.quantity(
        "validation_temperature_rise_maximum_case_relative_l2_error"
    ) >= result.quantity("validation_temperature_rise_median_case_relative_l2_error")
    assert result.quantity("output_query_transfer_relative_l2_error") >= 0.0
    assert result.quantity("output_query_transfer_maximum_case_relative_l2_error") >= 0.0
    assert result.quantity("validation_permutation_relative_l2_error") < 5.0e-5
    claims = {claim.name: claim for claim in result.verification.claims}
    assert set(claims) == {
        "operator_dataset_integrity",
        "held_out_field_error",
        "geometry_transfer",
        "output_query_transfer",
        "permutation_equivariance",
    }
    assert claims["held_out_field_error"].observable == (
        "validation_maximum_case_relative_l2_error"
    )
    assert claims["held_out_field_error"].evidence["global_relative_l2_error"] == (
        result.quantity("validation_relative_l2_error")
    )
    assert claims["geometry_transfer"].observable == (
        "validation_maximum_case_relative_l2_error"
    )
    assert claims["output_query_transfer"].observable == (
        "output_query_transfer_maximum_case_relative_l2_error"
    )
    assert claims["permutation_equivariance"].status == "passed"
    assert claims["operator_dataset_integrity"].status == "passed"
    assert claims["operator_dataset_integrity"].evidence[
        "training_unique_input_count"
    ] == 4
    assert provenance.verify_manifest(output / "result.json").verified is True
    manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["method"] == "gino"
    held_out = np.load(output / "held_out_fields.npz")
    assert held_out["input_geometry"].shape == (2, 16, 2)
    assert held_out["output_queries"].shape == (2, 16, 2)

    predictor = load_predictor(output / "operator_state.pt")
    predicted = predictor.predict(
        {"source": validation.fields["source"]},
        input_geometry=validation.coordinates["nodes"],
        output_queries=validation.coordinates["queries"],
    )
    assert predicted["temperature_rise"].shape == (2, 1, 16)
    assert np.isfinite(predicted["temperature_rise"]).all()

    input_permutation = np.arange(16)[::-1]
    output_permutation = np.roll(np.arange(16), 5)
    permuted = predictor.predict(
        {"source": validation.fields["source"][:, :, input_permutation]},
        input_geometry=validation.coordinates["nodes"][:, input_permutation, :],
        output_queries=validation.coordinates["queries"][:, output_permutation, :],
    )["temperature_rise"]
    restored = np.empty_like(permuted)
    restored[:, :, output_permutation] = permuted
    assert restored == pytest.approx(predicted["temperature_rise"], abs=2.0e-5)


def test_gino_native_geometry_batch_and_fail_closed_contracts():
    dataset = _dataset([1.0] * 6, prefix="shared")
    validation = _dataset([1.1, 1.1], prefix="shared-validation")
    outcome = train_gino(
        _specification(parameter_inputs=("geometry_scale",)),
        dataset,
        _options(),
        validation_dataset=validation,
    )
    assert outcome.metrics["training_geometry_count"] == 1.0
    assert outcome.metrics["training_case_count"] == 6.0
    assert outcome.metrics["training_unique_operator_input_count"] == 1.0
    assert outcome.metrics["training_duplicate_operator_input_fraction"] == pytest.approx(
        5.0 / 6.0
    )
    assert outcome.data_quality["partition_policy"] == (
        "exact_input_groups_are_partition_atomic"
    )
    assert outcome.geometry_transfer is True

    with pytest.raises(ValueError, match="neighbor_backend='native'"):
        GINOTrainingOptions(neighbor_backend="open3d")

    with pytest.raises(ValueError, match="output_weighting_function"):
        GINOTrainingOptions(output_weighting_function="gaussian")

    weighted = train_gino(
        _specification(parameter_inputs=("geometry_scale",)),
        dataset,
        _options(
            epochs=1,
            output_weighting_function="half_cos",
            output_weighting_scale=0.75,
        ),
        validation_dataset=validation,
    )
    assert weighted.model_configuration["gno_weighting_function"] == "half_cos"
    assert weighted.geometry_configuration["output_weighting_scale"] == 0.75

    missing_coordinates = datasets.ScientificFieldDataset(
        case_ids=dataset.case_ids,
        encodings=dataset.encodings,
        fields=dataset.fields,
    )
    with pytest.raises(ValueError, match="missing coordinate arrays"):
        train_gino(_specification(), missing_coordinates, _options(epochs=1))

    masked = datasets.ScientificFieldDataset(
        case_ids=dataset.case_ids,
        encodings=dataset.encodings,
        fields=dataset.fields,
        coordinates=dataset.coordinates,
        masks={"source": np.ones((dataset.case_count, 16), dtype=bool)},
    )
    with pytest.raises(NotImplementedError, match="padded point clouds"):
        train_gino(_specification(), masked, _options(epochs=1))

    with pytest.raises(ValueError, match="without graph neighbors"):
        train_gino(
            _specification(),
            dataset,
            _options(
                epochs=1,
                latent_shape=(5, 5),
                input_radius=1.0e-6,
                output_radius=1.0e-6,
            ),
            validation_dataset=validation,
        )


def test_gino_automatic_split_keeps_duplicate_inputs_partition_atomic():
    dataset = _dataset([0.8, 0.8, 1.0, 1.0, 1.2, 1.2], prefix="replica")
    outcome = train_gino(
        _specification(parameter_inputs=("geometry_scale",)),
        dataset,
        _options(validation_fraction=1.0 / 3.0, epochs=1),
    )

    assert set(outcome.train_case_ids).isdisjoint(outcome.validation_case_ids)
    train_scales = {
        float(dataset.parameters["geometry_scale"][dataset.case_ids.index(case_id)])
        for case_id in outcome.train_case_ids
    }
    validation_scales = {
        float(dataset.parameters["geometry_scale"][dataset.case_ids.index(case_id)])
        for case_id in outcome.validation_case_ids
    }
    assert train_scales.isdisjoint(validation_scales)
    assert outcome.metrics["training_unique_operator_input_count"] == 2.0
    assert outcome.metrics["validation_unique_operator_input_count"] == 1.0


def test_gino_rejects_contradictory_duplicate_operator_labels():
    dataset = _dataset([1.0, 1.0, 1.2, 1.2], prefix="conflict")
    fields = {name: np.asarray(values).copy() for name, values in dataset.fields.items()}
    fields["temperature_rise"][1] *= 1.2
    conflicting = datasets.ScientificFieldDataset(
        case_ids=dataset.case_ids,
        encodings=dataset.encodings,
        fields=fields,
        parameters=dataset.parameters,
        coordinates=dataset.coordinates,
        name=dataset.name,
        metadata=dataset.metadata,
    )

    with pytest.raises(ValueError, match="contradictory outputs"):
        train_gino(
            _specification(parameter_inputs=("geometry_scale",)),
            conflicting,
            _options(epochs=1),
        )


def test_gino_rejects_exact_input_leakage_across_explicit_partitions():
    training = _dataset([0.8, 1.0], prefix="train-leak")
    validation = _dataset([1.0, 1.2], prefix="validation-leak")

    with pytest.raises(ValueError, match="share 1 exact declared operator input"):
        train_gino(
            _specification(parameter_inputs=("geometry_scale",)),
            training,
            _options(epochs=1),
            validation_dataset=validation,
        )
