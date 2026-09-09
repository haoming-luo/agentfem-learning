from __future__ import annotations

import json

import numpy as np
import pytest
from agentfem import datasets, extensions, learning, models, provenance, studies
from agentfem.step_providers import step_providers

from agentfem_learning.neural_operators.neuraloperator import (
    GINOTrainingOptions,
    load_predictor,
    train_gino,
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
        coordinates={"nodes": coordinates},
        name="registered_rectangle_family",
        metadata={
            "topology": "registered rectangular point family",
            "coordinate_system": "cartesian",
            "coordinate_unit": "1",
        },
    )


def _specification(point_count=16):
    source, response = _encodings(point_count)
    return learning.NeuralOperatorSpec(
        architecture="gino",
        inputs=(source,),
        outputs=(response,),
        boundary_encoding="explicit point coordinates",
        required_checks=("held_out_field_error", "geometry_transfer"),
    )


def _options(**updates):
    values = {
        "input_geometry": "nodes",
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


def test_gino_trains_across_registered_geometries_and_reloads(tmp_path):
    _activate_extension()
    training = _dataset([0.8, 0.9, 1.0, 1.1], prefix="train")
    validation = _dataset([0.85, 1.05], prefix="validation")
    test = _dataset([0.95, 1.15], prefix="geometry-test")
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
    assert result.quantity("geometry_transfer_relative_l2_error") >= 0.0
    assert {claim.name for claim in result.verification.claims} == {
        "held_out_field_error",
        "geometry_transfer",
    }
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
    )
    assert predicted["temperature_rise"].shape == (2, 1, 16)
    assert np.isfinite(predicted["temperature_rise"]).all()


def test_gino_native_geometry_batch_and_fail_closed_contracts():
    dataset = _dataset([1.0] * 6, prefix="shared")
    outcome = train_gino(_specification(), dataset, _options())
    assert outcome.metrics["training_geometry_count"] == 1.0
    assert outcome.geometry_transfer is False

    with pytest.raises(ValueError, match="neighbor_backend='native'"):
        GINOTrainingOptions(neighbor_backend="open3d")

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
        )
