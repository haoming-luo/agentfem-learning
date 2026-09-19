"""Geometry-informed neural operators for registered point-field families."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...training import TrainingEpoch, TrainingLedger
from .api import _require_disjoint_case_ids, _resolve_device, _set_seed


@dataclass(frozen=True)
class GINOTrainingOptions:
    """Numerical and geometry choices for one GINO training run.

    Radii are measured after coordinates have been mapped to the unit box.
    Distinct geometries are evaluated as native micro-batches and contribute to
    one logical optimizer batch through gradient accumulation.
    """

    input_geometry: str = "input_geometry"
    output_queries: str | None = None
    coordinate_system: str = "cartesian"
    coordinate_unit: str = "1"
    latent_shape: tuple[int, ...] = (16, 16)
    n_modes: tuple[int, ...] = (8, 8)
    hidden_channels: int = 32
    n_layers: int = 4
    input_radius: float = 0.25
    output_radius: float = 0.25
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    validation_fraction: float = 0.2
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"
    patience: int = 20
    minimum_delta: float = 1.0e-6
    relative_l2_tolerance: float = 0.10
    progress: bool = False
    coordinate_bounds: tuple[tuple[float, float], ...] | None = None
    neighbor_backend: str = "native"

    def __post_init__(self) -> None:
        latent_shape = tuple(int(value) for value in self.latent_shape)
        n_modes = tuple(int(value) for value in self.n_modes)
        if not latent_shape or any(value < 2 for value in latent_shape):
            raise ValueError("latent_shape must contain at least two points per axis.")
        if len(n_modes) != len(latent_shape) or any(value < 1 for value in n_modes):
            raise ValueError("n_modes must be positive and match latent_shape dimension.")
        if any(mode > size for mode, size in zip(n_modes, latent_shape, strict=True)):
            raise ValueError("Each n_modes entry must not exceed its latent grid size.")
        if self.hidden_channels < 1 or self.n_layers < 1:
            raise ValueError("hidden_channels and n_layers must be positive.")
        if self.input_radius <= 0.0 or self.output_radius <= 0.0:
            raise ValueError("GINO neighborhood radii must be positive.")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly between zero and one.")
        if self.minimum_delta < 0.0 or self.relative_l2_tolerance <= 0.0:
            raise ValueError("Training tolerances must be non-negative and meaningful.")
        dtype = str(self.dtype).lower()
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        backend = str(self.neighbor_backend).lower().replace("-", "_")
        if backend != "native":
            raise ValueError(
                "The first GINO provider accepts neighbor_backend='native' only; "
                "accelerated backends require separate installed-wheel evidence."
            )
        bounds = self.coordinate_bounds
        if bounds is not None:
            bounds = tuple((float(lower), float(upper)) for lower, upper in bounds)
            if any(
                not np.isfinite((lower, upper)).all() or upper <= lower
                for lower, upper in bounds
            ):
                raise ValueError("coordinate_bounds must contain finite increasing pairs.")
        input_geometry = str(self.input_geometry).strip()
        output_queries = (
            None if self.output_queries is None else str(self.output_queries).strip()
        )
        coordinate_system = str(self.coordinate_system).strip()
        coordinate_unit = str(self.coordinate_unit).strip()
        if not input_geometry or output_queries == "":
            raise ValueError("GINO coordinate array names must not be empty.")
        if not coordinate_system or not coordinate_unit:
            raise ValueError("GINO coordinate_system and coordinate_unit must be explicit.")
        object.__setattr__(self, "latent_shape", latent_shape)
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "neighbor_backend", backend)
        object.__setattr__(self, "coordinate_bounds", bounds)
        object.__setattr__(self, "input_geometry", input_geometry)
        object.__setattr__(self, "output_queries", output_queries)
        object.__setattr__(self, "coordinate_system", coordinate_system)
        object.__setattr__(self, "coordinate_unit", coordinate_unit)


@dataclass(frozen=True)
class _PointStatistics:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray):
        mean = np.mean(values, axis=(0, 1), keepdims=True)
        scale = np.std(values, axis=(0, 1), keepdims=True)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        return cls(np.asarray(mean), np.asarray(scale))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.scale

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.mean

    def summary(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class _CoordinateTransform:
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, arrays: Sequence[np.ndarray], bounds=None):
        dimension = arrays[0].shape[-1]
        if bounds is None:
            lower = np.min(np.concatenate(arrays, axis=1), axis=(0, 1))
            upper = np.max(np.concatenate(arrays, axis=1), axis=(0, 1))
        else:
            if len(bounds) != dimension:
                raise ValueError(
                    f"coordinate_bounds has dimension {len(bounds)}, expected {dimension}."
                )
            lower = np.asarray([item[0] for item in bounds], dtype=float)
            upper = np.asarray([item[1] for item in bounds], dtype=float)
        if np.any(upper - lower <= np.finfo(float).eps):
            raise ValueError("Every GINO coordinate axis must have non-zero extent.")
        return cls(lower=np.asarray(lower), upper=np.asarray(upper))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.lower) / (self.upper - self.lower)

    def outside_fraction(self, values: np.ndarray) -> float:
        normalized = self.normalize(values)
        return float(np.mean(np.any((normalized < 0.0) | (normalized > 1.0), axis=-1)))

    def summary(self) -> dict[str, object]:
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}


@dataclass(frozen=True)
class GINOOutcome:
    """Trained GINO plus geometry-aware held-out evidence."""

    model: object
    ledger: TrainingLedger
    predictions: Mapping[str, np.ndarray]
    references: Mapping[str, np.ndarray]
    metrics: Mapping[str, float]
    input_statistics: _PointStatistics
    output_statistics: _PointStatistics
    coordinate_transform: _CoordinateTransform
    model_configuration: Mapping[str, object]
    geometry_configuration: Mapping[str, object]
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    validation_dataset: object
    test_case_ids: tuple[str, ...] = ()
    geometry_transfer: bool = False
    output_query_transfer: bool = False

    def write(self, path: str | Path, *, specification, dataset) -> dict[str, Path]:
        import torch

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "operator_state.pt"
        torch.save(
            {
                "state_dict": {
                    key: value
                    for key, value in self.model.state_dict().items()
                    if isinstance(value, torch.Tensor)
                },
                "model_configuration": dict(self.model_configuration),
                "geometry_configuration": dict(self.geometry_configuration),
                "input_statistics": self.input_statistics.summary(),
                "output_statistics": self.output_statistics.summary(),
                "coordinate_transform": self.coordinate_transform.summary(),
                "input_names": tuple(dataset.input_names),
                "parameter_names": tuple(specification.parameter_inputs),
                "output_names": tuple(dataset.output_names),
                "input_channels": {
                    name: int(dataset.fields[name].shape[1]) for name in dataset.input_names
                },
                "output_channels": {
                    name: int(dataset.fields[name].shape[1]) for name in dataset.output_names
                },
                "specification": specification.summary(),
                "dataset_fingerprint": dataset.fingerprint,
            },
            state_path,
        )
        prediction_path = output / "held_out_fields.npz"
        arrays: dict[str, np.ndarray] = {"case_ids": np.asarray(self.validation_case_ids)}
        input_name = str(self.geometry_configuration["input_geometry"])
        output_name = str(self.geometry_configuration["output_queries"])
        arrays["input_geometry"] = np.asarray(self.validation_dataset.coordinates[input_name])
        arrays["output_queries"] = np.asarray(self.validation_dataset.coordinates[output_name])
        for index, name in enumerate(self.predictions):
            arrays[f"prediction_{index}"] = self.predictions[name]
            arrays[f"reference_{index}"] = self.references[name]
        np.savez_compressed(prediction_path, **arrays)
        ledger_path = self.ledger.write(output / "training_ledger.json")
        metrics_path = output / "operator_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": dict(self.metrics),
                    "geometry": dict(self.geometry_configuration),
                    "train_case_ids": self.train_case_ids,
                    "validation_case_ids": self.validation_case_ids,
                    "test_case_ids": self.test_case_ids,
                    "dataset_fingerprint": dataset.fingerprint,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "model_state": state_path,
            "held_out_fields": prediction_path,
            "training_ledger": ledger_path,
            "operator_metrics": metrics_path,
        }


class GINOPredictor:
    """Reloaded geometry-informed operator with physical normalization."""

    def __init__(self, model, record: Mapping[str, object], *, device: str) -> None:
        self.model = model
        self.record = dict(record)
        self.device = str(device)
        self.input_names = tuple(record["input_names"])
        self.output_names = tuple(record["output_names"])
        self.input_statistics = _point_statistics_from_record(record["input_statistics"])
        self.output_statistics = _point_statistics_from_record(record["output_statistics"])
        transform = record["coordinate_transform"]
        self.coordinate_transform = _CoordinateTransform(
            lower=np.asarray(transform["lower"], dtype=float),
            upper=np.asarray(transform["upper"], dtype=float),
        )

    def predict(
        self,
        fields: Mapping[str, np.ndarray],
        *,
        input_geometry: np.ndarray,
        output_queries: np.ndarray | None = None,
        parameters: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        supplied = set(fields)
        expected = set(self.input_names)
        if supplied != expected:
            raise ValueError(
                "Prediction fields must match the trained input names exactly; "
                f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}."
            )
        arrays = [np.asarray(fields[name], dtype=float) for name in self.input_names]
        if not arrays or any(array.ndim != 3 for array in arrays):
            raise ValueError("Prediction inputs must use (case, channel, point) arrays.")
        case_count = arrays[0].shape[0]
        input_coordinates = _broadcast_coordinates(
            _as_case_coordinates(input_geometry, "input_geometry"), case_count
        )
        trained_input_name = self.record["geometry_configuration"]["input_geometry"]
        trained_output_name = self.record["geometry_configuration"]["output_queries"]
        if output_queries is None and trained_output_name != trained_input_name:
            raise ValueError(
                "This GINO was trained with independent output queries; "
                "output_queries=... is required for prediction."
            )
        output_coordinates = _as_case_coordinates(
            input_geometry if output_queries is None else output_queries,
            "output_queries",
        )
        output_coordinates = _broadcast_coordinates(output_coordinates, case_count)
        expected_dimension = int(self.record["model_configuration"]["gno_coord_dim"])
        if (
            input_coordinates.shape[-1] != expected_dimension
            or output_coordinates.shape[-1] != expected_dimension
        ):
            raise ValueError(
                f"Prediction coordinates must have dimension {expected_dimension}."
            )
        normalized_input = self.coordinate_transform.normalize(input_coordinates)
        normalized_output = self.coordinate_transform.normalize(output_coordinates)
        geometry = self.record["geometry_configuration"]
        _validate_neighborhood_coverage(
            normalized_input,
            normalized_output,
            latent_shape=tuple(geometry["latent_shape"]),
            input_radius=float(geometry["input_radius"]),
            output_radius=float(geometry["output_radius"]),
            label="prediction",
        )
        _validate_point_fields(
            arrays, case_count, input_coordinates.shape[1], "prediction inputs"
        )
        x = np.concatenate([array.transpose(0, 2, 1) for array in arrays], axis=-1)
        parameter_names = tuple(self.record.get("parameter_names", ()))
        x = _append_parameters(x, parameters, parameter_names)
        predictions = _predict_arrays(
            self.model,
            self.input_statistics.normalize(x),
            normalized_input,
            normalized_output,
            latent_shape=tuple(self.record["geometry_configuration"]["latent_shape"]),
            device=self.device,
            dtype_name=self.record["model_configuration"].get("dtype", "float32"),
        )
        physical = self.output_statistics.denormalize(predictions)
        return _split_point_outputs(
            physical,
            self.output_names,
            self.record["output_channels"],
        )


def train_gino(
    specification,
    dataset,
    options: GINOTrainingOptions,
    *,
    validation_dataset=None,
    test_dataset=None,
) -> GINOOutcome:
    """Train one GINO on registered point fields whose geometry may vary by case."""

    try:
        import neuralop
        import torch
        from neuralop.models import GINO
    except ImportError as exc:
        raise ImportError(
            "GINO training requires `agentfem-learning[neuraloperator]`."
        ) from exc

    _validate_gino_contract(specification, dataset, options)
    if validation_dataset is None:
        split = dataset.split(
            validation_fraction=options.validation_fraction,
            seed=options.seed,
        )
        training, validation = split.train, split.validation
    else:
        training, validation = dataset, validation_dataset
        _validate_gino_contract(specification, validation, options)
    if test_dataset is not None:
        _validate_gino_contract(specification, test_dataset, options)
    _require_disjoint_case_ids(training, validation, test_dataset)

    x_train, y_train, input_train, output_train = _point_data(specification, training, options)
    x_validation, y_validation, input_validation, output_validation = _point_data(
        specification, validation, options
    )
    coordinate_dimension = input_train.shape[-1]
    if len(options.latent_shape) != coordinate_dimension:
        raise ValueError(
            f"latent_shape has dimension {len(options.latent_shape)}, but coordinates "
            f"have dimension {coordinate_dimension}."
        )
    transform = _CoordinateTransform.fit((input_train, output_train), options.coordinate_bounds)
    input_statistics = _PointStatistics.fit(x_train)
    output_statistics = _PointStatistics.fit(y_train)
    x_train = input_statistics.normalize(x_train)
    y_train = output_statistics.normalize(y_train)
    input_train = transform.normalize(input_train)
    output_train = transform.normalize(output_train)
    _validate_neighborhood_coverage(
        input_train,
        output_train,
        latent_shape=options.latent_shape,
        input_radius=options.input_radius,
        output_radius=options.output_radius,
        label="training",
    )
    _validate_neighborhood_coverage(
        transform.normalize(input_validation),
        transform.normalize(output_validation),
        latent_shape=options.latent_shape,
        input_radius=options.input_radius,
        output_radius=options.output_radius,
        label="validation",
    )
    test_geometry = None
    if test_dataset is not None:
        _, _, test_input, test_output = _point_data(specification, test_dataset, options)
        normalized_test_input = transform.normalize(test_input)
        normalized_test_output = transform.normalize(test_output)
        _validate_neighborhood_coverage(
            normalized_test_input,
            normalized_test_output,
            latent_shape=options.latent_shape,
            input_radius=options.input_radius,
            output_radius=options.output_radius,
            label="test",
        )
        test_geometry = (normalized_test_input, normalized_test_output)

    device = _resolve_device(options.device, torch)
    dtype = torch.float64 if options.dtype == "float64" else torch.float32
    _set_seed(options.seed, torch)
    configuration = {
        "architecture": "gino",
        "in_channels": int(x_train.shape[-1]),
        "out_channels": int(y_train.shape[-1]),
        "gno_coord_dim": coordinate_dimension,
        "in_gno_radius": options.input_radius,
        "out_gno_radius": options.output_radius,
        "fno_in_channels": int(x_train.shape[-1]),
        "fno_n_modes": options.n_modes,
        "fno_hidden_channels": options.hidden_channels,
        "fno_n_layers": options.n_layers,
        "gno_use_open3d": False,
        "gno_use_torch_scatter": False,
        "in_gno_channel_mlp_hidden_layers": [options.hidden_channels] * 2,
        "out_gno_channel_mlp_hidden_layers": [options.hidden_channels] * 2,
        "dtype": options.dtype,
    }
    constructor = {
        key: value
        for key, value in configuration.items()
        if key not in {"architecture", "dtype"}
    }
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = GINO(**constructor)
    finally:
        torch.set_default_dtype(default_dtype)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    loss_function = torch.nn.MSELoss()
    ledger = TrainingLedger(
        seed=options.seed,
        device=device,
        dtype=options.dtype,
        framework="neuraloperator_gino",
        framework_version=neuralop.__version__,
    )
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    latent = _latent_grid(options.latent_shape, torch, dtype=dtype, device=device)
    groups = _geometry_groups(input_train, output_train)
    rng = random.Random(options.seed)
    for epoch in range(1, options.epochs + 1):
        model.train()
        micro_batches = []
        for indices in groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            micro_batches.extend(
                shuffled[start : start + options.batch_size]
                for start in range(0, len(shuffled), options.batch_size)
            )
        rng.shuffle(micro_batches)
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        weighted_loss = 0.0
        count = 0
        for position, indices in enumerate(micro_batches):
            index = np.asarray(indices, dtype=int)
            x_tensor = torch.as_tensor(x_train[index], dtype=dtype, device=device)
            y_tensor = torch.as_tensor(y_train[index], dtype=dtype, device=device)
            input_geometry = torch.as_tensor(
                input_train[index[0]][None, ...], dtype=dtype, device=device
            )
            output_queries = torch.as_tensor(
                output_train[index[0]][None, ...], dtype=dtype, device=device
            )
            prediction = model(input_geometry, latent, output_queries, x=x_tensor)
            loss = loss_function(prediction, y_tensor)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"GINO training became non-finite at epoch {epoch}.")
            (loss * len(index)).backward()
            accumulated += len(index)
            weighted_loss += float(loss.detach().cpu()) * len(index)
            count += len(index)
            is_last = position == len(micro_batches) - 1
            if accumulated >= options.batch_size or is_last:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0

        validation_prediction = _predict_arrays(
            model,
            input_statistics.normalize(x_validation),
            transform.normalize(input_validation),
            transform.normalize(output_validation),
            latent_shape=options.latent_shape,
            device=device,
            dtype_name=options.dtype,
        )
        validation_loss = float(
            np.mean((validation_prediction - output_statistics.normalize(y_validation)) ** 2)
        )
        improved = ledger.append(
            TrainingEpoch(
                epoch=epoch,
                training_loss=weighted_loss / max(count, 1),
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            ),
            minimum_delta=options.minimum_delta,
        )
        if improved:
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if options.progress and (epoch == 1 or epoch % 10 == 0):
            print(
                f"gino epoch {epoch}/{options.epochs} | "
                f"train={weighted_loss / max(count, 1):.4e} | "
                f"validation={validation_loss:.4e}",
                flush=True,
            )
        if epochs_without_improvement >= options.patience:
            ledger.stopped_early = True
            ledger.stop_reason = "validation_patience_exhausted"
            break
    model.load_state_dict(best_state)

    predictions, references, metrics = _evaluate_dataset(
        model,
        specification,
        validation,
        options,
        input_statistics,
        output_statistics,
        transform,
        device=device,
        prefix="validation",
    )
    metrics["validation_permutation_relative_l2_error"] = _permutation_consistency_error(
        model,
        input_statistics.normalize(x_validation),
        transform.normalize(input_validation),
        transform.normalize(output_validation),
        output_statistics,
        latent_shape=options.latent_shape,
        device=device,
        dtype_name=options.dtype,
    )
    train_fingerprints = set(_coordinate_fingerprints(input_train))
    validation_fingerprints = set(
        _coordinate_fingerprints(transform.normalize(input_validation))
    )
    geometry_transfer = bool(validation_fingerprints - train_fingerprints) and not bool(
        validation_fingerprints.intersection(train_fingerprints)
    )
    test_case_ids: tuple[str, ...] = ()
    output_query_transfer = False
    if test_dataset is not None:
        _, _, test_metrics = _evaluate_dataset(
            model,
            specification,
            test_dataset,
            options,
            input_statistics,
            output_statistics,
            transform,
            device=device,
            prefix="test",
        )
        metrics.update(test_metrics)
        test_case_ids = test_dataset.case_ids
        assert test_geometry is not None
        test_fingerprints = set(_coordinate_fingerprints(test_geometry[0]))
        if not test_fingerprints.intersection(train_fingerprints):
            geometry_transfer = True
            metrics["geometry_transfer_relative_l2_error"] = metrics["test_relative_l2_error"]
            metrics["geometry_transfer_maximum_case_relative_l2_error"] = metrics[
                "test_maximum_case_relative_l2_error"
            ]
        train_geometry_pairs = set(_geometry_groups(input_train, output_train))
        test_geometry_pairs = set(_geometry_groups(*test_geometry))
        output_query_transfer = test_fingerprints.issubset(train_fingerprints) and bool(
            test_geometry_pairs - train_geometry_pairs
        )
        if output_query_transfer:
            metrics["output_query_transfer_relative_l2_error"] = metrics[
                "test_relative_l2_error"
            ]
            metrics["output_query_transfer_maximum_case_relative_l2_error"] = metrics[
                "test_maximum_case_relative_l2_error"
            ]
    metrics.update(
        {
            "training_geometry_count": float(len(train_fingerprints)),
            "validation_geometry_count": float(len(validation_fingerprints)),
            "validation_coordinate_outside_training_fraction": transform.outside_fraction(
                np.concatenate((input_validation, output_validation), axis=1)
            ),
        }
    )
    geometry_configuration = {
        "input_geometry": options.input_geometry,
        "output_queries": options.output_queries or options.input_geometry,
        "coordinate_dimension": coordinate_dimension,
        "coordinate_system": options.coordinate_system,
        "coordinate_unit": options.coordinate_unit,
        "coordinate_bounds": transform.summary(),
        "latent_shape": options.latent_shape,
        "input_radius": options.input_radius,
        "output_radius": options.output_radius,
        "neighbor_backend": options.neighbor_backend,
        "batching": "exact_geometry_groups_with_gradient_accumulation",
        "variable_point_count": False,
    }
    return GINOOutcome(
        model=model,
        ledger=ledger,
        predictions=predictions,
        references=references,
        metrics=metrics,
        input_statistics=input_statistics,
        output_statistics=output_statistics,
        coordinate_transform=transform,
        model_configuration=configuration,
        geometry_configuration=geometry_configuration,
        train_case_ids=training.case_ids,
        validation_case_ids=validation.case_ids,
        validation_dataset=validation,
        test_case_ids=test_case_ids,
        geometry_transfer=geometry_transfer,
        output_query_transfer=output_query_transfer,
    )


def load_gino_predictor(path: str | Path, *, device: str = "cpu") -> GINOPredictor:
    try:
        import torch
        from neuralop.models import GINO
    except ImportError as exc:
        raise ImportError("Loading GINO requires `agentfem-learning[neuraloperator]`.") from exc
    selected_device = _resolve_device(device, torch)
    record = torch.load(Path(path), map_location=selected_device, weights_only=True)
    configuration = dict(record["model_configuration"])
    architecture = configuration.pop("architecture")
    if architecture != "gino":
        raise ValueError(f"Expected a GINO state, received architecture={architecture!r}.")
    dtype_name = configuration.pop("dtype", "float32")
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = GINO(**configuration)
    finally:
        torch.set_default_dtype(default_dtype)
    model.load_state_dict(record["state_dict"])
    model.to(selected_device)
    model.eval()
    return GINOPredictor(model, record, device=selected_device)


def _validate_gino_contract(specification, dataset, options: GINOTrainingOptions) -> None:
    if specification.architecture != "gino":
        raise ValueError("The geometry-aware provider requires architecture='gino'.")
    expected_inputs = tuple(item.name for item in specification.inputs)
    expected_outputs = tuple(item.name for item in specification.outputs)
    if tuple(dataset.input_names) != expected_inputs:
        raise ValueError(
            f"Dataset inputs {dataset.input_names!r} do not match {expected_inputs!r}."
        )
    if tuple(dataset.output_names) != expected_outputs:
        raise ValueError(
            f"Dataset outputs {dataset.output_names!r} do not match {expected_outputs!r}."
        )
    if dataset.masks:
        raise NotImplementedError(
            "GINO does not interpret padded point clouds as varying meshes. "
            "Use a registered fixed-point family or a future ragged dataset backend."
        )
    declared_system = dataset.metadata.get("coordinate_system")
    declared_unit = dataset.metadata.get("coordinate_unit")
    if declared_system is not None and str(declared_system) != options.coordinate_system:
        raise ValueError(
            "GINO coordinate_system disagrees with the dataset metadata: "
            f"{options.coordinate_system!r} != {declared_system!r}."
        )
    if declared_unit is not None and str(declared_unit) != options.coordinate_unit:
        raise ValueError(
            "GINO coordinate_unit disagrees with the dataset metadata: "
            f"{options.coordinate_unit!r} != {declared_unit!r}."
        )
    output_name = options.output_queries or options.input_geometry
    missing = {options.input_geometry, output_name}.difference(dataset.coordinates)
    if missing:
        raise ValueError(f"GINO dataset is missing coordinate arrays {sorted(missing)}.")
    input_geometry = _as_case_coordinates(
        dataset.coordinates[options.input_geometry], options.input_geometry
    )
    output_queries = _as_case_coordinates(dataset.coordinates[output_name], output_name)
    if (
        input_geometry.shape[0] != dataset.case_count
        or output_queries.shape[0] != dataset.case_count
    ):
        raise ValueError("GINO coordinate arrays must have one leading entry per case.")
    if input_geometry.shape[-1] != output_queries.shape[-1]:
        raise ValueError("Input geometry and output queries must share coordinate dimension.")
    if input_geometry.shape[-1] not in {2, 3}:
        raise ValueError(
            "The first GINO provider supports two- or three-dimensional coordinates."
        )
    input_arrays = [np.asarray(dataset.fields[name]) for name in expected_inputs]
    output_arrays = [np.asarray(dataset.fields[name]) for name in expected_outputs]
    _validate_point_fields(
        input_arrays, dataset.case_count, input_geometry.shape[1], "GINO inputs"
    )
    _validate_point_fields(
        output_arrays, dataset.case_count, output_queries.shape[1], "GINO outputs"
    )
    missing_parameters = set(specification.parameter_inputs).difference(dataset.parameters)
    if missing_parameters:
        raise ValueError(
            f"Dataset is missing operator parameters {sorted(missing_parameters)}."
        )
    for item in (*specification.inputs, *specification.outputs):
        encoding = dataset.encoding(item.name)
        expected = item.summary()
        for attribute in ("role", "unit", "representation", "geometry_encoding", "mesh_policy"):
            if encoding.get(attribute) != expected.get(attribute):
                raise ValueError(
                    f"Dataset field {item.name!r} disagrees with the specification for {attribute!r}."
                )
        if encoding["mesh_policy"] != "registered_mesh_family":
            raise ValueError(
                f"GINO field {item.name!r} must declare mesh_policy="
                "'registered_mesh_family'; arbitrary topology changes are not implied."
            )
        if encoding["representation"] not in {"point_samples", "mesh_dofs"}:
            raise ValueError(
                f"GINO field {item.name!r} requires point_samples or mesh_dofs representation."
            )


def _as_case_coordinates(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"Coordinate array {name!r} must use (case, point, dimension).")
    if not np.isfinite(array).all():
        raise ValueError(f"Coordinate array {name!r} contains non-finite values.")
    return array


def _validate_point_fields(arrays, case_count: int, point_count: int, label: str) -> None:
    if not arrays or any(array.ndim != 3 for array in arrays):
        raise ValueError(f"{label} must use (case, channel, point) arrays.")
    if any(array.shape[0] != case_count or array.shape[2] != point_count for array in arrays):
        raise ValueError(
            f"{label} must share case count {case_count} and point count {point_count}."
        )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{label} contain non-finite values.")


def _point_data(specification, dataset, options):
    input_geometry = _as_case_coordinates(
        dataset.coordinates[options.input_geometry], options.input_geometry
    )
    output_name = options.output_queries or options.input_geometry
    output_queries = _as_case_coordinates(dataset.coordinates[output_name], output_name)
    inputs = [
        np.asarray(dataset.fields[item.name], dtype=float).transpose(0, 2, 1)
        for item in specification.inputs
    ]
    outputs = [
        np.asarray(dataset.fields[item.name], dtype=float).transpose(0, 2, 1)
        for item in specification.outputs
    ]
    x = np.concatenate(inputs, axis=-1)
    parameter_values = (
        {name: dataset.parameters[name] for name in specification.parameter_inputs}
        if specification.parameter_inputs
        else None
    )
    x = _append_parameters(x, parameter_values, specification.parameter_inputs)
    return x, np.concatenate(outputs, axis=-1), input_geometry, output_queries


def _append_parameters(values, parameters, names):
    names = tuple(names)
    if not names:
        if parameters:
            raise ValueError(
                "This GINO was trained without parameter channels; "
                f"received {sorted(parameters)!r}."
            )
        return values
    if parameters is None:
        raise ValueError(f"This GINO requires parameters {names!r}.")
    supplied = set(parameters)
    expected = set(names)
    if supplied != expected:
        raise ValueError(
            "Prediction parameters must match the trained names exactly; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}."
        )
    channels = []
    for name in names:
        array = np.asarray(parameters[name], dtype=float)
        if array.shape[0] != values.shape[0]:
            raise ValueError(f"GINO parameter {name!r} must have {values.shape[0]} cases.")
        flattened = array.reshape(values.shape[0], -1)
        channels.append(
            np.broadcast_to(
                flattened[:, None, :], (values.shape[0], values.shape[1], flattened.shape[1])
            )
        )
    return np.concatenate((values, *channels), axis=-1)


def _geometry_groups(input_geometry, output_queries):
    groups = defaultdict(list)
    for index in range(input_geometry.shape[0]):
        digest = hashlib.sha256()
        for values in (input_geometry[index], output_queries[index]):
            contiguous = np.ascontiguousarray(values, dtype=np.float64)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        groups[digest.hexdigest()].append(index)
    return dict(groups)


def _coordinate_fingerprints(coordinates) -> tuple[str, ...]:
    fingerprints = []
    for values in coordinates:
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
        fingerprints.append(digest.hexdigest())
    return tuple(fingerprints)


def _broadcast_coordinates(values: np.ndarray, case_count: int) -> np.ndarray:
    if values.shape[0] == case_count:
        return values
    if values.shape[0] == 1:
        return np.broadcast_to(values, (case_count, *values.shape[1:]))
    raise ValueError(
        f"Coordinate arrays must provide one shared geometry or {case_count} cases; "
        f"received {values.shape[0]}."
    )


def _latent_grid(shape, torch, *, dtype, device):
    axes = [torch.linspace(0.0, 1.0, size, dtype=dtype, device=device) for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)[None, ...]


def _numpy_latent_grid(shape) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, int(size)) for size in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(shape))


def _validate_neighborhood_coverage(
    input_geometry,
    output_queries,
    *,
    latent_shape,
    input_radius,
    output_radius,
    label,
) -> None:
    latent = _numpy_latent_grid(latent_shape)
    worst_input = 0.0
    worst_output = 0.0
    for case in range(input_geometry.shape[0]):
        worst_input = max(
            worst_input,
            _maximum_nearest_distance(latent, input_geometry[case]),
        )
        worst_output = max(
            worst_output,
            _maximum_nearest_distance(output_queries[case], latent),
        )
    tolerance = 64.0 * np.finfo(float).eps
    if worst_input > input_radius + tolerance or worst_output > output_radius + tolerance:
        raise ValueError(
            f"GINO {label} geometry contains queries without graph neighbors: "
            f"required input_radius>={worst_input:.6g}, "
            f"output_radius>={worst_output:.6g}; received "
            f"{input_radius:.6g} and {output_radius:.6g}."
        )


def _maximum_nearest_distance(queries, sources, *, chunk_size: int = 4096) -> float:
    maximum = 0.0
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - sources[None, :, :]) ** 2, axis=-1)
        maximum = max(maximum, float(np.sqrt(np.min(squared, axis=1)).max()))
    return maximum


def _predict_arrays(
    model, x, input_geometry, output_queries, *, latent_shape, device, dtype_name
):
    import torch

    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    latent = _latent_grid(latent_shape, torch, dtype=dtype, device=device)
    groups = _geometry_groups(input_geometry, output_queries)
    prediction = None
    model.eval()
    with torch.no_grad():
        for indices in groups.values():
            index = np.asarray(indices, dtype=int)
            values = (
                model(
                    torch.as_tensor(
                        input_geometry[index[0]][None, ...], dtype=dtype, device=device
                    ),
                    latent,
                    torch.as_tensor(
                        output_queries[index[0]][None, ...], dtype=dtype, device=device
                    ),
                    x=torch.as_tensor(x[index], dtype=dtype, device=device),
                )
                .detach()
                .cpu()
                .numpy()
            )
            if prediction is None:
                prediction = np.empty(
                    (x.shape[0], values.shape[1], values.shape[2]), dtype=values.dtype
                )
            prediction[index] = values
    if prediction is None:
        raise ValueError("GINO prediction requires at least one case.")
    return prediction


def _permutation_consistency_error(
    model,
    x,
    input_geometry,
    output_queries,
    output_statistics,
    *,
    latent_shape,
    device,
    dtype_name,
) -> float:
    """Measure point-order dependence without using reference solution fields."""

    baseline = _predict_arrays(
        model,
        x,
        input_geometry,
        output_queries,
        latent_shape=latent_shape,
        device=device,
        dtype_name=dtype_name,
    )
    input_permutation = np.arange(input_geometry.shape[1])[::-1]
    output_permutation = np.roll(
        np.arange(output_queries.shape[1]), max(output_queries.shape[1] // 3, 1)
    )
    permuted = _predict_arrays(
        model,
        x[:, input_permutation, :],
        input_geometry[:, input_permutation, :],
        output_queries[:, output_permutation, :],
        latent_shape=latent_shape,
        device=device,
        dtype_name=dtype_name,
    )
    restored = np.empty_like(permuted)
    restored[:, output_permutation, :] = permuted
    baseline_physical = output_statistics.denormalize(baseline)
    restored_physical = output_statistics.denormalize(restored)
    denominator = max(float(np.linalg.norm(baseline_physical)), np.finfo(float).eps)
    return float(np.linalg.norm(restored_physical - baseline_physical) / denominator)


def _evaluate_dataset(
    model,
    specification,
    dataset,
    options,
    input_statistics,
    output_statistics,
    transform,
    *,
    device,
    prefix,
):
    x, y, input_geometry, output_queries = _point_data(specification, dataset, options)
    normalized = _predict_arrays(
        model,
        input_statistics.normalize(x),
        transform.normalize(input_geometry),
        transform.normalize(output_queries),
        latent_shape=options.latent_shape,
        device=device,
        dtype_name=options.dtype,
    )
    physical = output_statistics.denormalize(normalized)
    denominator = max(float(np.linalg.norm(y)), np.finfo(float).eps)
    residual = physical - y
    case_axes = tuple(range(1, residual.ndim))
    case_denominator = np.maximum(np.linalg.norm(y, axis=case_axes), np.finfo(float).eps)
    case_relative_l2 = np.linalg.norm(residual, axis=case_axes) / case_denominator
    metrics = {
        f"{prefix}_relative_l2_error": float(np.linalg.norm(residual) / denominator),
        f"{prefix}_maximum_case_relative_l2_error": float(np.max(case_relative_l2)),
        f"{prefix}_median_case_relative_l2_error": float(np.median(case_relative_l2)),
        f"{prefix}_p95_case_relative_l2_error": float(np.percentile(case_relative_l2, 95.0)),
    }
    start = 0
    for item in specification.outputs:
        channels = int(dataset.fields[item.name].shape[1])
        stop = start + channels
        output_residual = residual[..., start:stop]
        output_reference = y[..., start:stop]
        output_denominator = np.maximum(
            np.linalg.norm(output_reference, axis=case_axes), np.finfo(float).eps
        )
        output_relative_l2 = (
            np.linalg.norm(output_residual, axis=case_axes) / output_denominator
        )
        metrics[f"{prefix}_{item.name}_maximum_case_relative_l2_error"] = float(
            np.max(output_relative_l2)
        )
        metrics[f"{prefix}_{item.name}_median_case_relative_l2_error"] = float(
            np.median(output_relative_l2)
        )
        start = stop
    return (
        _split_point_outputs(
            physical,
            tuple(item.name for item in specification.outputs),
            {item.name: dataset.fields[item.name].shape[1] for item in specification.outputs},
        ),
        {item.name: np.asarray(dataset.fields[item.name]) for item in specification.outputs},
        metrics,
    )


def _split_point_outputs(values, names, channels):
    result = {}
    start = 0
    for name in names:
        stop = start + int(channels[name])
        result[name] = values[..., start:stop].transpose(0, 2, 1)
        start = stop
    return result


def _point_statistics_from_record(record):
    return _PointStatistics(
        mean=np.asarray(record["mean"], dtype=float),
        scale=np.asarray(record["scale"], dtype=float),
    )


__all__ = [
    "GINOOutcome",
    "GINOPredictor",
    "GINOTrainingOptions",
    "load_gino_predictor",
    "train_gino",
]
