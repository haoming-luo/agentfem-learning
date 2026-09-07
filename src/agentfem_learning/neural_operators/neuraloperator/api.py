"""Training and reload API for the official NeuralOperator provider."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...training import TrainingEpoch, TrainingLedger


@dataclass(frozen=True)
class NeuralOperatorTrainingOptions:
    """Compact numerical choices for a supervised FNO or TFNO run."""

    n_modes: tuple[int, ...] = (8, 8)
    hidden_channels: int = 32
    n_layers: int = 4
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

    def __post_init__(self) -> None:
        modes = tuple(int(value) for value in self.n_modes)
        if not modes or any(value < 1 for value in modes):
            raise ValueError("n_modes must contain positive spatial mode counts.")
        if self.hidden_channels < 1 or self.n_layers < 1:
            raise ValueError("hidden_channels and n_layers must be positive.")
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
        object.__setattr__(self, "n_modes", modes)
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True)
class _ChannelStatistics:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray):
        axes = (0, *range(2, values.ndim))
        mean = np.mean(values, axis=axes, keepdims=True)
        scale = np.std(values, axis=axes, keepdims=True)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        return cls(mean=np.asarray(mean), scale=np.asarray(scale))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.scale

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.mean

    def summary(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class NeuralOperatorOutcome:
    """Trained model, physical predictions, and held-out evidence."""

    model: object
    ledger: TrainingLedger
    predictions: Mapping[str, np.ndarray]
    references: Mapping[str, np.ndarray]
    metrics: Mapping[str, float]
    input_statistics: _ChannelStatistics
    output_statistics: _ChannelStatistics
    model_configuration: Mapping[str, object]
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    validation_dataset: object
    test_case_ids: tuple[str, ...] = ()
    resolution_transfer: bool = False

    def write(self, path: str | Path, *, specification, dataset) -> dict[str, Path]:
        """Write reloadable weights, predictions, and bounded training evidence."""

        import torch

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "operator_state.pt"
        tensor_state = {
            key: value
            for key, value in self.model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        torch.save(
            {
                # NeuralOperator attaches callable-rich metadata to its custom
                # state mapping.  Persist tensors only so reload can keep
                # PyTorch's safe ``weights_only`` boundary.
                "state_dict": tensor_state,
                "model_configuration": dict(self.model_configuration),
                "input_statistics": self.input_statistics.summary(),
                "output_statistics": self.output_statistics.summary(),
                "input_names": tuple(dataset.input_names),
                "parameter_names": tuple(specification.parameter_inputs),
                "output_names": tuple(dataset.output_names),
                "input_shapes": {
                    name: tuple(dataset.fields[name].shape[1:]) for name in dataset.input_names
                },
                "output_shapes": {
                    name: tuple(dataset.fields[name].shape[1:]) for name in dataset.output_names
                },
                "specification": specification.summary(),
                "dataset_fingerprint": dataset.fingerprint,
            },
            state_path,
        )
        prediction_path = output / "held_out_fields.npz"
        arrays: dict[str, np.ndarray] = {}
        for index, name in enumerate(self.predictions):
            arrays[f"prediction_{index}"] = self.predictions[name]
            arrays[f"reference_{index}"] = self.references[name]
        arrays["case_ids"] = np.asarray(self.validation_case_ids)
        np.savez_compressed(prediction_path, **arrays)
        ledger_path = self.ledger.write(output / "training_ledger.json")
        metrics_path = output / "operator_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": dict(self.metrics),
                    "output_names": tuple(self.predictions),
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


class NeuralOperatorPredictor:
    """Reloaded FNO/TFNO model with its physical channel normalization."""

    def __init__(self, model, record: Mapping[str, object], *, device: str) -> None:
        self.model = model
        self.record = dict(record)
        self.device = str(device)
        self.input_names = tuple(record["input_names"])
        self.output_names = tuple(record["output_names"])
        self.input_statistics = _statistics_from_record(record["input_statistics"])
        self.output_statistics = _statistics_from_record(record["output_statistics"])

    def predict(
        self,
        fields: Mapping[str, np.ndarray],
        *,
        parameters: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        import torch

        supplied_fields = set(fields)
        expected_fields = set(self.input_names)
        if supplied_fields != expected_fields:
            raise ValueError(
                "Prediction fields must match the trained input names exactly; "
                f"missing={sorted(expected_fields - supplied_fields)}, "
                f"extra={sorted(supplied_fields - expected_fields)}."
            )
        arrays = [np.asarray(fields[name], dtype=float) for name in self.input_names]
        _validate_field_family(arrays, label="prediction inputs")
        values = np.concatenate(arrays, axis=1)
        parameter_names = tuple(self.record.get("parameter_names", ()))
        if parameter_names:
            if parameters is None:
                raise ValueError(f"This operator requires parameters {parameter_names!r}.")
            supplied_parameters = set(parameters)
            expected_parameters = set(parameter_names)
            if supplied_parameters != expected_parameters:
                raise ValueError(
                    "Prediction parameters must match the trained names exactly; "
                    f"missing={sorted(expected_parameters - supplied_parameters)}, "
                    f"extra={sorted(supplied_parameters - expected_parameters)}."
                )
            values = np.concatenate(
                (
                    values,
                    _parameter_channels(
                        parameters,
                        parameter_names,
                        spatial_shape=values.shape[2:],
                        case_count=values.shape[0],
                    ),
                ),
                axis=1,
            )
        elif parameters:
            raise ValueError(
                "This operator was trained without parameter channels; "
                f"received {sorted(parameters)!r}."
            )
        normalized = self.input_statistics.normalize(values)
        dtype = (
            torch.float64
            if self.record["model_configuration"].get("dtype") == "float64"
            else torch.float32
        )
        tensor = torch.as_tensor(normalized, dtype=dtype, device=self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(tensor).detach().cpu().numpy()
        physical = self.output_statistics.denormalize(prediction)
        return _split_channels(
            physical,
            self.output_names,
            self.record["output_shapes"],
        )


def train_operator(
    specification,
    dataset,
    options: NeuralOperatorTrainingOptions,
    *,
    validation_dataset=None,
    test_dataset=None,
) -> NeuralOperatorOutcome:
    """Train an official NeuralOperator FNO/TFNO on AgentFEM field data."""

    try:
        import neuralop
        import torch
        from neuralop.models import FNO, TFNO
    except ImportError as exc:
        raise ImportError(
            "Neural-operator training requires the optional "
            "`agentfem-learning[neuraloperator]` dependencies."
        ) from exc

    _validate_contract(specification, dataset)
    if validation_dataset is None:
        split = dataset.split(
            validation_fraction=options.validation_fraction,
            seed=options.seed,
        )
        training = split.train
        validation = split.validation
    else:
        training = dataset
        validation = validation_dataset
        _validate_contract(specification, validation)
    if test_dataset is not None:
        _validate_contract(specification, test_dataset, allow_resolution_change=True)
    _require_disjoint_case_ids(training, validation, test_dataset)

    device = _resolve_device(options.device, torch)
    dtype = torch.float64 if options.dtype == "float64" else torch.float32
    _set_seed(options.seed, torch)

    x_train, y_train = _matrices(specification, training)
    x_validation, y_validation = _matrices(specification, validation)
    if len(options.n_modes) != x_train.ndim - 2:
        raise ValueError(
            f"n_modes has dimension {len(options.n_modes)}, but the fields have "
            f"{x_train.ndim - 2} spatial dimensions."
        )
    input_statistics = _ChannelStatistics.fit(x_train)
    output_statistics = _ChannelStatistics.fit(y_train)
    x_train = input_statistics.normalize(x_train)
    y_train = output_statistics.normalize(y_train)
    x_validation = input_statistics.normalize(x_validation)
    y_validation = output_statistics.normalize(y_validation)

    configuration = {
        "architecture": specification.architecture,
        "n_modes": options.n_modes,
        "in_channels": int(x_train.shape[1]),
        "out_channels": int(y_train.shape[1]),
        "hidden_channels": int(options.hidden_channels),
        "n_layers": int(options.n_layers),
        "dtype": options.dtype,
    }
    model_class = TFNO if specification.architecture == "tfno" else FNO
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = model_class(
            n_modes=options.n_modes,
            in_channels=configuration["in_channels"],
            out_channels=configuration["out_channels"],
            hidden_channels=options.hidden_channels,
            n_layers=options.n_layers,
        )
    finally:
        torch.set_default_dtype(default_dtype)
    model = model.to(device=device)

    x_tensor = torch.as_tensor(x_train, dtype=dtype)
    y_tensor = torch.as_tensor(y_train, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(options.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=min(options.batch_size, len(training.case_ids)),
        shuffle=True,
        generator=generator,
    )
    x_validation_tensor = torch.as_tensor(x_validation, dtype=dtype, device=device)
    y_validation_tensor = torch.as_tensor(y_validation, dtype=dtype, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    loss_function = torch.nn.MSELoss()
    ledger = TrainingLedger(
        seed=options.seed,
        device=device,
        dtype=options.dtype,
        framework="neuraloperator",
        framework_version=neuralop.__version__,
    )
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    for epoch in range(1, options.epochs + 1):
        model.train()
        weighted_loss = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device=device)
            y_batch = y_batch.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Neural-operator training became non-finite at epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(x_batch)
            count += len(x_batch)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(x_validation_tensor), y_validation_tensor).detach().cpu()
            )
        training_loss = weighted_loss / max(count, 1)
        improved = ledger.append(
            TrainingEpoch(
                epoch=epoch,
                training_loss=training_loss,
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
                f"operator epoch {epoch}/{options.epochs} | "
                f"train={training_loss:.4e} | validation={validation_loss:.4e}",
                flush=True,
            )
        if epochs_without_improvement >= options.patience:
            ledger.stopped_early = True
            ledger.stop_reason = "validation_patience_exhausted"
            break
    model.load_state_dict(best_state)

    predictions, references, metrics = _evaluate(
        model,
        specification,
        validation,
        input_statistics,
        output_statistics,
        device=device,
        dtype=dtype,
        prefix="validation",
    )
    test_case_ids: tuple[str, ...] = ()
    resolution_transfer = False
    if test_dataset is not None:
        _, _, test_metrics = _evaluate(
            model,
            specification,
            test_dataset,
            input_statistics,
            output_statistics,
            device=device,
            dtype=dtype,
            prefix="test",
        )
        metrics.update(test_metrics)
        test_case_ids = test_dataset.case_ids
        resolution_transfer = any(
            test_dataset.fields[name].shape[2:]
            != training.fields[name].shape[2:]
            for name in (*training.input_names, *training.output_names)
        )
    return NeuralOperatorOutcome(
        model=model,
        ledger=ledger,
        predictions=predictions,
        references=references,
        metrics=metrics,
        input_statistics=input_statistics,
        output_statistics=output_statistics,
        model_configuration=configuration,
        train_case_ids=training.case_ids,
        validation_case_ids=validation.case_ids,
        validation_dataset=validation,
        test_case_ids=test_case_ids,
        resolution_transfer=resolution_transfer,
    )


def load_predictor(path: str | Path, *, device: str = "cpu") -> NeuralOperatorPredictor:
    """Load a provider artifact without requiring the original training process."""

    try:
        import torch
        from neuralop.models import FNO, TFNO
    except ImportError as exc:
        raise ImportError(
            "Loading this model requires `agentfem-learning[neuraloperator]`."
        ) from exc

    selected_device = _resolve_device(device, torch)
    record = torch.load(Path(path), map_location=selected_device, weights_only=True)
    configuration = dict(record["model_configuration"])
    architecture = configuration.pop("architecture")
    dtype_name = configuration.pop("dtype", "float32")
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    model_class = TFNO if architecture == "tfno" else FNO
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = model_class(**configuration)
    finally:
        torch.set_default_dtype(default_dtype)
    model.load_state_dict(record["state_dict"])
    model.to(selected_device)
    model.eval()
    return NeuralOperatorPredictor(model, record, device=selected_device)


def _validate_contract(
    specification, dataset, *, allow_resolution_change: bool = False
) -> None:
    if specification.architecture not in {"fno", "tfno"}:
        raise ValueError(
            "The NeuralOperator provider currently supports architecture='fno' "
            "or architecture='tfno'."
        )
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
            "The first FNO/TFNO provider does not train on masked geometries. "
            "Use a complete structured domain or the coordinate-aware GINO route."
        )
    for name, coordinates in dataset.coordinates.items():
        if not np.allclose(coordinates, coordinates[0:1], rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"FNO coordinate array {name!r} changes between cases; "
                "use a geometry-aware operator rather than padding varying meshes."
            )
    for item in (*specification.inputs, *specification.outputs):
        dataset_encoding = dataset.encoding(item.name)
        specification_encoding = item.summary()
        for attribute in (
            "role",
            "unit",
            "representation",
            "geometry_encoding",
            "mesh_policy",
        ):
            if dataset_encoding.get(attribute) != specification_encoding.get(attribute):
                raise ValueError(
                    f"Dataset field {item.name!r} disagrees with the specification "
                    f"for {attribute!r}."
                )
    missing_parameters = set(specification.parameter_inputs).difference(dataset.parameters)
    if missing_parameters:
        raise ValueError(
            f"Dataset is missing declared operator parameters {sorted(missing_parameters)}."
        )
    arrays = [dataset.fields[name] for name in (*expected_inputs, *expected_outputs)]
    _validate_field_family(arrays, label="operator fields")
    if any(item.shape is None for item in (*specification.inputs, *specification.outputs)):
        raise ValueError(
            "The FNO provider requires explicit channel-first FieldEncoding.shape values."
        )
    spatial_dimension = arrays[0].ndim - 2
    if spatial_dimension != len(specification.inputs[0].shape) - 1:
        raise ValueError("Field encodings must declare channel-first per-case shapes.")
    if not allow_resolution_change:
        for item in (*specification.inputs, *specification.outputs):
            if tuple(dataset.fields[item.name].shape[1:]) != tuple(item.shape or ()):
                raise ValueError(
                    f"Dataset field {item.name!r} does not match its declared shape."
                )


def _validate_field_family(arrays: list[np.ndarray], *, label: str) -> None:
    if not arrays or any(array.ndim < 3 for array in arrays):
        raise ValueError(f"{label} must use (case, channel, spatial...) arrays.")
    sample_counts = {array.shape[0] for array in arrays}
    spatial_shapes = {array.shape[2:] for array in arrays}
    if len(sample_counts) != 1 or len(spatial_shapes) != 1:
        raise ValueError(f"{label} must share case counts and spatial grids.")


def _require_disjoint_case_ids(*collections) -> None:
    owners: dict[str, int] = {}
    overlaps = set()
    for collection_index, collection in enumerate(collections):
        if collection is None:
            continue
        for case_id in collection.case_ids:
            if case_id in owners and owners[case_id] != collection_index:
                overlaps.add(case_id)
            owners[case_id] = collection_index
    if overlaps:
        raise ValueError(
            "Training, validation, and test case IDs must be disjoint; "
            f"overlap={sorted(overlaps)}."
        )


def _matrices(specification, dataset) -> tuple[np.ndarray, np.ndarray]:
    inputs = [
        np.asarray(dataset.fields[item.name], dtype=float) for item in specification.inputs
    ]
    outputs = [
        np.asarray(dataset.fields[item.name], dtype=float) for item in specification.outputs
    ]
    input_values = np.concatenate(inputs, axis=1)
    if specification.parameter_inputs:
        input_values = np.concatenate(
            (
                input_values,
                _parameter_channels(
                    dataset.parameters,
                    specification.parameter_inputs,
                    spatial_shape=input_values.shape[2:],
                    case_count=input_values.shape[0],
                ),
            ),
            axis=1,
        )
    return input_values, np.concatenate(outputs, axis=1)


def _parameter_channels(
    parameters,
    names,
    *,
    spatial_shape,
    case_count: int,
) -> np.ndarray:
    channels = []
    for name in names:
        values = np.asarray(parameters[name], dtype=float)
        if values.shape[0] != case_count:
            raise ValueError(f"Operator parameter {name!r} must have {case_count} cases.")
        flattened = values.reshape((case_count, -1))
        expanded = flattened.reshape(
            (case_count, flattened.shape[1], *(1 for _ in spatial_shape))
        )
        channels.append(
            np.broadcast_to(expanded, (case_count, flattened.shape[1], *spatial_shape))
        )
    return np.concatenate(channels, axis=1)


def _evaluate(
    model,
    specification,
    dataset,
    input_statistics,
    output_statistics,
    *,
    device,
    dtype,
    prefix: str,
):
    import torch

    inputs, outputs = _matrices(specification, dataset)
    normalized = input_statistics.normalize(inputs)
    model.eval()
    with torch.no_grad():
        prediction = (
            model(torch.as_tensor(normalized, dtype=dtype, device=device))
            .detach()
            .cpu()
            .numpy()
        )
    prediction = output_statistics.denormalize(prediction)
    output_names = tuple(item.name for item in specification.outputs)
    output_shapes = {
        item.name: dataset.fields[item.name].shape[1:] for item in specification.outputs
    }
    predictions = _split_channels(prediction, output_names, output_shapes)
    references = _split_channels(outputs, output_names, output_shapes)
    metrics: dict[str, float] = {}
    aggregate_numerator = 0.0
    aggregate_denominator = 0.0
    for name in output_names:
        difference = predictions[name] - references[name]
        numerator = float(np.sum(difference**2))
        denominator = float(np.sum(references[name] ** 2))
        relative = np.sqrt(numerator / max(denominator, np.finfo(float).eps))
        metrics[f"{prefix}_{name}_relative_l2_error"] = float(relative)
        aggregate_numerator += numerator
        aggregate_denominator += denominator
    metrics[f"{prefix}_relative_l2_error"] = float(
        np.sqrt(aggregate_numerator / max(aggregate_denominator, np.finfo(float).eps))
    )
    return predictions, references, metrics


def _split_channels(values, names, shapes) -> dict[str, np.ndarray]:
    selected = np.asarray(values)
    result = {}
    offset = 0
    for name in names:
        channel_count = int(next(iter(shapes[name])))
        result[name] = selected[:, offset : offset + channel_count]
        offset += channel_count
    if offset != selected.shape[1]:
        raise ValueError("Output channel schema does not match model prediction.")
    return result


def _statistics_from_record(record) -> _ChannelStatistics:
    return _ChannelStatistics(
        mean=np.asarray(record["mean"], dtype=float),
        scale=np.asarray(record["scale"], dtype=float),
    )


def _resolve_device(requested: str, torch) -> str:
    selected = str(requested).lower()
    if selected == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if selected == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    if selected not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps.")
    return selected


def _set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "NeuralOperatorOutcome",
    "NeuralOperatorPredictor",
    "NeuralOperatorTrainingOptions",
    "load_predictor",
    "train_operator",
]
