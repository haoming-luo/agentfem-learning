"""AgentFEM Step provider backed by the official NeuralOperator package."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from agentfem import datasets, learning, results, verification
from agentfem.step_providers import StepOptionContract, StepProvider

from agentfem_learning import __version__

from .api import NeuralOperatorTrainingOptions, train_operator

_IMPLEMENTED_CHECKS = {"held_out_field_error", "resolution_transfer"}


class NeuralOperatorStep:
    """Train one field operator and return the common AgentFEM result."""

    def __init__(
        self,
        specification: learning.NeuralOperatorSpec,
        dataset: datasets.ScientificFieldDataset,
        *,
        options: NeuralOperatorTrainingOptions,
        validation_dataset=None,
        test_dataset=None,
        output=None,
        name: str = "neural_operator",
    ) -> None:
        self.specification = specification
        self.dataset = dataset
        self.options = options
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.output = None if output is None else Path(output).expanduser().resolve()
        self.name = str(name)
        self.step_number = 0
        self.execution_context = None
        self.last_result = None

    def solve(self):
        return self.solve_result()

    def solve_result(self, *, name: str | None = None):
        if self.last_result is not None:
            return self.last_result
        outcome = train_operator(
            self.specification,
            self.dataset,
            self.options,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset,
        )
        missing_checks = tuple(
            item
            for item in self.specification.required_checks
            if item not in _IMPLEMENTED_CHECKS
            or (item == "resolution_transfer" and self.test_dataset is None)
        )
        result = results.SimulationResult(
            self.name if name is None else str(name),
            metadata={
                "provider": "agentfem-learning.neuraloperator",
                "provider_version": __version__,
                "method": self.specification.architecture,
                "specification": self.specification.summary(),
                "training_options": asdict(self.options),
                "training": outcome.ledger.summary(),
                "dataset": self.dataset.summary(),
                "verification_coverage": {
                    "required": self.specification.required_checks,
                    "implemented": tuple(
                        item
                        for item in self.specification.required_checks
                        if item in _IMPLEMENTED_CHECKS
                        and not (item == "resolution_transfer" and self.test_dataset is None)
                    ),
                    "missing": missing_checks,
                },
                "capability": {
                    "family": "function_to_function_learning",
                    "architectures": ("fno", "tfno"),
                    "maturity": "experimental_provider",
                    "supports_structured_grids": True,
                    "supports_geometry_varying_meshes": False,
                },
            },
        )
        result.add_scientific_inputs(
            specification=self.specification,
            dataset=self.dataset,
        )
        result.add_quantities(outcome.metrics, kind="operator_verification")
        result.add_quantity(
            "best_validation_loss",
            outcome.ledger.best.validation_loss,
            kind="optimization",
        )
        result.add_quantity(
            "best_epoch",
            outcome.ledger.best_epoch,
            kind="optimization",
        )
        result.add_history(
            "training_loss",
            [item.epoch for item in outcome.ledger.records],
            [item.training_loss for item in outcome.ledger.records],
            abscissa_name="epoch",
            abscissa_unit=None,
            description="Mean squared loss in normalized output channels.",
        )
        result.add_history(
            "validation_loss",
            [item.epoch for item in outcome.ledger.records],
            [item.validation_loss for item in outcome.ledger.records],
            abscissa_name="epoch",
            abscissa_unit=None,
            description="Independent held-out loss used for model selection.",
        )

        if self.output is not None:
            artifacts = outcome.write(
                self.output,
                specification=self.specification,
                dataset=self.dataset,
            )
            for artifact_name, artifact_path in artifacts.items():
                result.add_artifact(artifact_name, artifact_path)
            for index, encoding in enumerate(self.specification.outputs):
                result.add_field(
                    encoding.name,
                    artifact=artifacts["held_out_fields"],
                    unit=encoding.unit,
                    location="mesh_independent_coordinates",
                    description="Held-out neural-operator prediction and reference field.",
                    processing={
                        "representation": encoding.representation,
                        "prediction_dataset": f"prediction_{index}",
                        "reference_dataset": f"reference_{index}",
                        "dataset_fingerprint": self.dataset.fingerprint,
                    },
                )

        claims = [_held_out_claim(outcome.metrics, self.options)]
        if self.test_dataset is not None:
            claims.append(_resolution_claim(outcome.metrics, self.options))
        claims.extend(_inconclusive_claim(name) for name in missing_checks)
        result.add_verification(
            verification.VerificationReport(
                claims=tuple(claims),
                computed=True,
                converged=outcome.ledger.converged,
                scope="held-out supervised field-operator mapping",
                quality_policy="experimental_operator",
            )
        )
        if self.output is not None:
            result.write_manifest(self.output / "result.json", include_histories=True)
        self.last_result = result
        return result


def _held_out_claim(metrics, options):
    return verification.VerificationClaim.compare(
        name="held_out_field_error",
        observable="validation_relative_l2_error",
        actual=float(metrics["validation_relative_l2_error"]),
        expected=0.0,
        reference="independent cases excluded from optimizer updates",
        absolute_tolerance=options.relative_l2_tolerance,
        validity_domain="declared field encodings and sampled parameter domain",
        evidence={"metric": "global relative L2 over held-out physical fields"},
    )


def _resolution_claim(metrics, options):
    return verification.VerificationClaim.compare(
        name="resolution_transfer",
        observable="test_relative_l2_error",
        actual=float(metrics["test_relative_l2_error"]),
        expected=0.0,
        reference="independent test dataset on a declared target resolution",
        absolute_tolerance=options.relative_l2_tolerance,
        validity_domain="same channel semantics and physical domain",
        evidence={"metric": "global relative L2 on resolution-transfer cases"},
    )


def _inconclusive_claim(name: str):
    return verification.VerificationClaim(
        name=str(name),
        observable=str(name),
        reference="provider-supplied scientific verifier",
        status="inconclusive",
        criterion="an explicit evaluator must be provided for this scientific check",
        kind="verification",
        validity_domain="not evaluated by the generic supervised operator trainer",
        message="The generic provider does not infer this physical check from loss alone.",
    )


def _accepts_neuraloperator(_model, request) -> bool:
    target = request.target
    return isinstance(target, learning.NeuralOperatorSpec) and target.architecture in {
        "fno",
        "tfno",
    }


def _lower_neuraloperator(model, request):
    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "The neural-operator provider trains a declared field mapping and "
            f"does not consume assembled FEM options: {tuple(unsupported)!r}."
        )
    dataset = request.option("dataset")
    if not isinstance(dataset, datasets.ScientificFieldDataset):
        raise TypeError(
            "The NeuralOperator provider requires dataset=ScientificFieldDataset(...)."
        )
    options = NeuralOperatorTrainingOptions(
        n_modes=tuple(request.option("n_modes", (8, 8))),
        hidden_channels=int(request.option("hidden_channels", 32)),
        n_layers=int(request.option("n_layers", 4)),
        epochs=int(request.option("epochs", 100)),
        batch_size=int(request.option("batch_size", 8)),
        learning_rate=float(request.option("learning_rate", 1.0e-3)),
        weight_decay=float(request.option("weight_decay", 1.0e-6)),
        validation_fraction=float(request.option("validation_fraction", 0.2)),
        seed=int(request.option("seed", 2026)),
        device=str(request.option("device", "auto")),
        dtype=str(request.option("dtype", "float32")),
        patience=int(request.option("patience", 20)),
        minimum_delta=float(request.option("minimum_delta", 1.0e-6)),
        relative_l2_tolerance=float(request.option("relative_l2_tolerance", 0.10)),
        progress=bool(request.option("progress", False)),
    )
    return model.add_step(
        NeuralOperatorStep(
            request.target,
            dataset,
            options=options,
            validation_dataset=request.option("validation_dataset"),
            test_dataset=request.option("test_dataset"),
            output=request.option("output"),
            name=request.option("name") or "neural_operator",
        )
    )


NEURALOPERATOR_PROVIDER = StepProvider(
    name="neuraloperator_field_operator",
    analyses=(
        "linear_static",
        "nonlinear_static",
        "first_order_transient",
        "second_order_dynamics",
        "nonlinear_transient",
    ),
    accepts=_accepts_neuraloperator,
    lower=_lower_neuraloperator,
    priority=500,
    procedure="neural_operator_training",
    description="Official NeuralOperator FNO/TFNO provider for structured field datasets.",
    option_contract=StepOptionContract(
        accepted=(
            "K",
            "F",
            "constraints",
            "solver_options",
            "dataset",
            "validation_dataset",
            "test_dataset",
            "name",
            "output",
            "n_modes",
            "hidden_channels",
            "n_layers",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "validation_fraction",
            "seed",
            "device",
            "dtype",
            "patience",
            "minimum_delta",
            "relative_l2_tolerance",
            "progress",
        ),
        required=("dataset",),
    ),
)


__all__ = ["NEURALOPERATOR_PROVIDER", "NeuralOperatorStep"]
