"""AgentFEM Step provider for geometry-informed neural operators."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from agentfem import datasets, learning, results, verification
from agentfem.step_providers import StepOptionContract, StepProvider

from agentfem_learning import __version__

from .checks import OperatorCheck, OperatorCheckContext
from .gino import GINOTrainingOptions, train_gino

_PROVIDER_CHECKS = {"held_out_field_error", "geometry_transfer"}


class GINOStep:
    """Train a geometry-informed operator through the common result lifecycle."""

    def __init__(
        self,
        specification,
        dataset,
        *,
        options,
        validation_dataset=None,
        test_dataset=None,
        check_evaluators=(),
        output=None,
        name="geometry_operator",
    ) -> None:
        self.specification = specification
        self.dataset = dataset
        self.options = options
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.check_evaluators = _validated_checks(check_evaluators)
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
        outcome = train_gino(
            self.specification,
            self.dataset,
            self.options,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset,
        )
        custom_claims = tuple(
            check.evaluate(
                OperatorCheckContext(
                    specification=self.specification,
                    training_dataset=self.dataset,
                    validation_dataset=outcome.validation_dataset,
                    predictions=outcome.predictions,
                    references=outcome.references,
                    metrics=outcome.metrics,
                )
            )
            for check in self.check_evaluators
        )
        implemented = {"held_out_field_error"}
        if outcome.geometry_transfer:
            implemented.add("geometry_transfer")
        implemented.update(check.name for check in self.check_evaluators)
        missing = tuple(
            item for item in self.specification.required_checks if item not in implemented
        )
        result = results.SimulationResult(
            self.name if name is None else str(name),
            metadata={
                "provider": "agentfem-learning.neuraloperator.gino",
                "provider_version": __version__,
                "method": "gino",
                "specification": self.specification.summary(),
                "training_options": asdict(self.options),
                "training": outcome.ledger.summary(),
                "dataset": self.dataset.summary(),
                "geometry": dict(outcome.geometry_configuration),
                "check_evaluators": [check.summary() for check in self.check_evaluators],
                "verification_coverage": {
                    "required": self.specification.required_checks,
                    "implemented": tuple(
                        item
                        for item in self.specification.required_checks
                        if item in implemented
                    ),
                    "missing": missing,
                },
                "capability": {
                    "family": "geometry_informed_function_to_function_learning",
                    "architectures": ("gino",),
                    "maturity": "experimental_provider",
                    "supports_geometry_varying_meshes": True,
                    "supports_variable_point_count": False,
                    "supports_new_topology_claim": False,
                },
            },
        )
        result.add_scientific_inputs(
            specification=self.specification,
            dataset=self.dataset,
            geometry=outcome.geometry_configuration,
            operator_checks=tuple(check.summary() for check in self.check_evaluators),
        )
        result.add_quantities(outcome.metrics, kind="operator_verification")
        result.add_quantity(
            "best_validation_loss", outcome.ledger.best.validation_loss, kind="optimization"
        )
        result.add_quantity("best_epoch", outcome.ledger.best_epoch, kind="optimization")
        for history_name, attribute in (
            ("training_loss", "training_loss"),
            ("validation_loss", "validation_loss"),
        ):
            result.add_history(
                history_name,
                [item.epoch for item in outcome.ledger.records],
                [getattr(item, attribute) for item in outcome.ledger.records],
                abscissa_name="epoch",
                abscissa_unit=None,
                description="Normalized point-field loss; scientific acceptance uses held-out checks.",
            )
        if self.output is not None:
            artifacts = outcome.write(
                self.output, specification=self.specification, dataset=self.dataset
            )
            for artifact_name, artifact_path in artifacts.items():
                result.add_artifact(artifact_name, artifact_path)
            for index, encoding in enumerate(self.specification.outputs):
                result.add_field(
                    encoding.name,
                    artifact=artifacts["held_out_fields"],
                    unit=encoding.unit,
                    location="geometry_output_queries",
                    description="Held-out GINO prediction and reference point field.",
                    processing={
                        "representation": encoding.representation,
                        "prediction_dataset": f"prediction_{index}",
                        "reference_dataset": f"reference_{index}",
                        "coordinate_array": outcome.geometry_configuration["output_queries"],
                        "dataset_fingerprint": self.dataset.fingerprint,
                    },
                )
        claims = [_held_out_claim(outcome.metrics, self.options)]
        if outcome.geometry_transfer:
            claims.append(_geometry_transfer_claim(outcome.metrics, self.options))
        claims.extend(custom_claims)
        claims.extend(_inconclusive_claim(item) for item in missing)
        result.add_verification(
            verification.VerificationReport(
                claims=tuple(claims),
                computed=True,
                converged=outcome.ledger.converged,
                scope="registered geometry-varying point-field operator",
                quality_policy="experimental_geometry_operator",
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
        validity_domain="declared point fields, registered geometries, and parameter domain",
        evidence={"metric": "global relative L2 over held-out physical point fields"},
    )


def _geometry_transfer_claim(metrics, options):
    metric = (
        "geometry_transfer_relative_l2_error"
        if "geometry_transfer_relative_l2_error" in metrics
        else "validation_relative_l2_error"
    )
    return verification.VerificationClaim.compare(
        name="geometry_transfer",
        observable=metric,
        actual=float(metrics[metric]),
        expected=0.0,
        reference="cases whose exact geometry fingerprints are absent from training",
        absolute_tolerance=options.relative_l2_tolerance,
        validity_domain="registered topology deformations; not a new-topology claim",
        evidence={"geometry_identity": "exact input/output coordinate fingerprint"},
    )


def _inconclusive_claim(name):
    return verification.VerificationClaim(
        name=str(name),
        observable=str(name),
        reference="provider-supplied scientific verifier",
        status="inconclusive",
        criterion="an explicit evaluator must be provided for this scientific check",
        kind="verification",
        validity_domain="not inferred from supervised loss",
        message="GINO does not convert optimization loss into a physical claim.",
    )


def _accepts_gino(_model, request) -> bool:
    return (
        isinstance(request.target, learning.NeuralOperatorSpec)
        and request.target.architecture == "gino"
    )


def _lower_gino(model, request):
    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "The GINO provider trains a declared geometry/field mapping and does not "
            f"consume assembled FEM options: {tuple(unsupported)!r}."
        )
    dataset = request.option("dataset")
    if not isinstance(dataset, datasets.ScientificFieldDataset):
        raise TypeError("The GINO provider requires dataset=ScientificFieldDataset(...).")
    options = GINOTrainingOptions(
        input_geometry=str(request.option("input_geometry", "input_geometry")),
        output_queries=request.option("output_queries"),
        coordinate_system=str(
            request.option(
                "coordinate_system", dataset.metadata.get("coordinate_system", "cartesian")
            )
        ),
        coordinate_unit=str(
            request.option("coordinate_unit", dataset.metadata.get("coordinate_unit", "1"))
        ),
        latent_shape=tuple(request.option("latent_shape", (16, 16))),
        n_modes=tuple(request.option("n_modes", (8, 8))),
        hidden_channels=int(request.option("hidden_channels", 32)),
        n_layers=int(request.option("n_layers", 4)),
        input_radius=float(request.option("input_radius", 0.25)),
        output_radius=float(request.option("output_radius", 0.25)),
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
        coordinate_bounds=request.option("coordinate_bounds"),
        neighbor_backend=str(request.option("neighbor_backend", "native")),
    )
    return model.add_step(
        GINOStep(
            request.target,
            dataset,
            options=options,
            validation_dataset=request.option("validation_dataset"),
            test_dataset=request.option("test_dataset"),
            check_evaluators=request.option("check_evaluators", ()),
            output=request.option("output"),
            name=request.option("name") or "geometry_operator",
        )
    )


GINO_PROVIDER = StepProvider(
    name="neuraloperator_geometry_operator",
    analyses=(
        "linear_static",
        "nonlinear_static",
        "first_order_transient",
        "second_order_dynamics",
        "nonlinear_transient",
    ),
    accepts=_accepts_gino,
    lower=_lower_gino,
    priority=510,
    procedure="geometry_informed_neural_operator_training",
    description="Official NeuralOperator GINO provider for registered point-field families.",
    option_contract=StepOptionContract(
        accepted=(
            "K",
            "F",
            "constraints",
            "solver_options",
            "dataset",
            "validation_dataset",
            "test_dataset",
            "check_evaluators",
            "name",
            "output",
            "input_geometry",
            "output_queries",
            "coordinate_system",
            "coordinate_unit",
            "latent_shape",
            "n_modes",
            "hidden_channels",
            "n_layers",
            "input_radius",
            "output_radius",
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
            "coordinate_bounds",
            "neighbor_backend",
        ),
        required=("dataset",),
    ),
)


def _validated_checks(values):
    checks = tuple(values or ())
    if any(not isinstance(item, OperatorCheck) for item in checks):
        raise TypeError("check_evaluators must contain OperatorCheck records.")
    names = tuple(item.name for item in checks)
    if len(set(names)) != len(names):
        raise ValueError("Operator check evaluator names must be unique.")
    reserved = set(names).intersection(_PROVIDER_CHECKS)
    if reserved:
        raise ValueError(
            f"Custom checks may not replace provider-owned checks {tuple(sorted(reserved))}."
        )
    return checks


__all__ = ["GINO_PROVIDER", "GINOStep"]
