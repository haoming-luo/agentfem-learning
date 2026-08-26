"""AgentFEM Step provider for the first XDEM reference workflow."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from agentfem import learning, results, verification
from agentfem.step_providers import StepOptionContract, StepProvider

from .reference import ReferenceTrainingOptions, train_mode_iii_reference


class XDEMReferenceStep:
    """Executable neural-field Step returning an ordinary SimulationResult."""

    def __init__(
        self,
        spec: learning.NeuralFieldSpec,
        *,
        options: ReferenceTrainingOptions,
        output=None,
        name: str = "xdem_mode_iii_tip",
    ) -> None:
        self.spec = spec
        self.options = options
        # Resolve once at the provider boundary. SimulationResult can then
        # store artifact paths relative to its own manifest, independent of
        # the process working directory used later by a human, agent, or GUI.
        self.output = None if output is None else Path(output).expanduser().resolve()
        self.name = str(name)
        self.step_number = 0
        self.execution_context = None
        self.last_result = None

    def solve(self):
        """Execute once and return the common scientific result."""

        return self.solve_result()

    def solve_result(self, *, name: str | None = None):
        """Train the neural field and attach reference-based evidence."""

        if self.last_result is not None:
            return self.last_result
        outcome = train_mode_iii_reference(self.spec, self.options)
        integration_evidence = learning.integration_consistency_check(
            self.spec.integration,
            training_value=outcome.metrics["training_integral_energy"],
            validation_value=outcome.metrics["predicted_energy"],
            refinement_values=(outcome.metrics["refined_integral_energy"],),
            relative_tolerance=0.05,
        )
        selected_name = self.name if name is None else str(name)
        result = results.SimulationResult(
            selected_name,
            metadata={
                "provider": "agentfem-learning.xdem",
                "provider_version": "0.1.0a1",
                "method": "extended_deep_energy_reference",
                "specification": self.spec.summary(),
                "training": {
                    **asdict(self.options),
                    "torch_version": torch.__version__,
                    "resolved_device": outcome.device,
                },
                "integration_evidence": integration_evidence.summary(),
                "capability": {
                    "problem": "williams_mode_iii_tip",
                    "maturity": "experimental_reference",
                    "supports_crack_growth": False,
                    "supports_general_xdem": False,
                },
            },
        )
        result.add_quantities(
            outcome.metrics,
            units={
                "predicted_energy": "J/m",
                "reference_energy": "J/m",
                "learned_tip_amplitude": "m",
            },
            kind="neural_field_verification",
        )
        result.add_quantity(
            "final_training_loss",
            outcome.losses[-1],
            kind="optimization",
        )
        result.add_quantity(
            "training_validation_integration_gap",
            integration_evidence.training_validation_gap,
            kind="integration_verification",
        )
        result.add_quantity(
            "validation_refinement_integration_gap",
            integration_evidence.refinement_gap,
            kind="integration_verification",
        )
        result.add_quantity(
            "training_loss_reduction",
            outcome.losses[-1] / outcome.losses[0],
            kind="optimization",
        )
        result.add_history(
            "training_loss",
            outcome.epochs,
            outcome.losses,
            abscissa_name="optimizer_step",
            abscissa_unit=None,
            description="Normalized energy and boundary-condition objective.",
        )

        if self.output is not None:
            output = self.output
            output.mkdir(parents=True, exist_ok=True)
            field_path = outcome.write_field(output / "mode_iii_field.npz")
            weights_path = output / "model_state.pt"
            torch.save(
                {
                    "state_dict": outcome.model.state_dict(),
                    "specification": self.spec.summary(),
                    "training_options": asdict(self.options),
                    "torch_version": torch.__version__,
                },
                weights_path,
            )
            result.add_artifact("neural_field", field_path)
            result.add_artifact("model_state", weights_path)
            integration_path = output / "integration_evidence.json"
            integration_path.write_text(
                json.dumps(integration_evidence.summary(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            result.add_artifact("integration_evidence", integration_path)
            crack_geometry_path = output / "crack_geometry.json"
            crack_geometry_path.write_text(
                json.dumps(
                    self.spec.metadata["geometry"]["cracks"],
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            result.add_artifact("crack_geometry", crack_geometry_path)
            result.add_field(
                "W",
                artifact=field_path,
                unit="m",
                location="mesh_independent_coordinates",
                description="Predicted and reference anti-plane displacement samples.",
                processing={
                    "representation": "neural_field_samples",
                    "provider": "agentfem-learning.xdem",
                    "coordinates_dataset": "coordinates",
                    "prediction_dataset": "prediction",
                    "reference_dataset": "reference",
                    "discontinuity_representation": "paired_one_sided_samples",
                    "crack_trace_coordinates_dataset": "crack_trace_coordinates",
                    "crack_trace_side_dataset": "crack_trace_side",
                    "crack_trace_prediction_dataset": "crack_trace_prediction",
                    "crack_geometry_artifact": "crack_geometry",
                },
            )

        claims = (
            _upper_bound_claim(
                "mode_iii_field_error",
                "relative_l2_error",
                outcome.metrics["relative_l2_error"],
                0.08,
            ),
            _upper_bound_claim(
                "williams_boundary_error",
                "relative_boundary_error",
                outcome.metrics["relative_boundary_error"],
                0.05,
            ),
            _upper_bound_claim(
                "mode_iii_energy_error",
                "relative_energy_error",
                outcome.metrics["relative_energy_error"],
                0.10,
            ),
            _upper_bound_claim(
                "crack_jump_error",
                "crack_jump_relative_error",
                outcome.metrics["crack_jump_relative_error"],
                0.10,
            ),
            _upper_bound_claim(
                "crack_face_traction_error",
                "crack_face_traction_relative_error",
                outcome.metrics["crack_face_traction_relative_error"],
                0.15,
            ),
            _upper_bound_claim(
                "independent_integration_consistency",
                "maximum_independent_integration_gap",
                max(
                    integration_evidence.training_validation_gap,
                    integration_evidence.refinement_gap or 0.0,
                ),
                integration_evidence.relative_tolerance,
            ),
        )
        finite = bool(np.all(np.isfinite(outcome.losses)))
        reduced = bool(outcome.losses[-1] < outcome.losses[0])
        result.add_verification(
            verification.VerificationReport(
                claims=claims,
                computed=True,
                converged=finite and reduced,
                scope="normalized Williams Mode-III slit-annulus reference",
                quality_policy="experimental_reference",
            )
        )
        if self.output is not None:
            result.write_manifest(
                self.output / "result.json",
                include_histories=True,
            )
        self.last_result = result
        return result


def _upper_bound_claim(name, observable, actual, tolerance):
    return verification.VerificationClaim.compare(
        name=name,
        observable=observable,
        actual=float(actual),
        expected=0.0,
        reference="analytical leading Williams Mode-III field on a slit annulus",
        absolute_tolerance=float(tolerance),
        validity_domain=(
            "normalized linear anti-plane elasticity; prescribed Williams data on "
            "inner and outer circles; traction-free branch-cut faces"
        ),
        evidence={
            "provider": "agentfem-learning.xdem",
            "problem": "williams_mode_iii_tip",
        },
    )


def _accepts_xdem_reference(_model, request) -> bool:
    target = request.target
    return (
        isinstance(target, learning.NeuralFieldSpec)
        and target.metadata.get("provider") == "agentfem-learning.xdem"
        and target.metadata.get("problem") == "williams_mode_iii_tip"
    )


def _lower_xdem_reference(model, request):
    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "The XDEM neural-field provider consumes the declared objective and "
            f"conditions, not assembled FEM options: {tuple(unsupported)!r}."
        )
    options = ReferenceTrainingOptions(
        adam_epochs=int(request.option("epochs", 500)),
        lbfgs_steps=int(request.option("lbfgs_steps", 6)),
        learning_rate=float(request.option("learning_rate", 5.0e-3)),
        boundary_penalty=float(request.option("boundary_penalty", 400.0)),
        hidden_layers=tuple(request.option("hidden_layers", (24, 24))),
        seed=int(request.option("seed", 2026)),
        device=str(request.option("device", "cpu")),
        dtype=str(request.option("dtype", "float64")),
        progress=bool(request.option("progress", False)),
    )
    step = XDEMReferenceStep(
        request.target,
        options=options,
        output=request.option("output"),
        name=request.option("name") or "xdem_mode_iii_tip",
    )
    return model.add_step(step)


XDEM_REFERENCE_PROVIDER = StepProvider(
    name="xdem_reference_neural_field",
    analyses=("linear_static", "nonlinear_static"),
    accepts=_accepts_xdem_reference,
    lower=_lower_xdem_reference,
    priority=500,
    description=(
        "Experimental PyTorch deep-energy provider for the normalized Williams "
        "Mode-III crack-tip reference problem."
    ),
    procedure="xdem_reference",
    option_contract=StepOptionContract(
        accepted=(
            "K",
            "F",
            "constraints",
            "solver_options",
            "name",
            "output",
            "epochs",
            "lbfgs_steps",
            "learning_rate",
            "boundary_penalty",
            "hidden_layers",
            "seed",
            "device",
            "dtype",
            "progress",
        )
    ),
)


__all__ = ["XDEM_REFERENCE_PROVIDER", "XDEMReferenceStep"]
