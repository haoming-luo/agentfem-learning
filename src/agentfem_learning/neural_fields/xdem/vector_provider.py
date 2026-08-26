"""AgentFEM Step provider for vector-elastic Williams neural fields."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from agentfem import learning, results, verification
from agentfem.step_providers import StepOptionContract, StepProvider

from .provider import _upper_bound_claim
from .reference import ReferenceTrainingOptions
from .vector_reference import train_vector_reference


class XDEMVectorStep:
    """Executable vector-elastic neural-field Step."""

    def __init__(
        self,
        spec: learning.NeuralFieldSpec,
        *,
        options: ReferenceTrainingOptions,
        output=None,
        name: str = "xdem_vector_tip",
    ) -> None:
        self.spec = spec
        self.options = options
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
        outcome = train_vector_reference(self.spec, self.options)
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
                "method": "extended_deep_energy_vector_reference",
                "specification": self.spec.summary(),
                "training": {
                    **asdict(self.options),
                    "torch_version": torch.__version__,
                    "resolved_device": outcome.device,
                },
                "integration_evidence": integration_evidence.summary(),
                "stress_intensity": outcome.stress_intensity.summary(),
                "capability": {
                    "problem": "williams_vector_tip",
                    "maturity": "experimental_reference",
                    "assumptions": ("plane_stress", "plane_strain"),
                    "modes": ("I", "II", "mixed"),
                    "supports_crack_growth": False,
                    "supports_multiple_cracks": False,
                },
            },
        )
        result.add_quantities(
            outcome.metrics,
            units={
                "predicted_energy": "J/m",
                "reference_energy": "J/m",
                "learned_K_I": "Pa*sqrt(m)",
                "learned_K_II": "Pa*sqrt(m)",
            },
            kind="neural_field_verification",
        )
        result.add_quantity(
            "final_training_loss", outcome.losses[-1], kind="optimization"
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
        result.add_history(
            "training_loss",
            outcome.epochs,
            outcome.losses,
            abscissa_name="optimizer_step",
            abscissa_unit=None,
            description="Vector elastic energy and boundary-condition objective.",
        )

        if self.output is not None:
            output = self.output
            output.mkdir(parents=True, exist_ok=True)
            field_path = outcome.write_field(output / "vector_tip_field.npz")
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
            sif_path = output / "stress_intensity.json"
            sif_path.write_text(
                json.dumps(
                    outcome.stress_intensity.summary(), indent=2, sort_keys=True
                ),
                encoding="utf-8",
            )
            geometry_path = output / "crack_geometry.json"
            geometry_path.write_text(
                json.dumps(
                    self.spec.metadata["geometry"]["cracks"],
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            result.add_artifact("neural_field", field_path)
            result.add_artifact("model_state", weights_path)
            result.add_artifact("stress_intensity", sif_path)
            result.add_artifact("crack_geometry", geometry_path)
            result.add_field(
                "U",
                artifact=field_path,
                unit="m",
                location="mesh_independent_coordinates",
                description="Predicted two-dimensional displacement samples.",
                processing={
                    "representation": "neural_field_samples",
                    "coordinates_dataset": "coordinates",
                    "value_dataset": "displacement",
                    "reference_dataset": "reference_displacement",
                    "discontinuity_representation": "paired_one_sided_samples",
                    "crack_trace_coordinates_dataset": "crack_trace_coordinates",
                    "crack_trace_side_dataset": "crack_trace_side",
                    "crack_trace_value_dataset": "crack_trace_displacement",
                },
            )
            result.add_field(
                "S",
                artifact=field_path,
                unit="Pa",
                location="mesh_independent_coordinates",
                description="Autograd Cauchy stress samples in the reference basis.",
                processing={
                    "representation": "neural_field_samples",
                    "coordinates_dataset": "coordinates",
                    "value_dataset": "stress",
                },
            )

        claims = (
            _upper_bound_claim(
                "vector_field_error",
                "relative_l2_error",
                outcome.metrics["relative_l2_error"],
                0.03,
            ),
            _upper_bound_claim(
                "vector_boundary_error",
                "relative_boundary_error",
                outcome.metrics["relative_boundary_error"],
                0.03,
            ),
            _upper_bound_claim(
                "vector_energy_error",
                "relative_energy_error",
                outcome.metrics["relative_energy_error"],
                0.03,
            ),
            _upper_bound_claim(
                "stress_intensity_error",
                "stress_intensity_relative_error",
                outcome.metrics["stress_intensity_relative_error"],
                0.03,
            ),
            _upper_bound_claim(
                "stress_intensity_path_variation",
                "stress_intensity_path_variation",
                outcome.metrics["stress_intensity_path_variation"],
                0.03,
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
                scope="stationary mixed-mode Williams vector field on a slit annulus",
                quality_policy="experimental_reference",
            )
        )
        if self.output is not None:
            result.write_manifest(self.output / "result.json", include_histories=True)
        self.last_result = result
        return result


def _accepts_vector_reference(_model, request) -> bool:
    target = request.target
    return (
        isinstance(target, learning.NeuralFieldSpec)
        and target.metadata.get("provider") == "agentfem-learning.xdem"
        and target.metadata.get("problem") == "williams_vector_tip"
    )


def _lower_vector_reference(model, request):
    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "The XDEM vector provider consumes its declared scientific "
            f"conditions, not assembled FEM options: {tuple(unsupported)!r}."
        )
    declared_assumption = request.target.metadata["material"]["assumption"]
    study = model.study
    if int(getattr(study, "dimension", 0)) != 2:
        raise ValueError("The XDEM vector reference requires a two-dimensional Study.")
    if getattr(study, "assumption", None) != declared_assumption:
        raise ValueError(
            "Study and NeuralFieldSpec must use the same plane-stress or "
            f"plane-strain assumption; study={study.assumption!r}, "
            f"spec={declared_assumption!r}."
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
    step = XDEMVectorStep(
        request.target,
        options=options,
        output=request.option("output"),
        name=request.option("name") or "xdem_vector_tip",
    )
    return model.add_step(step)


XDEM_VECTOR_PROVIDER = StepProvider(
    name="xdem_vector_lefm_neural_field",
    analyses=("linear_static", "nonlinear_static"),
    accepts=_accepts_vector_reference,
    lower=_lower_vector_reference,
    priority=510,
    description=(
        "Experimental PyTorch deep-energy provider for stationary 2D "
        "mixed-mode Williams vector fields."
    ),
    procedure="xdem_vector_reference",
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


__all__ = ["XDEM_VECTOR_PROVIDER", "XDEMVectorStep"]
