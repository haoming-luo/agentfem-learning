"""AgentFEM Step provider for finite-domain, predefined-crack XDEM-D."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from agentfem import learning, results, verification
from agentfem.step_providers import StepOptionContract, StepProvider

from .finite_domain_solver import problem_from_spec, train_finite_domain
from .reference import ReferenceTrainingOptions


def _upper_bound_claim(name, observable, actual, tolerance, *, evidence):
    return verification.VerificationClaim.compare(
        name=name,
        observable=observable,
        actual=float(actual),
        expected=0.0,
        reference="declared finite-domain XDEM-D scientific contract",
        absolute_tolerance=float(tolerance),
        validity_domain=(
            "stationary 2D homogeneous isotropic linear elasticity; rectangular "
            "domain; mutually separated internal straight cracks; constant "
            "displacement and traction boundary conditions"
        ),
        evidence=dict(evidence),
    )


class XDEMFiniteDomainStep:
    """Executable finite-domain neural-field Step."""

    def __init__(self, spec, *, options, output=None, name="xdem_finite_domain"):
        self.spec = spec
        self.options = options
        self.output = None if output is None else Path(output).expanduser().resolve()
        self.name = str(name)
        self.step_number = 0
        self.execution_context = None
        self.last_result = None

    def solve(self):
        return self.solve_result()

    def solve_result(self, *, name=None):
        if self.last_result is not None:
            return self.last_result
        outcome = train_finite_domain(self.spec, self.options)
        problem = problem_from_spec(self.spec)
        selected_name = self.name if name is None else str(name)
        result = results.SimulationResult(
            selected_name,
            metadata={
                "provider": "agentfem-learning.xdem",
                "provider_version": "0.1.0a1",
                "method": "finite_domain_xdem_d",
                "specification": self.spec.summary(),
                "training": {
                    **asdict(self.options),
                    "torch_version": torch.__version__,
                    "resolved_device": outcome.device,
                },
                "stress_intensity": outcome.stress_intensity.summary(),
                "capability": {
                    "problem": "finite_domain_static_xdem_d",
                    "maturity": "experimental_solver",
                    "assumptions": ("plane_stress", "plane_strain"),
                    "modes": ("I", "II", "mixed"),
                    "supports_multiple_cracks": True,
                    "supports_crack_growth": False,
                    "supported_geometry": "separated_internal_straight_cracks",
                },
            },
        )
        result.add_quantities(
            outcome.metrics,
            units={
                "training_internal_energy": "J/m",
                "validation_internal_energy": "J/m",
                "refined_internal_energy": "J/m",
                "validation_external_work": "J/m",
            },
            kind="neural_field_verification",
        )
        result.add_quantity("final_training_loss", outcome.losses[-1], kind="optimization")
        result.add_history(
            "training_loss",
            outcome.epochs,
            outcome.losses,
            abscissa_name="optimizer_step",
            description="Finite-domain total-potential-energy objective.",
        )

        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            field_path = outcome.write_field(self.output / "finite_domain_field.npz")
            weight_path = self.output / "model_state.pt"
            torch.save(
                {
                    "state_dict": outcome.model.state_dict(),
                    "specification": self.spec.summary(),
                    "training_options": asdict(self.options),
                    "torch_version": torch.__version__,
                },
                weight_path,
            )
            sif_path = self.output / "stress_intensity.json"
            sif_path.write_text(
                json.dumps(outcome.stress_intensity.summary(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            geometry_path = self.output / "crack_geometry.json"
            geometry_path.write_text(
                json.dumps(problem.cracks.summary(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            result.add_artifact("neural_field", field_path)
            result.add_artifact("model_state", weight_path)
            result.add_artifact("stress_intensity", sif_path)
            result.add_artifact("crack_geometry", geometry_path)
            result.add_field(
                "U",
                artifact=field_path,
                unit="m",
                location="mesh_independent_coordinates",
                description="Finite-domain XDEM-D displacement samples.",
                processing={
                    "representation": "neural_field_samples",
                    "coordinates_dataset": "coordinates",
                    "value_dataset": "displacement",
                    "discontinuity_representation": "paired_one_sided_samples",
                    "crack_trace_coordinates_dataset": "crack_trace_coordinates",
                    "crack_trace_side_dataset": "crack_trace_side",
                    "crack_trace_id_dataset": "crack_trace_id",
                    "crack_trace_value_dataset": "crack_trace_displacement",
                },
            )
            result.add_field(
                "S",
                artifact=field_path,
                unit="Pa",
                location="mesh_independent_coordinates",
                description="Autograd Cauchy stress samples.",
                processing={
                    "representation": "neural_field_samples",
                    "coordinates_dataset": "coordinates",
                    "value_dataset": "stress",
                },
            )

        evidence = {
            "provider": "agentfem-learning.xdem",
            "problem": "finite_domain_static_xdem_d",
            "scientific_fingerprint": self.spec.metadata["scientific_fingerprint"],
        }
        claims = (
            _upper_bound_claim(
                "finite_domain_boundary_error",
                "relative_boundary_error",
                outcome.metrics["relative_boundary_error"],
                0.05,
                evidence=evidence,
            ),
            _upper_bound_claim(
                "finite_domain_integration_consistency",
                "validation_refinement_energy_gap",
                outcome.metrics["validation_refinement_energy_gap"],
                0.05,
                evidence=evidence,
            ),
            _upper_bound_claim(
                "per_tip_path_variation",
                "maximum_stress_intensity_path_variation",
                outcome.metrics["maximum_stress_intensity_path_variation"],
                0.08,
                evidence=evidence,
            ),
        )
        finite = bool(np.all(np.isfinite(outcome.losses)))
        reduced = bool(len(outcome.losses) and outcome.losses[-1] < outcome.losses[0])
        result.add_verification(
            verification.VerificationReport(
                claims=claims,
                computed=True,
                converged=finite and reduced,
                scope=(
                    "finite-domain XDEM-D numerical consistency; external "
                    "single- and multi-crack benchmark validation remains separate"
                ),
                quality_policy="experimental_solver",
            )
        )
        if self.output is not None:
            result.write_manifest(self.output / "result.json", include_histories=True)
        self.last_result = result
        return result


def _accepts_finite_domain(_model, request):
    target = request.target
    return (
        isinstance(target, learning.NeuralFieldSpec)
        and target.metadata.get("provider") == "agentfem-learning.xdem"
        and target.metadata.get("problem") == "finite_domain_static_xdem_d"
    )


def _lower_finite_domain(model, request):
    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "The finite-domain XDEM provider consumes its declared material, "
            f"cracks, and boundary conditions, not FEM assembly options: {tuple(unsupported)!r}."
        )
    problem = problem_from_spec(request.target)
    study = model.study
    if int(getattr(study, "dimension", 0)) != 2:
        raise ValueError("Finite-domain XDEM-D requires a two-dimensional Study.")
    material_assumption = problem.material.summary()["assumption"]
    if getattr(study, "assumption", None) != material_assumption:
        raise ValueError(
            "Study and XDEM problem must use the same plane-stress or plane-strain "
            f"assumption; study={study.assumption!r}, problem={material_assumption!r}."
        )
    options = ReferenceTrainingOptions(
        adam_epochs=int(request.option("epochs", 1500)),
        lbfgs_steps=int(request.option("lbfgs_steps", 12)),
        learning_rate=float(request.option("learning_rate", 2.0e-3)),
        boundary_penalty=float(request.option("boundary_penalty", 500.0)),
        hidden_layers=tuple(request.option("hidden_layers", (48, 48, 48))),
        seed=int(request.option("seed", 2026)),
        device=str(request.option("device", "cpu")),
        dtype=str(request.option("dtype", "float64")),
        progress=bool(request.option("progress", False)),
    )
    return model.add_step(
        XDEMFiniteDomainStep(
            request.target,
            options=options,
            output=request.option("output"),
            name=request.option("name") or problem.name,
        )
    )


XDEM_FINITE_DOMAIN_PROVIDER = StepProvider(
    name="xdem_finite_domain_lefm_neural_field",
    analyses=("linear_static", "nonlinear_static"),
    accepts=_accepts_finite_domain,
    lower=_lower_finite_domain,
    priority=515,
    description=(
        "Experimental finite-domain XDEM-D provider for separated internal "
        "straight cracks and per-tip mixed-mode evidence."
    ),
    procedure="xdem_finite_domain",
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


__all__ = [
    "XDEM_FINITE_DOMAIN_PROVIDER",
    "XDEMFiniteDomainStep",
]
