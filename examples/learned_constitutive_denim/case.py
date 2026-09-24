"""Run the fixed DENIM material-point acceptance path from a local bundle."""

from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path

import numpy as np
from agentfem import constitutive, learning

from agentfem_learning.learned_constitutive.denim import state_schema
from agentfem_learning.learned_constitutive.denim.loader import load_denim_v1
from agentfem_learning.learned_constitutive.provider import TorchConstitutiveProvider
from agentfem_learning.learned_constitutive.registry import register_architecture

MODEL_REVISION = "5629df0a23a3d1ed43e9de2150e3d33cb979fdc1"
DATASET_REVISION = "c84f416e5a71daa157e406c50afc3fc73509b9ca"


def specification(bundle: Path):
    return learning.learned_constitutive(
        provider="agentfem-learning.torch-constitutive",
        architecture_id="denim.v1",
        artifact=str(bundle),
        revision=MODEL_REVISION,
        model_name="denim-expanded",
        model_version="1.1.0",
        tangent_convention=constitutive.MaterialTangentConvention.cauchy_small_strain(
            symmetric=False
        ),
        parameter_schema=(
            learning.MaterialParameterSpec("young", "Pa", minimum=0.0),
            learning.MaterialParameterSpec("poisson", "1", minimum=-1.0, maximum=0.5),
            learning.MaterialParameterSpec("yield_stress", "Pa", minimum=0.0),
        ),
        parameters={"young": 190.0e9, "poisson": 0.3, "yield_stress": 280.0e6},
        state_schema=state_schema(2),
        required_inputs=("strain_new", "state_old", "parameters"),
        capabilities=("stress", "state", "batch", "energy", "consistent_tangent"),
        applicability_domain={
            "policy": "report_without_silent_fallback",
            "maximum_absolute_strain": 0.03,
            "maximum_peeq": 0.03,
        },
        dataset_id="HaomingLuo/AgentFEM-Material-Loading-Memory",
        dataset_revision=DATASET_REVISION,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("denim-material-point.json"))
    args = parser.parse_args()
    register_architecture("denim.v1", load_denim_v1, replace=True)
    provider = TorchConstitutiveProvider()
    spec = specification(args.bundle.resolve())
    material = provider.create(spec)
    coordinate = np.linspace(0.0, 1.0, 121)
    amplitude = np.where(coordinate <= 0.5, 2.0 * coordinate, 2.0 * (1.0 - coordinate))
    scalar = 0.008 * np.sin(2.0 * pi * coordinate) * amplitude
    basis = np.asarray((1.0, -0.5, -0.5, 0.0, 0.0, 0.0))
    state = material.state_schema.initial_state()
    stress_history = []
    peeq_history = []
    residual_history = []
    old_strain = np.zeros(6)
    for step, value in enumerate(scalar, start=1):
        new_strain = value * basis
        response = material.update(
            constitutive.SmallStrainMaterialPointInput(
                strain_old=old_strain,
                strain_new=new_strain,
                time=float(step),
                time_increment=1.0,
                parameters=spec.parameters,
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        stress_history.append(response.cauchy_stress.tolist())
        peeq_history.append(float(material.state_schema.unpack(response.state_new)["peeq"]))
        residual_history.append(float(response.diagnostics["yield_residual"]))
        state = response.state_new
        old_strain = new_strain
    record = {
        "schema": "agentfem-learning.denim-material-point.v1",
        "specification": spec.to_dict(),
        "runtime": provider.evidence(spec),
        "maximum_absolute_stress_mpa": float(np.max(np.abs(stress_history)) / 1.0e6),
        "final_peeq": peeq_history[-1],
        "maximum_yield_residual_pa": float(np.max(np.abs(residual_history))),
        "stress": stress_history,
        "peeq": peeq_history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key not in {"stress", "peeq"}}, indent=2))


if __name__ == "__main__":
    main()
