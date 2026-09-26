"""Solve a three-dimensional bar with a prepared DENIM bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from agentfem import extensions, fields, materials, mesh, models, solvers, steps, studies
from case import specification
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem_learning.learned_constitutive.denim import evaluate_global_bar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("denim-global-bar.json"))
    parser.add_argument("--displacement", type=float, default=4.0e-3)
    args = parser.parse_args()
    extensions.load_extension("agentfem-learning.learned-constitutive")

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)
    model = models.create(
        study=studies.static_solid(
            dimension=3,
            nonlinear=True,
            name="denim_global_bar",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    learned = model.material(materials.learned(specification(args.bundle.resolve())))
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=args.displacement,
    )
    step = model.step(
        target=displacement,
        material=learned,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            maximum_iterations=18,
            line_search="backtracking",
        ),
        progress=True,
    )
    result = step.solve_result()
    local_displacement = float(np.max(np.abs(displacement.value.x.array), initial=0.0))
    stress = step.response.cauchy_stress.owned_values
    peeq = step.state.committed["peeq"].owned_values
    local_stress = float(np.max(np.abs(stress), initial=0.0))
    local_peeq = float(np.max(peeq, initial=0.0))
    record = {
        "schema": "agentfem-learning.denim-global-bar.v1",
        "problem_identity": "denim-v1-symmetry-bar@1",
        "status": result.status,
        "converged": bool(step.last_solve_info.converged),
        "accepted_increments": len(step.accepted_increments),
        "maximum_displacement": float(domain.comm.allreduce(local_displacement, op=MPI.MAX)),
        "maximum_absolute_stress_pa": float(domain.comm.allreduce(local_stress, op=MPI.MAX)),
        "maximum_peeq": float(domain.comm.allreduce(local_peeq, op=MPI.MAX)),
        "provider": learned.provider.summary(),
        "specification_fingerprint": learned.specification.fingerprint,
        "learned_constitutive": result.metadata["learned_constitutive"],
    }
    record["acceptance"] = evaluate_global_bar(record).summary()
    if domain.comm.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
