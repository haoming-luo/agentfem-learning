"""Solve a three-dimensional bar with the verified local DENIM bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from agentfem import extensions, fields, materials, mesh, models, steps, studies
from case import specification
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("denim-global-bar.json"))
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
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((300.0e6, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=learned,
        incrementation=steps.fixed(4),
        progress=True,
    )
    result = step.solve_result()
    local_max = float(np.max(np.abs(displacement.value.x.array), initial=0.0))
    record = {
        "schema": "agentfem-learning.denim-global-bar.v1",
        "status": result.status,
        "converged": bool(step.last_solve_info.converged),
        "accepted_increments": len(step.accepted_increments),
        "maximum_displacement": float(domain.comm.allreduce(local_max, op=MPI.MAX)),
        "maximum_equivalent_stress_pa": step.state.equivalent_stress().global_max(),
        "provider": dict(learned.runtime_evidence),
        "specification_fingerprint": learned.specification.fingerprint,
    }
    if domain.comm.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
