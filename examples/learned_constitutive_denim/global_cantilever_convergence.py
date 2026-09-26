"""Run a cost-bounded mesh and increment convergence study with DENIM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from agentfem import extensions, fields, materials, mesh, models, solvers, steps, studies
from agentfem.results import reaction_resultant
from case import specification
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem_learning.learned_constitutive.denim import (
    evaluate_structural_convergence,
)


def _global_max(domain, values) -> float:
    local = float(np.max(np.asarray(values), initial=0.0))
    return float(domain.comm.allreduce(local, op=MPI.MAX))


def solve_case(spec, *, cells: tuple[int, int, int], increments: int) -> dict[str, object]:
    started = time.perf_counter()
    domain = dolfinx_mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.asarray((4.0, 1.0, 1.0))],
        list(cells),
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(
            dimension=3,
            nonlinear=True,
            name="denim_cantilever_convergence",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    learned = model.material(materials.learned(spec))
    left = mesh.face(domain, axis="x", value=0.0)
    right = mesh.face(domain, axis="x", value=4.0)
    for component in range(3):
        model.fix(displacement, on=left, component=component)
    model.fix(displacement, on=right, component=1, value=-0.04)
    step = model.step(
        target=displacement,
        material=learned,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            maximum_iterations=24,
            line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()
    reaction = reaction_resultant(step, on=right, component=1)
    index_map = displacement.value.function_space.dofmap.index_map
    dofs = int(index_map.size_global) * int(
        displacement.value.function_space.dofmap.index_map_bs
    )
    cells_global = int(domain.topology.index_map(domain.topology.dim).size_global)
    stress = step.response.cauchy_stress.owned_values
    peeq = step.state.committed["peeq"].owned_values
    diagnostics = result.metadata["learned_constitutive"]["diagnostics"]
    elapsed = float(domain.comm.allreduce(time.perf_counter() - started, op=MPI.MAX))
    return {
        "cells_per_axis": list(cells),
        "cell_count": cells_global,
        "dof_count": dofs,
        "increments": int(increments),
        "status": result.status,
        "converged": bool(step.last_solve_info.converged),
        "accepted_increments": len(step.accepted_increments),
        "reaction_y": float(reaction),
        "maximum_absolute_stress_pa": _global_max(domain, np.abs(stress)),
        "maximum_peeq": _global_max(domain, peeq),
        "applicability_counts": diagnostics["applicability_counts"],
        "elapsed_seconds": elapsed,
    }


def _relative_change(previous: float, current: float) -> float:
    return abs(current - previous) / max(abs(current), np.finfo(float).eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("denim-global-convergence.json"),
    )
    args = parser.parse_args()
    extensions.load_extension("agentfem-learning.learned-constitutive")
    spec = specification(args.bundle.resolve())
    meshes = ((4, 1, 1), (8, 2, 2), (12, 3, 3))
    increment_levels = (4, 8, 16)
    cases: dict[tuple[tuple[int, int, int], int], dict[str, object]] = {}
    for selected_mesh, selected_increments in (
        *((value, 16) for value in meshes),
        *((meshes[-1], value) for value in increment_levels[:-1]),
    ):
        key = (selected_mesh, selected_increments)
        cases[key] = solve_case(
            spec,
            cells=selected_mesh,
            increments=selected_increments,
        )

    mesh_cases = [cases[(value, 16)] for value in meshes]
    increment_cases = [cases[(meshes[-1], value)] for value in increment_levels]
    record = {
        "schema": "agentfem-learning.denim-structural-convergence.v1",
        "problem_identity": "denim-v1-displacement-cantilever@1",
        "loading": {
            "beam_dimensions": [4.0, 1.0, 1.0],
            "prescribed_end_displacement_y": -0.04,
            "boundary_semantics": "left_face_clamped_right_face_y_displacement",
            "displacement_space": "continuous_lagrange_degree_2",
        },
        "axes": {
            "mesh": mesh_cases,
            "increment": increment_cases,
        },
        "convergence": {
            "mesh_reaction_last_two_relative_change": _relative_change(
                mesh_cases[-2]["reaction_y"], mesh_cases[-1]["reaction_y"]
            ),
            "mesh_peeq_last_two_relative_change": _relative_change(
                mesh_cases[-2]["maximum_peeq"], mesh_cases[-1]["maximum_peeq"]
            ),
            "increment_reaction_last_two_relative_change": _relative_change(
                increment_cases[-2]["reaction_y"], increment_cases[-1]["reaction_y"]
            ),
            "increment_peeq_last_two_relative_change": _relative_change(
                increment_cases[-2]["maximum_peeq"], increment_cases[-1]["maximum_peeq"]
            ),
        },
        "runtime": materials.learned(spec).implementation.runtime_evidence(),
        "execution_policy": {
            "unique_case_count": len(cases),
            "full_cartesian_product": False,
            "shared_reference_case": {
                "cells_per_axis": list(meshes[-1]),
                "increments": 16,
            },
        },
    }
    record["acceptance"] = evaluate_structural_convergence(record).summary()
    if MPI.COMM_WORLD.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
    if not record["acceptance"]["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
