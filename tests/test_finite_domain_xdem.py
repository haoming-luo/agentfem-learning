from __future__ import annotations

import json

import numpy as np
import pytest
from agentfem import extensions, fracture, models, studies
from agentfem.step_providers import step_providers

from agentfem_learning.neural_fields.xdem import (
    UnsupportedFiniteDomainError,
    displacement_bc,
    finite_domain_spec,
    rectangular_domain,
    static_crack_problem,
    stress_intensity_reports,
    tip_integration_plan,
    traction_bc,
)
from agentfem_learning.neural_fields.xdem.finite_domain_solver import train_finite_domain
from agentfem_learning.neural_fields.xdem.reference import ReferenceTrainingOptions


def _material():
    return fracture.linear_elastic_fracture_material(
        young_modulus=210.0e9,
        poisson_ratio=0.3,
        assumption="plane_strain",
    )


def _two_cracks():
    return fracture.crack_set(
        fracture.segment("lower", start=(-1.0, -0.5), end=(1.0, -0.5)),
        fracture.segment("upper", start=(-1.0, 0.5), end=(1.0, 0.5)),
        name="two_separated_cracks",
    )


def test_finite_domain_contract_preserves_boundaries_cracks_and_tip_identity():
    problem = static_crack_problem(
        domain=rectangular_domain((-3.0, 3.0, -2.0, 2.0), name="plate"),
        material=_material(),
        cracks=_two_cracks(),
        conditions=(
            displacement_bc("fixed_left", "left", (0.0, 0.0)),
            traction_bc("pull_right", "right", (1.0e6, 0.0)),
        ),
        name="two_crack_plate",
    )
    spec = finite_domain_spec(problem, domain_samples=512, boundary_samples=64)

    assert problem.tip_ids == (
        "lower:start",
        "lower:end",
        "upper:start",
        "upper:end",
    )
    assert problem.summary()["rigid_body_constraint_rank"] == 3
    assert spec.metadata["problem"] == "finite_domain_static_xdem_d"
    assert spec.metadata["executable"] is True
    assert spec.metadata["external_benchmark_verified"] is False
    assert len(spec.representations[0].enrichments) == 4
    assert spec.required_checks[-3:] == (
        "per_tip_stress_intensity",
        "stress_intensity_path_variation",
        "optimization_repeatability",
    )
    json.dumps(spec.summary())


def test_finite_domain_contract_fails_closed_for_boundary_cracks_and_rigid_modes():
    domain = rectangular_domain((-1.0, 1.0, -1.0, 1.0))
    boundary_crack = fracture.crack_set(
        fracture.segment("edge", start=(-1.0, 0.0), end=(0.0, 0.0))
    )
    with pytest.raises(UnsupportedFiniteDomainError, match="strictly inside"):
        static_crack_problem(
            domain=domain,
            material=_material(),
            cracks=boundary_crack,
            conditions=(displacement_bc("fixed", "left", (0.0, 0.0)),),
        )

    internal = fracture.crack_set(fracture.segment("main", start=(-0.5, 0.0), end=(0.5, 0.0)))
    with pytest.raises(UnsupportedFiniteDomainError, match="rigid-body modes"):
        static_crack_problem(
            domain=domain,
            material=_material(),
            cracks=internal,
            conditions=(displacement_bc("x_only", "left", (0.0, None)),),
        )


class _LocalWilliamsField:
    """Manufactured disjoint tip neighborhoods for extractor verification."""

    def __init__(self, cracks, material, targets):
        self.cracks = cracks
        self.material = material
        self.targets = dict(targets)
        self.fields = {
            tip.tip_id: fracture.WilliamsField2D(
                tip,
                material,
                k_i=self.targets[tip.tip_id][0],
                k_ii=self.targets[tip.tip_id][1],
            )
            for tip in cracks.tips
        }

    def _evaluate(self, points, method):
        coordinates = np.asarray(points, dtype=float)
        tip_points = np.asarray([tip.point for tip in self.cracks.tips])
        selected = np.argmin(
            np.linalg.norm(coordinates[:, None, :] - tip_points[None, :, :], axis=2),
            axis=1,
        )
        sample = getattr(next(iter(self.fields.values())), method)(coordinates[:1])
        values = np.empty((len(coordinates), *sample.shape[1:]), dtype=float)
        for index, tip in enumerate(self.cracks.tips):
            mask = selected == index
            if np.any(mask):
                values[mask] = getattr(self.fields[tip.tip_id], method)(coordinates[mask])
        return values

    def displacement(self, points, *, side=None):
        del side
        return self._evaluate(points, "displacement")

    def displacement_gradient(self, points, *, side=None):
        del side
        return self._evaluate(points, "displacement_gradient")

    def stress(self, points, *, side=None):
        del side
        return self._evaluate(points, "stress")


def test_multi_crack_report_extracts_every_tip_with_stable_identity():
    cracks = _two_cracks()
    material = _material()
    targets = {
        "lower:start": (1.0e6, 0.1e6),
        "lower:end": (1.2e6, -0.2e6),
        "upper:start": (0.8e6, 0.3e6),
        "upper:end": (1.5e6, 0.4e6),
    }
    plan = tip_integration_plan(
        cracks,
        bounds=(-3.0, 3.0, -2.0, 2.0),
        radial_count=20,
        angular_count=80,
        relative_path_tolerance=0.02,
    )
    collection = stress_intensity_reports(
        _LocalWilliamsField(cracks, material, targets),
        cracks=cracks,
        material=material,
        plan=plan,
        metadata={"evidence": "manufactured_disjoint_tip_neighborhoods"},
    )

    assert collection.status == "accepted"
    assert tuple(item.tip_id for item in collection.reports) == tuple(targets)
    for tip_id, (expected_i, expected_ii) in targets.items():
        report = collection.report(tip_id)
        assert report.k_i == pytest.approx(expected_i, rel=3.0e-3)
        assert report.k_ii == pytest.approx(expected_ii, rel=3.0e-3)
        assert report.path_variation < 0.02
        assert len(report.integration_radii) == 3


def test_tip_plan_rejects_stale_crack_identity():
    cracks = _two_cracks()
    plan = tip_integration_plan(cracks, bounds=(-3.0, 3.0, -2.0, 2.0))
    changed = fracture.crack_set(fracture.segment("main", start=(-1.0, 0.0), end=(1.0, 0.0)))
    with pytest.raises(ValueError, match="different crack set"):
        stress_intensity_reports(
            _LocalWilliamsField(
                changed,
                _material(),
                {
                    "main:start": (1.0, 0.0),
                    "main:end": (1.0, 0.0),
                },
            ),
            cracks=changed,
            material=_material(),
            plan=plan,
        )


def test_finite_domain_solver_runs_two_cracks_and_returns_four_tip_reports(tmp_path):
    problem = static_crack_problem(
        domain=rectangular_domain((-3.0, 3.0, -2.0, 2.0)),
        material=_material(),
        cracks=_two_cracks(),
        conditions=(
            displacement_bc("fixed_left", "left", (0.0, 0.0)),
            traction_bc("pull_right", "right", (1.0e6, 0.0)),
        ),
    )
    spec = finite_domain_spec(problem, domain_samples=128, boundary_samples=16)
    outcome = train_finite_domain(
        spec,
        ReferenceTrainingOptions(
            adam_epochs=3,
            lbfgs_steps=0,
            hidden_layers=(8,),
            seed=31,
        ),
    )

    assert np.all(np.isfinite(outcome.losses))
    assert len(outcome.stress_intensity.reports) == 4
    assert set(outcome.crack_trace_id) == {"lower", "upper"}
    assert outcome.displacement.shape[1] == 2
    assert outcome.stress.shape[1:] == (2, 2)

    expected = "xdem_finite_domain_lefm_neural_field"
    if not any(item.name == expected for item in step_providers()):
        extensions.load_extension("agentfem-learning.xdem")
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain")
    )
    step = model.step(
        target=spec,
        epochs=2,
        lbfgs_steps=0,
        hidden_layers=(8,),
        output=tmp_path,
        seed=32,
    )
    result = step.solve_result()

    assert result.metadata["capability"]["supports_multiple_cracks"] is True
    assert len(result.metadata["stress_intensity"]["reports"]) == 4
    assert (tmp_path / "finite_domain_field.npz").is_file()
    assert (tmp_path / "stress_intensity.json").is_file()
