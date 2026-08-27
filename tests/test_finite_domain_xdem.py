from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from agentfem import extensions, fracture, models, studies
from agentfem.step_providers import step_providers

from agentfem_learning.neural_fields.xdem import (
    UnsupportedFiniteDomainError,
    XDEMFiniteDomainStep,
    center_crack_domain_problem,
    crack_opening_sif_reports,
    displacement_bc,
    finite_domain_spec,
    point_displacement,
    rectangular_domain,
    static_crack_problem,
    stress_intensity_reports,
    tip_integration_plan,
    traction_bc,
    two_collinear_cracks_domain_problem,
    two_collinear_cracks_reference,
    xvem_mixed_mode_domain_problem,
)
from agentfem_learning.neural_fields.xdem.finite_domain_solver import (
    FiniteDomainVectorNetwork,
    problem_from_spec,
    train_finite_domain,
)
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
    with pytest.raises(UnsupportedFiniteDomainError, match="cannot be an active"):
        static_crack_problem(
            domain=domain,
            material=_material(),
            cracks=boundary_crack,
            conditions=(displacement_bc("fixed", "left", (0.0, 0.0)),),
        )

    supported_boundary_crack = fracture.crack_set(
        fracture.segment(
            "edge",
            start=(-1.0, 0.0),
            end=(0.0, 0.0),
            metadata={"active_ends": ("end",)},
        )
    )
    supported = static_crack_problem(
        domain=domain,
        material=_material(),
        cracks=supported_boundary_crack,
        conditions=(displacement_bc("fixed", "left", (0.0, 0.0)),),
    )
    assert supported.tip_ids == ("edge:start", "edge:end")
    assert supported.active_tip_ids == ("edge:end",)

    internal = fracture.crack_set(
        fracture.segment("main", start=(-0.5, 0.0), end=(0.5, 0.0))
    )
    with pytest.raises(UnsupportedFiniteDomainError, match="rigid-body modes"):
        static_crack_problem(
            domain=domain,
            material=_material(),
            cracks=internal,
            conditions=(displacement_bc("x_only", "left", (0.0, None)),),
        )


def test_point_gauges_remove_rigid_modes_without_clamping_remote_boundaries(tmp_path):
    material = _material()
    single = center_crack_domain_problem(material, remote_stress=1.0e6)
    double = two_collinear_cracks_domain_problem(material, remote_stress=1.0e6)

    assert single.summary()["rigid_body_constraint_rank"] == 3
    assert double.summary()["rigid_body_constraint_rank"] == 3
    assert single.tip_ids == ("main:start", "main:end")
    assert double.tip_ids == (
        "left:start",
        "left:end",
        "right:start",
        "right:end",
    )
    assert sum(hasattr(item, "point") for item in double.conditions) == 2
    restored = finite_domain_spec(double, domain_samples=128, boundary_samples=16)
    summary = restored.metadata["scientific_problem"]
    assert [item.get("location") for item in summary["conditions"]][-2:] == [
        "point",
        "point",
    ]
    network = FiniteDomainVectorNetwork(
        single,
        displacement_scale=1.0e6 * 24.0 / 210.0e9,
        hidden_layers=(8,),
    )
    assert network.representation_family == "additive_jump"
    assert len(network.jump_networks) == 1
    assert network.regular_network[0].in_features == 2
    assert network.has_hard_point_gauge is True
    assert network.tip_amplitudes[:, 0].detach().numpy() == pytest.approx(
        np.sqrt(np.pi / 2.0)
    )
    assert network.tip_amplitudes[:, 1].detach().numpy() == pytest.approx(0.0)
    gauge_displacement = network(network.gauge_points)
    rows = torch.arange(3)
    assert torch.allclose(
        gauge_displacement[rows, network.gauge_components],
        network.gauge_values,
        atol=1.0e-13,
        rtol=0.0,
    )
    sample = torch.tensor(((0.0, 0.25), (0.5, -0.25)), dtype=torch.float64)
    components = network.raw_displacement_components(sample)
    assert set(components) == {"regular_network", "jump_network", "williams_tip"}
    assert torch.allclose(sum(components.values()), network._raw_displacement(sample))
    smoke = train_finite_domain(
        finite_domain_spec(single, domain_samples=128, boundary_samples=16),
        ReferenceTrainingOptions(
            adam_epochs=2,
            lbfgs_steps=0,
            hidden_layers=(8,),
            seed=2026,
        ),
    )
    assert np.all(np.isfinite(smoke.losses))
    assert len(smoke.stress_intensity.reports) == 2
    assert smoke.metrics["training_quadrature_weight_error"] < 1.0e-12
    result = XDEMFiniteDomainStep(
        finite_domain_spec(single, domain_samples=128, boundary_samples=16),
        options=ReferenceTrainingOptions(
            adam_epochs=2,
            lbfgs_steps=0,
            hidden_layers=(8,),
            seed=2027,
        ),
        output=tmp_path,
    ).solve_result()
    assert result.metadata["published_benchmark"]["status"] in {
        "accepted",
        "failed",
    }
    assert result.artifacts["published_benchmark"].is_file()

    with pytest.raises(UnsupportedFiniteDomainError, match="outside the domain"):
        static_crack_problem(
            domain=rectangular_domain((-2.0, 2.0, -2.0, 2.0)),
            material=material,
            cracks=fracture.crack_set(
                fracture.segment("main", start=(-0.5, 0.0), end=(0.5, 0.0))
            ),
            conditions=(
                point_displacement("outside", (3.0, 0.0), (0.0, 0.0)),
                point_displacement("rotation", (2.0, -2.0), (None, 0.0)),
            ),
        )


def test_published_crack_coordinate_is_segment_local_and_scale_invariant():
    material = _material()
    problem = center_crack_domain_problem(
        material,
        half_crack_length=1.0,
        half_width=4.0,
        half_height=4.0,
        remote_stress=1.0,
    )
    network = FiniteDomainVectorNetwork(
        problem,
        displacement_scale=1.0e-6,
        hidden_layers=(8,),
        crack_decay=8.0,
    )
    epsilon = 1.0e-8
    points = torch.tensor(
        (
            (0.0, epsilon),
            (0.0, -epsilon),
            (1.25, epsilon),
            (-1.25, -epsilon),
            (1.0, epsilon),
        ),
        dtype=torch.float64,
    )
    coordinate = network.published_crack_coordinates(points)[:, 0]

    assert coordinate[0] == pytest.approx(1.0, rel=1.0e-6)
    assert coordinate[1] == pytest.approx(-1.0, rel=1.0e-6)
    assert coordinate[2:] == pytest.approx(0.0, abs=0.0)

    scaled = center_crack_domain_problem(
        material,
        half_crack_length=10.0,
        half_width=40.0,
        half_height=40.0,
        remote_stress=1.0,
    )
    scaled_network = FiniteDomainVectorNetwork(
        scaled,
        displacement_scale=1.0e-6,
        hidden_layers=(8,),
        crack_decay=8.0,
    )
    sample = torch.tensor(((0.35, 0.2),), dtype=torch.float64)
    assert torch.allclose(
        network.published_crack_coordinates(sample),
        scaled_network.published_crack_coordinates(10.0 * sample),
        atol=1.0e-14,
        rtol=1.0e-13,
    )

    face = torch.tensor(((0.25, 0.0),), dtype=torch.float64, requires_grad=True)
    bounded = network.bounded_sheet_coordinates(face)[:, 0]
    normal_derivative = torch.autograd.grad(bounded.sum(), face)[0][0, 1]
    assert normal_derivative == pytest.approx(0.0, abs=1.0e-14)


def test_published_crack_coordinate_supports_multiple_oriented_segments():
    problem = static_crack_problem(
        domain=rectangular_domain((-4.0, 4.0, -4.0, 4.0)),
        material=_material(),
        cracks=fracture.crack_set(
            fracture.segment("horizontal", start=(-1.0, -1.0), end=(1.0, -1.0)),
            fracture.segment("diagonal", start=(-1.0, 0.0), end=(1.0, 2.0)),
        ),
        conditions=(
            displacement_bc("fixed", "left", (0.0, 0.0)),
            traction_bc("pull", "right", (1.0, 0.0)),
        ),
        metadata={"neural_representation": "published_crack_coordinate"},
    )
    network = FiniteDomainVectorNetwork(
        problem,
        displacement_scale=1.0e-6,
        hidden_layers=(8,),
    )
    diagonal = problem.cracks.crack("diagonal")
    center = 0.5 * (
        np.asarray(diagonal.start, dtype=float)
        + np.asarray(diagonal.end, dtype=float)
    )
    normal = np.asarray(diagonal.normal, dtype=float)
    points = torch.tensor(
        np.vstack((center + 1.0e-8 * normal, center - 1.0e-8 * normal)),
        dtype=torch.float64,
    )
    values = network.published_crack_coordinates(points)

    assert values.shape == (2, 2)
    assert values[0, 1] == pytest.approx(1.0, rel=1.0e-6)
    assert values[1, 1] == pytest.approx(-1.0, rel=1.0e-6)
    assert network.regular_network[0].in_features == 4


def test_xvem_public_problem_has_exact_spatial_boundary_and_one_active_tip():
    problem = xvem_mixed_mode_domain_problem()
    spec = finite_domain_spec(problem, domain_samples=128, boundary_samples=16)
    restored = problem_from_spec(spec)
    network = FiniteDomainVectorNetwork(
        restored,
        displacement_scale=1.0e-4,
        hidden_layers=(8,),
    )

    assert problem.tip_ids == ("main:start", "main:end")
    assert problem.active_tip_ids == ("main:end",)
    assert len(spec.representations[0].enrichments) == 1
    assert network.has_hard_spatial_boundary is True
    assert network.has_hard_point_gauge is False
    assert np.allclose(
        (
            network.tip_amplitudes.detach() * network.tip_sif_scales[:, None]
        ).numpy(),
        0.0,
    )

    edge = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    points = torch.cat(
        (
            torch.stack((torch.full_like(edge, -1.0), edge), dim=1),
            torch.stack((torch.full_like(edge, 1.0), edge), dim=1),
            torch.stack((edge, torch.full_like(edge, -1.0)), dim=1),
            torch.stack((edge, torch.full_like(edge, 1.0)), dim=1),
        ),
        dim=0,
    )
    actual = network(points)
    expected = network._spatial_field(points)
    assert torch.allclose(actual, expected, atol=1.0e-14, rtol=1.0e-13)


def test_xvem_public_patch_test_accepts_mixed_mode_sif(tmp_path):
    problem = xvem_mixed_mode_domain_problem()
    result = XDEMFiniteDomainStep(
        finite_domain_spec(problem, domain_samples=512, boundary_samples=32),
        options=ReferenceTrainingOptions(
            adam_epochs=100,
            lbfgs_steps=2,
            hidden_layers=(24, 24),
            learning_rate=1.0e-3,
            seed=2026,
        ),
        output=tmp_path,
    ).solve_result()

    benchmark = result.metadata["published_benchmark"]
    assert benchmark["status"] == "accepted"
    assert benchmark["tips"][0]["tip_id"] == "main:end"
    assert max(benchmark["tips"][0]["relative_errors"]) < 0.05
    assert result.metadata["capability"]["validation_class"] == "extended_patch_test"
    assert result.metadata["tip_enrichment"]["active_tip_ids"] == ("main:end",)

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


def test_crack_opening_extrapolation_recovers_exact_mixed_mode_tips():
    cracks = _two_cracks()
    material = _material()
    targets = {
        "lower:start": (2.0e6, -0.5e6),
        "lower:end": (1.5e6, 0.25e6),
        "upper:start": (1.25e6, 0.75e6),
        "upper:end": (2.25e6, -0.25e6),
    }
    plan = tip_integration_plan(
        cracks,
        bounds=(-3.0, 3.0, -2.0, 2.0),
        radius_fractions=(0.2, 0.3, 0.4),
    )
    reports = crack_opening_sif_reports(
        _LocalWilliamsField(cracks, material, targets),
        cracks=cracks,
        material=material,
        plan=plan,
    )

    assert tuple(item.tip_id for item in reports) == tuple(targets)
    for report in reports:
        assert (report.k_i, report.k_ii) == pytest.approx(
            targets[report.tip_id], rel=2.0e-5
        )
        assert report.path_variation < 2.0e-5


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
    paraview = outcome.write_paraview(tmp_path / "direct_discontinuous.vtu")
    text = paraview.read_text(encoding="utf-8")
    assert 'Name="Displacement"' in text
    assert 'Name="CrackSide"' in text
    assert 'NumberOfComponents="3"' in text

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
    assert (tmp_path / "finite_domain_discontinuous.vtu").is_file()
    assert result.artifacts["paraview_discontinuous"].suffix == ".vtu"


def test_published_two_crack_reference_requires_all_four_tips():
    cracks = fracture.crack_set(
        fracture.segment("left", start=(-3.0, 0.0), end=(-1.0, 0.0)),
        fracture.segment("right", start=(1.0, 0.0), end=(3.0, 0.0)),
    )
    material = _material()
    reference = two_collinear_cracks_reference()
    targets = dict(reference.expected)
    plan = tip_integration_plan(
        cracks,
        bounds=(-8.0, 8.0, -6.0, 6.0),
        relative_path_tolerance=0.02,
    )
    reports = stress_intensity_reports(
        _LocalWilliamsField(cracks, material, targets),
        cracks=cracks,
        material=material,
        plan=plan,
    )
    comparison = reference.compare(reports)
    assert comparison["status"] == "accepted"
    assert len(comparison["tips"]) == 4
