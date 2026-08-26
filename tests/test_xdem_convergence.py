from __future__ import annotations

from types import SimpleNamespace

from agentfem import fracture

from agentfem_learning.neural_fields.xdem import (
    center_crack_domain_problem,
    finite_domain_spec,
    run_convergence_slices,
    run_finite_domain_convergence,
)
from agentfem_learning.neural_fields.xdem.reference import ReferenceTrainingOptions


def _outcome(scale):
    reports = tuple(
        SimpleNamespace(
            tip_id=tip_id,
            k_i=scale * value,
            k_ii=0.0,
            j_integral=scale * scale * value,
            path_variation=0.01,
        )
        for tip_id, value in (
            ("lower:start", 1.0),
            ("lower:end", 1.1),
            ("upper:start", 1.1),
            ("upper:end", 1.0),
        )
    )
    return SimpleNamespace(
        metrics={"relative_boundary_error": 0.01},
        stress_intensity=SimpleNamespace(reports=reports),
    )


def test_multi_axis_convergence_keeps_four_tip_evidence_and_accepts_stable_levels():
    scales = {
        ("network", "coarse"): 0.96,
        ("network", "fine"): 1.0,
        ("integration", "coarse"): 0.97,
        ("integration", "fine"): 1.0,
        ("seed", "2026"): 1.01,
        ("seed", "2027"): 1.0,
        ("rings", "24x96"): 0.98,
        ("rings", "36x144"): 1.0,
    }

    def runner(controls):
        return _outcome(scales[(controls["axis"], controls["level"])])

    slices = {
        axis: tuple(
            (level, {"axis": axis, "level": level})
            for level in levels
        )
        for axis, levels in {
            "network": ("coarse", "fine"),
            "integration": ("coarse", "fine"),
            "seed": ("2026", "2027"),
            "rings": ("24x96", "36x144"),
        }.items()
    }
    required_tips = ("lower:start", "lower:end", "upper:start", "upper:end")
    report = run_convergence_slices(
        slices,
        runner=runner,
        relative_tolerance=0.08,
        required_tip_ids=required_tips,
    )
    summary = report.summary()
    assert summary["status"] == "accepted"
    assert {item["axis"] for item in summary["axes"]} == set(slices)
    assert all(len(case["tips"]) == 4 for case in summary["cases"])
    assert all(axis["complete_tip_set"] for axis in summary["axes"])


def test_finite_domain_convergence_runs_four_axes_sequentially(tmp_path):
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1000.0,
        poisson_ratio=0.25,
        assumption="plane_strain",
    )
    problem = center_crack_domain_problem(material, remote_stress=1.0)
    report = run_finite_domain_convergence(
        finite_domain_spec(problem, domain_samples=64, boundary_samples=16),
        ReferenceTrainingOptions(
            adam_epochs=1,
            lbfgs_steps=0,
            hidden_layers=(4,),
            seed=11,
        ),
        network_layers=((4,), (6,)),
        integration_counts=(64, 96),
        seeds=(11, 12),
        ring_resolutions=((8, 24), (10, 32)),
        relative_tolerance=1.0,
        path_variation_tolerance=1.0,
        output=tmp_path / "convergence.json",
    )

    assert report.axes == ("network", "integration", "seed", "rings")
    assert len(report.cases) == 8
    assert all(len(case.tips) == 2 for case in report.cases)
    assert (tmp_path / "convergence.json").is_file()
