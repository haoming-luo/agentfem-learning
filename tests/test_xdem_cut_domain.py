from __future__ import annotations

import numpy as np
import pytest
from agentfem import fracture

from agentfem_learning.neural_fields.xdem import (
    UnsupportedCutCellError,
    displacement_bc,
    rectangular_domain,
    static_crack_problem,
    straight_crack_cut_quadrature,
)


def _material():
    return fracture.linear_elastic_fracture_material(
        young_modulus=1000.0,
        poisson_ratio=0.25,
        assumption="plane_strain",
    )


def _problem(*cracks):
    return static_crack_problem(
        domain=rectangular_domain((-1.0, 1.0, -1.0, 1.0)),
        material=_material(),
        cracks=fracture.crack_set(*cracks),
        conditions=(displacement_bc("fixed_left", "left", (0.0, 0.0)),),
    )


def test_cut_domain_quadrature_conserves_area_and_tracks_inclined_crack_sides():
    problem = _problem(
        fracture.segment("inclined", start=(-0.6, -0.2), end=(0.7, 0.45))
    )

    rule = straight_crack_cut_quadrature(problem, 256)

    assert rule.weights.sum() == pytest.approx(problem.domain.area, rel=1.0e-12)
    assert set(np.unique(rule.side_codes[:, 0])) == {-1, 1}
    assert any(kind == "cut" for kind in rule.cell_kinds)
    assert any(kind == "tip" for kind in rule.cell_kinds)
    assert any(kind == "near_crack" for kind in rule.cell_kinds)
    assert len(rule.coordinates) > 256
    assert rule.summary()["schema_version"] == "0.2.0"
    assert rule.summary()["local_rules"]["tip_cell"] == "tensor_gauss_4x4"
    assert len(rule.fingerprint) == 64


def test_cut_domain_quadrature_keeps_diagonal_rule_strictly_one_sided():
    problem = _problem(
        fracture.segment("diagonal", start=(-0.75, -0.75), end=(0.75, 0.75))
    )

    rule = straight_crack_cut_quadrature(problem, 64)

    assert np.all(rule.side_codes != 0)
    assert any(kind in {"cut", "tip", "one_sided"} for kind in rule.cell_kinds)
    assert rule.weights.sum() == pytest.approx(4.0, rel=1.0e-12)


def test_cut_domain_variants_are_deterministic_but_independent():
    problem = _problem(
        fracture.segment("main", start=(-0.6, 0.0), end=(0.6, 0.0))
    )

    training = straight_crack_cut_quadrature(problem, 256, variant=2026)
    repeated = straight_crack_cut_quadrature(problem, 256, variant=2026)
    validation = straight_crack_cut_quadrature(problem, 256, variant=3026)

    assert training.fingerprint == repeated.fingerprint
    assert training.fingerprint != validation.fingerprint
    assert training.grid_shape != validation.grid_shape


def test_grid_aligned_crack_refines_faces_and_vertex_tips_without_area_loss():
    problem = _problem(
        fracture.segment("aligned", start=(-0.5, 0.0), end=(0.5, 0.0))
    )

    rule = straight_crack_cut_quadrature(problem, 256)

    assert rule.cell_kinds.count("aligned") >= 4
    assert rule.cell_kinds.count("tip") >= 16
    assert rule.weights.sum() == pytest.approx(problem.domain.area, rel=1.0e-12)
    assert set(np.unique(rule.side_codes[:, 0])) == {-1, 1}


def test_cut_domain_quadrature_fails_closed_when_one_cell_contains_two_cracks():
    problem = _problem(
        fracture.segment("lower", start=(-0.2, 0.05), end=(0.2, 0.05)),
        fracture.segment("upper", start=(-0.2, 0.08), end=(0.2, 0.08)),
    )

    with pytest.raises(UnsupportedCutCellError, match="increase integration density"):
        straight_crack_cut_quadrature(problem, 64)
