# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from agentfem import constitutive

from agentfem_learning.learned_constitutive import (
    read_uniaxial_csv,
    run_small_strain_material_history,
    simplify_uniaxial_history,
    truncate_to_strain_domain,
    uniaxial_stress_free_path,
    uniaxial_stress_metrics,
    validate_sample_partitions,
)


class _ElasticMaterial:
    rate_independent = True

    def __init__(self, young=200.0e9, poisson=0.3):
        self.parameters = {"young": young, "poisson": poisson}
        self.state_schema = constitutive.MaterialStateSchema("test.empty")
        self.parameter_schema = constitutive.MaterialParameterSchema(
            "test.elastic",
            (
                constitutive.MaterialParameter("young", "Pa", lower=0.0),
                constitutive.MaterialParameter("poisson", "1", lower=-1.0, upper=0.5),
            ),
        )
        self.tangent_convention = constitutive.small_strain_tangent_convention()

    def update(self, point):
        young = self.parameters["young"]
        poisson = self.parameters["poisson"]
        shear = young / (2.0 * (1.0 + poisson))
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        stress = lame * np.trace(point.strain_new) * np.eye(3) + 2.0 * shear * point.strain_new
        tangent = np.zeros((6, 6))
        tangent[:3, :3] = lame
        tangent[np.arange(3), np.arange(3)] += 2.0 * shear
        tangent[np.arange(3, 6), np.arange(3, 6)] = 2.0 * shear
        return constitutive.SmallStrainMaterialPointOutput(
            cauchy_stress=stress,
            consistent_tangent=tangent,
            state_new=point.state_old,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
        )


class _ScopedElasticMaterial(_ElasticMaterial):
    def update(self, point):
        response = super().update(point)
        if abs(point.strain_new[0, 0]) > 0.001:
            return replace(response, applicability="out_of_domain")
        return response


def _write_history(path, count=101):
    coordinate = np.linspace(0.0, 10.0, count)
    strain = 0.01 * np.sin(2.0 * np.pi * coordinate / coordinate[-1])
    stress = 200.0e9 * strain
    path.write_text(
        "time_s,strain_percent,stress_mpa\n"
        + "".join(
            f"{time},{100.0 * value},{sigma / 1.0e6}\n"
            for time, value, sigma in zip(coordinate, strain, stress, strict=True)
        ),
        encoding="utf-8",
    )


def test_explicit_csv_mapping_hash_domain_and_reduction(tmp_path):
    source = tmp_path / "history.csv"
    _write_history(source)
    history = read_uniaxial_csv(
        source,
        sample_id="coupon-1",
        source_identifier="doi:10.0000/example",
        license="CC-BY-4.0",
        coordinate_column="time_s",
        strain_column="strain_percent",
        stress_column="stress_mpa",
        strain_scale=0.01,
        stress_scale=1.0e6,
    )
    assert history.point_count == 101
    assert len(history.source_sha256) == 64
    bounded = truncate_to_strain_domain(history, 0.008)
    assert np.max(np.abs(bounded.strain)) <= 0.008
    reduced = simplify_uniaxial_history(history, maximum_points=19)
    assert 3 <= reduced.point_count <= 19
    assert reduced.strain[0] == pytest.approx(history.strain[0])
    assert reduced.strain[-1] == pytest.approx(history.strain[-1])


def test_mixed_control_recovers_uniaxial_stress_and_poisson_contraction(tmp_path):
    source = tmp_path / "uniaxial.csv"
    source.write_text(
        "time,strain,stress\n0,0,0\n1,0.001,200000000\n2,0.002,400000000\n",
        encoding="utf-8",
    )
    measured = read_uniaxial_csv(
        source,
        sample_id="elastic-coupon",
        source_identifier="doi:10.0000/elastic",
        license="CC-BY-4.0",
        coordinate_column="time",
        strain_column="strain",
        stress_column="stress",
    )
    path = uniaxial_stress_free_path(measured)
    response = run_small_strain_material_history(_ElasticMaterial(), path)
    assert response.path.control == "mixed"
    assert response.path.strain[-1, 1, 1] == pytest.approx(-0.3 * 0.002)
    assert response.path.strain[-1, 2, 2] == pytest.approx(-0.3 * 0.002)
    assert response.stress[-1, 0, 0] == pytest.approx(400.0e6)
    assert np.max(np.abs(response.stress[:, 1:, 1:])) < 1.0e-4
    assert response.diagnostics[-1]["maximum_stress_control_residual"] < 1.0e-4


def test_metrics_and_sample_partitions_are_explicit(tmp_path):
    source = tmp_path / "history.csv"
    _write_history(source, count=21)
    history = read_uniaxial_csv(
        source,
        sample_id="coupon-1",
        source_identifier="doi:10.0000/example",
        license="CC-BY-4.0",
        coordinate_column="time_s",
        strain_column="strain_percent",
        stress_column="stress_mpa",
        strain_scale=0.01,
        stress_scale=1.0e6,
    )
    metrics = uniaxial_stress_metrics(history, history.stress)
    assert metrics["stress_rmse_pa"] == pytest.approx(0.0)
    assert metrics["signed_work_relative_error"] == pytest.approx(0.0)
    assert validate_sample_partitions(
        {"a", "b", "c"}, fit=["a"], validation=["b"], held_out=["c"]
    )["held_out"] == ["c"]
    with pytest.raises(ValueError, match="overlap"):
        validate_sample_partitions(
            {"a", "b", "c"}, fit=["a"], validation=["b"], held_out=["a"]
        )


def test_partial_history_reports_uncommitted_scope_failure(tmp_path):
    source = tmp_path / "scope.csv"
    source.write_text(
        "time,strain,stress\n0,0,0\n1,0.001,200000000\n2,0.002,400000000\n",
        encoding="utf-8",
    )
    measured = read_uniaxial_csv(
        source,
        sample_id="scope-coupon",
        source_identifier="doi:10.0000/scope",
        license="CC-BY-4.0",
        coordinate_column="time",
        strain_column="strain",
        stress_column="stress",
    )
    response = run_small_strain_material_history(
        _ScopedElasticMaterial(),
        uniaxial_stress_free_path(measured),
        failure_policy="return_partial",
    )
    assert not response.completed
    assert not response.accepted
    assert response.path.point_count == 2
    assert response.failure["step_index"] == 2
    assert response.failure["state_commit"] == "failed_trial_not_committed"
