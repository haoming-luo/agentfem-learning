from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from agentfem import extensions, fracture, models, provenance, studies
from agentfem.step_providers import step_providers

from agentfem_learning.neural_fields.xdem import (
    WilliamsVectorNetwork,
    mode_i_tip_spec,
    vector_tip_spec,
)
from agentfem_learning.neural_fields.xdem.reference import ReferenceTrainingOptions
from agentfem_learning.neural_fields.xdem.vector_reference import (
    train_vector_reference,
    williams_vector_field,
)


def _activate_extension():
    expected = "xdem_vector_lefm_neural_field"
    if any(item.name == expected for item in step_providers()):
        return
    extensions.load_extension("agentfem-learning.xdem")


@pytest.mark.parametrize("assumption", ["plane_stress", "plane_strain"])
def test_torch_williams_vector_field_matches_core_reference(assumption):
    cracks = fracture.crack_set(
        fracture.segment("branch_cut", start=(-1.0, 0.0), end=(0.0, 0.0))
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1200.0,
        poisson_ratio=0.23,
        assumption=assumption,
    )
    reference = fracture.WilliamsField2D(
        cracks.tip("branch_cut:end"), material, k_i=3.0, k_ii=-1.5
    )
    points = np.asarray(
        ((0.4, 0.2), (-0.3, 0.15), (-0.3, -0.15), (0.1, -0.45))
    )

    actual = williams_vector_field(
        torch.tensor(points, dtype=torch.float64),
        young_modulus=1200.0,
        poisson_ratio=0.23,
        assumption=assumption,
        k_i=3.0,
        k_ii=-1.5,
    )

    np.testing.assert_allclose(
        actual.detach().numpy(), reference.displacement(points), rtol=1.0e-12
    )


def test_vector_spec_preserves_material_crack_and_mode_semantics():
    spec = vector_tip_spec(
        assumption="plane_stress", k_i=2.0, k_ii=-0.5, domain_samples=256
    )

    assert spec.metadata["problem"] == "williams_vector_tip"
    assert spec.metadata["material"]["assumption"] == "plane_stress"
    assert spec.metadata["loading"] == {"K_I": 2.0, "K_II": -0.5}
    assert spec.fields[0].components == ("U1", "U2")
    assert spec.integration.validation.independent_of == (
        "vector_slit_annulus_energy_points",
    )
    json.dumps(spec.summary())


def test_vector_enrichment_has_jump_only_on_the_declared_branch_cut():
    network = WilliamsVectorNetwork(
        radius=1.0,
        tip_core_radius=0.05,
        young_modulus=1000.0,
        poisson_ratio=0.25,
        assumption="plane_strain",
        reference_k_i=1.0,
        reference_k_ii=0.0,
        hidden_layers=(8,),
    )
    epsilon = 1.0e-8
    points = torch.tensor(
        (
            (0.5, epsilon),
            (0.5, -epsilon),
            (-0.5, epsilon),
            (-0.5, -epsilon),
        ),
        dtype=torch.float64,
    )
    values = network(points).detach()

    assert torch.linalg.vector_norm(values[0] - values[1]) < 1.0e-8
    assert torch.linalg.vector_norm(values[2] - values[3]) > 1.0e-5


def test_vector_reference_training_returns_common_sif_evidence():
    spec = mode_i_tip_spec(domain_samples=256, boundary_samples=32)
    outcome = train_vector_reference(
        spec,
        ReferenceTrainingOptions(
            adam_epochs=20,
            lbfgs_steps=0,
            hidden_layers=(8,),
            seed=13,
        ),
    )

    assert np.all(np.isfinite(outcome.losses))
    assert outcome.losses[-1] < outcome.losses[0]
    assert outcome.stress_intensity.tip_id == "branch_cut:end"
    assert outcome.stress_intensity.metadata["provider"] == "agentfem-learning.xdem"


def test_vector_step_returns_verified_result_and_discontinuous_fields(tmp_path):
    _activate_extension()
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        name="vector_reference",
    )
    result = model.step(
        target=vector_tip_spec(assumption="plane_stress", k_i=1.0, k_ii=0.5),
        epochs=500,
        output=tmp_path,
        seed=2026,
    ).solve_result()

    assert result.trust_level == "verified"
    assert result.quantity("relative_l2_error") < 0.03
    assert result.quantity("stress_intensity_relative_error") < 0.03
    assert result.quantity("stress_intensity_path_variation") < 0.03
    assert (tmp_path / "vector_tip_field.npz").is_file()
    assert (tmp_path / "stress_intensity.json").is_file()
    with np.load(tmp_path / "vector_tip_field.npz") as field:
        sides = field["crack_trace_side"]
        trace = field["crack_trace_displacement"]
    assert set(sides.tolist()) == {-1, 1}
    half = len(trace) // 2
    assert np.max(np.linalg.norm(trace[:half] - trace[half:], axis=1)) > 1.0e-5
    assert provenance.verify_manifest(tmp_path / "result.json").verified is True


def test_vector_provider_rejects_study_spec_assumption_mismatch():
    _activate_extension()
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain")
    )

    with pytest.raises(ValueError, match="same plane-stress"):
        model.step(target=mode_i_tip_spec(assumption="plane_stress"))
