# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest
from agentfem.constitutive import (
    MaterialTangentConvention,
    SmallStrainMaterialBatchInput,
    SmallStrainMaterialPointInput,
    check_small_strain_material_tangent,
)
from agentfem.learning import LearnedConstitutiveSpec, MaterialParameterSpec

from agentfem_learning.learned_constitutive.artifacts import (
    ModelBundleError,
    file_sha256,
    load_model_bundle,
)
from agentfem_learning.learned_constitutive.denim import DENIM, state_schema
from agentfem_learning.learned_constitutive.denim.export import denim_manifest
from agentfem_learning.learned_constitutive.denim.loader import load_denim_v1
from agentfem_learning.learned_constitutive.provider import TorchConstitutiveProvider
from agentfem_learning.learned_constitutive.registry import (
    ArchitectureRegistryError,
    architecture_loader,
    register_architecture,
)

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")


def _bundle(tmp_path, *, channels=2):
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    law = DENIM(channels=channels).double()
    weights = root / "weights.safetensors"
    safetensors.save_file(law.state_dict(), str(weights))
    manifest = denim_manifest(
        weights_sha256=file_sha256(weights),
        channels=channels,
        model_revision="fixed-test-revision",
        dataset_id="tests/material-paths",
        dataset_revision="fixed-dataset-revision",
    )
    manifest_path = root / "model.json"
    readme_path = root / "README.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    readme_path.write_text("# Test model bundle\n", encoding="utf-8")
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(path)}  {path.name}\n"
            for path in (weights, manifest_path, readme_path)
        ),
        encoding="utf-8",
    )
    return root


def _spec(root, *, channels=2):
    return LearnedConstitutiveSpec(
        provider="agentfem-learning.torch-constitutive",
        architecture_id="denim.v1",
        artifact=str(root),
        revision="fixed-test-revision",
        model_name="denim-expanded",
        model_version="1.1.0",
        tangent_convention=MaterialTangentConvention.cauchy_small_strain(symmetric=False),
        parameter_schema=(
            MaterialParameterSpec("young", "Pa", minimum=0.0),
            MaterialParameterSpec("poisson", "1", minimum=-1.0, maximum=0.5),
            MaterialParameterSpec("yield_stress", "Pa", minimum=0.0),
        ),
        parameters={
            "young": 190.0e9,
            "poisson": 0.3,
            "yield_stress": 280.0e6,
        },
        state_schema=state_schema(channels),
        required_inputs=("strain_new", "state_old", "parameters"),
        capabilities=("stress", "state", "batch", "energy", "consistent_tangent"),
        dtype_policy="float64",
        applicability_domain={
            "policy": "report_without_silent_fallback",
            "maximum_absolute_strain": 0.03,
            "maximum_peeq": 0.03,
        },
        dataset_id="tests/material-paths",
        dataset_revision="fixed-dataset-revision",
    )


def _material(tmp_path, *, channels=2):
    register_architecture("denim.v1", load_denim_v1, replace=True)
    provider = TorchConstitutiveProvider()
    material = provider.create(_spec(_bundle(tmp_path, channels=channels), channels=channels))
    return provider, material


def test_bundle_checksum_failure_is_fail_closed(tmp_path):
    root = _bundle(tmp_path)
    record = json.loads((root / "model.json").read_text())
    record["weights_sha256"] = "0" * 64
    (root / "model.json").write_text(json.dumps(record))
    with pytest.raises(ModelBundleError, match="checksum mismatch"):
        load_model_bundle(root)


def test_bundle_schema_and_nested_manifest_are_immutable(tmp_path):
    bundle = load_model_bundle(_bundle(tmp_path))
    assert bundle.summary()["schema"] == "agentfem.learned_constitutive.bundle"
    assert bundle.summary()["schema_version"] == "1.0.0"
    with pytest.raises(TypeError):
        bundle.manifest["architecture"]["channels"] = 7


def test_missing_architecture_has_stable_error_code():
    with pytest.raises(
        ArchitectureRegistryError,
        match="AFM-LEARNING-ARCHITECTURE-001",
    ):
        architecture_loader("tests.not-registered")


def test_state_schema_tracks_memory_channels():
    assert state_schema(1).size == 19
    assert state_schema(2).size == 25
    assert state_schema(4).size == 37


def test_provider_rejects_missing_bundle_and_state_schema_mismatch(tmp_path):
    provider = TorchConstitutiveProvider()
    with pytest.raises(ModelBundleError, match="must contain model.json"):
        provider.create(_spec(tmp_path / "missing"))
    root = _bundle(tmp_path / "mismatch", channels=2)
    register_architecture("denim.v1", load_denim_v1, replace=True)
    with pytest.raises(ModelBundleError, match="State schema differs"):
        provider.create(_spec(root, channels=1))


def test_provider_rejects_model_identity_mismatch(tmp_path):
    root = _bundle(tmp_path)
    specification = _spec(root)
    specification = LearnedConstitutiveSpec.from_dict(
        {**specification.to_dict(), "model_version": "9.9.9"}
    )
    with pytest.raises(ModelBundleError, match="model version differs"):
        TorchConstitutiveProvider().create(specification)


def test_scalar_batch_and_autodiff_tangent(tmp_path):
    provider, material = _material(tmp_path)
    schema = material.state_schema
    parameters = {"young": 190.0e9, "poisson": 0.3, "yield_stress": 280.0e6}
    strains = np.asarray(
        [
            [2.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.4e-3, -4.0e-4, 0.0, 2.0e-4, 0.0, 0.0],
        ]
    )
    request = SmallStrainMaterialBatchInput(
        strain_old=np.zeros_like(strains),
        strain_new=strains,
        time=1.0,
        time_increment=1.0,
        parameters=parameters,
        state_old=np.repeat(schema.initial_state()[None, :], 2, axis=0),
        state_schema=schema,
    )
    response = material.update_batch(request)
    assert response.cauchy_stress.shape == (2, 6)
    assert response.consistent_tangent.shape == (2, 6, 6)
    assert response.state_new.shape == (2, 25)
    assert np.all(response.stored_energy_density >= 0.0)
    assert set(response.energy_density_components) == {
        "elastic_storage",
        "hardening_storage_increment",
        "isotropic_hardening_storage",
        "kinematic_hardening_storage",
        "modeled_dissipation_increment",
        "plastic_work_increment",
    }
    point = request.point(0)
    scalar = material.update(point)
    np.testing.assert_allclose(scalar.cauchy_stress, response.cauchy_stress[0])
    np.testing.assert_allclose(scalar.state_new, response.state_new[0])
    assert scalar.energy_density_components == {
        name: pytest.approx(values[0])
        for name, values in response.energy_density_components.items()
    }
    check = check_small_strain_material_tangent(
        material,
        point,
        relative_step=1.0e-8,
        tolerance=2.0e-5,
    )
    assert check.accepted, check.summary()
    requested = _bundle(tmp_path / "evidence")
    evidence = provider.evidence(_spec(requested))
    assert evidence["offline_runtime"] is True
    assert evidence["framework"] == "pytorch"
    assert requested.is_dir()


def test_elastic_origin_tangent_is_finite(tmp_path):
    _, material = _material(tmp_path)
    point = SmallStrainMaterialPointInput(
        strain_old=np.zeros(6),
        strain_new=np.zeros(6),
        time=0.0,
        time_increment=1.0,
        parameters={"young": 190.0e9, "poisson": 0.3, "yield_stress": 280.0e6},
        state_old=material.state_schema.initial_state(),
        state_schema=material.state_schema,
    )
    response = material.update(point)
    assert np.all(np.isfinite(response.consistent_tangent))
    check = check_small_strain_material_tangent(
        material,
        point,
        relative_step=1.0e-9,
        tolerance=1.0e-7,
    )
    assert check.accepted, check.summary()


def test_provider_caches_one_loaded_model(tmp_path):
    root = _bundle(tmp_path)
    specification = _spec(root)
    register_architecture("denim.v1", load_denim_v1, replace=True)
    provider = TorchConstitutiveProvider()
    assert provider.create(specification) is provider.create(specification)


def test_nonfinite_or_mismatched_state_is_rejected(tmp_path):
    _, material = _material(tmp_path)
    schema = material.state_schema
    with pytest.raises(ValueError, match="state_old"):
        SmallStrainMaterialPointInput(
            strain_old=np.zeros(6),
            strain_new=np.zeros(6),
            time=1.0,
            time_increment=1.0,
            parameters={"young": 1.0, "poisson": 0.3, "yield_stress": 1.0},
            state_old=np.full(schema.size, np.nan),
            state_schema=schema,
        )
