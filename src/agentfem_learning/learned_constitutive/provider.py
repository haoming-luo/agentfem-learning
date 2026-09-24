# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Generic local PyTorch learned-constitutive provider."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from agentfem.constitutive import SmallStrainUserMaterial
from agentfem.learning import LearnedConstitutiveSpec

from .artifacts import ModelBundle, ModelBundleError, load_model_bundle
from .registry import architecture_loader
from .torch_runtime import TorchRuntime


@dataclass
class TorchConstitutiveProvider:
    """Lower framework-neutral specifications to cached PyTorch materials."""

    runtime: TorchRuntime = field(default_factory=TorchRuntime)
    name: str = "agentfem-learning.torch-constitutive"
    _cache: dict[tuple[str, str, str], SmallStrainUserMaterial] = field(
        default_factory=dict,
        init=False,
    )
    _load_seconds: dict[tuple[str, str, str], float] = field(default_factory=dict, init=False)

    def _validate_specification(
        self,
        specification: LearnedConstitutiveSpec,
        bundle: ModelBundle,
    ) -> None:
        manifest = bundle.manifest
        if specification.architecture_id != bundle.architecture_id:
            raise ModelBundleError("Specification architecture differs from model.json.")
        if specification.model_name != bundle.model_name:
            raise ModelBundleError("Specification model name differs from model.json.")
        if specification.model_version != bundle.model_version:
            raise ModelBundleError("Specification model version differs from model.json.")
        if specification.artifact_sha256 is not None:
            accepted = {bundle.manifest_sha256, bundle.weights_sha256}
            if specification.artifact_sha256 not in accepted:
                raise ModelBundleError("Specification checksum differs from the local bundle.")
        if specification.revision is not None:
            revision = manifest.get("model_revision")
            if revision is not None and specification.revision != revision:
                raise ModelBundleError("Specification revision differs from the local bundle.")
        if tuple(manifest["voigt_order"]) != tuple(
            specification.tangent_convention.component_order
        ):
            raise ModelBundleError("Voigt order differs between specification and bundle.")
        if manifest["shear_convention"] != specification.tangent_convention.shear_convention:
            raise ModelBundleError("Shear convention differs between specification and bundle.")
        if manifest["kinematics"] != specification.tangent_convention.kinematic_measure:
            raise ModelBundleError("Kinematics differ between specification and bundle.")
        if manifest["stress_measure"] != specification.tangent_convention.stress_measure:
            raise ModelBundleError("Stress measure differs between specification and bundle.")
        expected_parameters = {str(item["name"]): item for item in manifest["parameter_schema"]}
        declared_parameters = {item.name: item for item in specification.parameter_schema}
        if expected_parameters.keys() != declared_parameters.keys():
            raise ModelBundleError("Material parameter names differ from the model bundle.")
        for name, expected in expected_parameters.items():
            declared = declared_parameters[name]
            for key in ("unit", "minimum", "maximum"):
                if expected.get(key) != getattr(declared, key):
                    raise ModelBundleError(
                        f"Material parameter {name!r} {key} differs from the bundle."
                    )
        if tuple(manifest["required_inputs"]) != tuple(specification.required_inputs):
            raise ModelBundleError("Required inputs differ from the model bundle.")
        manifest_capabilities = manifest["capabilities"]
        available = {
            name
            for name in ("stress", "state", "batch", "energy", "diagnostics")
            if manifest_capabilities.get(name) is True
        }
        if manifest_capabilities.get("tangent") not in {None, "none"}:
            available.add("consistent_tangent")
        if not set(specification.capabilities).issubset(available):
            raise ModelBundleError("Requested capabilities differ from the model bundle.")
        allowed_dtypes = tuple(manifest["dtype_policy"].get("allowed", ()))
        if specification.dtype_policy not in allowed_dtypes:
            raise ModelBundleError("Requested dtype is not allowed by the model bundle.")
        if self.runtime.dtype != specification.dtype_policy:
            raise ModelBundleError(
                "Provider runtime dtype differs from the immutable specification."
            )
        if dict(manifest["applicability_domain"]) != dict(specification.applicability_domain):
            raise ModelBundleError("Applicability domain differs from the model bundle.")
        for key in ("dataset_id", "dataset_revision"):
            if manifest.get(key) != getattr(specification, key):
                raise ModelBundleError(f"{key} differs from the model bundle.")

    def create(
        self,
        specification: LearnedConstitutiveSpec,
    ) -> SmallStrainUserMaterial:
        bundle = load_model_bundle(Path(specification.artifact))
        self._validate_specification(specification, bundle)
        key = (bundle.manifest_sha256, self.runtime.device, self.runtime.dtype)
        material = self._cache.get(key)
        if material is None:
            started = time.perf_counter()
            loader = architecture_loader(bundle.architecture_id)
            material = loader(bundle, self.runtime)
            self._cache[key] = material
            self._load_seconds[key] = time.perf_counter() - started
        if material.state_schema.summary() != specification.state_schema.summary():
            raise ModelBundleError("State schema differs between specification and bundle.")
        if material.tangent_convention != specification.tangent_convention:
            raise ModelBundleError(
                "Tangent convention differs between specification and bundle."
            )
        return material

    def evidence(
        self,
        specification: LearnedConstitutiveSpec,
    ) -> dict[str, object]:
        bundle = load_model_bundle(Path(specification.artifact))
        key = (bundle.manifest_sha256, self.runtime.device, self.runtime.dtype)
        return {
            **bundle.summary(),
            **self.runtime.summary(),
            "provider": self.name,
            "tangent_generation": bundle.manifest["capabilities"].get("tangent"),
            "model_load_seconds": self._load_seconds.get(key),
            "offline_runtime": True,
        }


TORCH_CONSTITUTIVE_PROVIDER = TorchConstitutiveProvider()


__all__ = ["TORCH_CONSTITUTIVE_PROVIDER", "TorchConstitutiveProvider"]
