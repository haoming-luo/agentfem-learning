# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Verified, local-only model bundles for constitutive inference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODEL_BUNDLE_SCHEMA = "agentfem.learned_constitutive.bundle"
MODEL_BUNDLE_SCHEMA_VERSION = "1.0.0"


class ModelBundleError(RuntimeError):
    """A local model bundle is missing, corrupt, or incompatible."""

    code = "AFM-LEARNING-BUNDLE-001"

    def __init__(self, message: str):
        super().__init__(f"{self.code}: {message}")


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelBundle:
    """Verified model manifest and immutable local artifact identity."""

    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    weights_path: Path
    weights_sha256: str

    @property
    def architecture_id(self) -> str:
        return str(self.manifest["architecture_id"])

    @property
    def model_name(self) -> str:
        return str(self.manifest["model_name"])

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    def summary(self) -> dict[str, object]:
        return {
            "kind": "learned_constitutive_model_bundle",
            "schema": self.manifest["schema"],
            "schema_version": self.manifest["schema_version"],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture_id": self.architecture_id,
            "manifest_sha256": self.manifest_sha256,
            "weights_sha256": self.weights_sha256,
            "weights_format": self.manifest["weights_format"],
            "dataset_id": self.manifest.get("dataset_id"),
            "dataset_revision": self.manifest.get("dataset_revision"),
        }

    as_dict = summary


def load_model_bundle(path) -> ModelBundle:
    """Load and verify a prepared local bundle without network access."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "model.json"
    checksum_path = root / "SHA256SUMS"
    readme_path = root / "README.md"
    if not root.is_dir() or not all(
        item.is_file() for item in (manifest_path, checksum_path, readme_path)
    ):
        raise ModelBundleError(
            f"Model bundle {root} must contain model.json, weights, "
            "SHA256SUMS, and README.md. Downloading is an explicit preparation "
            "step and never occurs inside a solve."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError(f"Cannot read model manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelBundleError("model.json must contain one JSON object.")
    required = {
        "schema",
        "schema_version",
        "model_name",
        "model_version",
        "architecture_id",
        "provider_compatibility",
        "kinematics",
        "stress_measure",
        "voigt_order",
        "shear_convention",
        "parameter_schema",
        "state_schema",
        "required_inputs",
        "capabilities",
        "dtype_policy",
        "applicability_domain",
        "weights_file",
        "weights_format",
        "weights_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ModelBundleError(f"Model manifest is missing {missing!r}.")
    if manifest["schema"] != MODEL_BUNDLE_SCHEMA:
        raise ModelBundleError(f"Unsupported model bundle schema {manifest['schema']!r}.")
    if manifest["schema_version"] != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ModelBundleError(
            "Unsupported model bundle schema version "
            f"{manifest['schema_version']!r}; expected "
            f"{MODEL_BUNDLE_SCHEMA_VERSION!r}."
        )
    if manifest["weights_format"] != "safetensors":
        raise ModelBundleError(
            "Runtime bundles require safetensors; migrate trusted legacy checkpoints first."
        )
    expected = str(manifest["weights_sha256"]).lower().removeprefix("sha256:")
    if not _SHA256.fullmatch(expected):
        raise ModelBundleError("weights_sha256 is not a SHA-256 digest.")
    weights_path = root / str(manifest["weights_file"])
    if weights_path.parent != root or not weights_path.is_file():
        raise ModelBundleError("weights_file must be a file in the bundle root.")
    actual = file_sha256(weights_path)
    if actual != expected:
        raise ModelBundleError(
            f"Weight checksum mismatch: expected {expected}, found {actual}."
        )
    encoded = manifest_path.read_bytes()
    checksums: dict[str, str] = {}
    try:
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not _SHA256.fullmatch(digest) or not name:
                raise ValueError(f"invalid checksum line {line!r}")
            if Path(name).name != name or name in checksums:
                raise ValueError(f"unsafe or duplicate checksum path {name!r}")
            checksums[name] = digest
    except (OSError, ValueError) as exc:
        raise ModelBundleError(f"Cannot validate SHA256SUMS: {exc}") from exc
    required_checksums = {
        "model.json": sha256(encoded).hexdigest(),
        weights_path.name: actual,
        "README.md": file_sha256(readme_path),
    }
    for name, digest in required_checksums.items():
        if checksums.get(name) != digest:
            raise ModelBundleError(f"SHA256SUMS does not authenticate {name}.")
    return ModelBundle(
        root=root,
        manifest=_freeze_json(manifest),
        manifest_sha256=sha256(encoded).hexdigest(),
        weights_path=weights_path,
        weights_sha256=actual,
    )


__all__ = [
    "MODEL_BUNDLE_SCHEMA",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "ModelBundle",
    "ModelBundleError",
    "file_sha256",
    "load_model_bundle",
]
