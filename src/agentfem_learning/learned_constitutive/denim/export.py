# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""One-time migration from trusted DENIM checkpoints to safe local bundles."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from ..artifacts import (
    MODEL_BUNDLE_SCHEMA,
    MODEL_BUNDLE_SCHEMA_VERSION,
    file_sha256,
)
from ..torch_runtime import require_torch


def denim_manifest(
    *,
    weights_sha256: str,
    channels: int,
    model_name: str = "denim-expanded",
    model_version: str = "1.1.0",
    model_revision: str | None = None,
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
) -> dict[str, object]:
    return {
        "schema": MODEL_BUNDLE_SCHEMA,
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "model_revision": model_revision,
        "architecture_id": "denim.v1",
        "architecture": {
            "channels": int(channels),
            "hidden": 24,
            "isotropic_neurons": 12,
            "bisection_iterations": 28,
            "direction_iterations": 6,
        },
        "provider_compatibility": "agentfem-learning>=0.1.0a1",
        "kinematics": "small_strain",
        "stress_measure": "cauchy",
        "voigt_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "shear_convention": "tensor",
        "parameter_schema": [
            {"name": "young", "unit": "Pa", "minimum": 0.0},
            {"name": "poisson", "unit": "1", "minimum": -1.0, "maximum": 0.5},
            {"name": "yield_stress", "unit": "Pa", "minimum": 0.0},
        ],
        "state_schema_version": "1.0.0",
        "state_schema": {
            "variables": [
                {"name": "plastic_strain", "shape": [6]},
                {"name": "peeq", "shape": []},
                {"name": "memories", "shape": [int(channels), 6]},
                {"name": "previous_flow", "shape": [6]},
            ],
            "size": 13 + 6 * int(channels),
        },
        "required_inputs": ["strain_new", "state_old", "parameters"],
        "capabilities": {
            "stress": True,
            "state": True,
            "batch": True,
            "energy": True,
            "diagnostics": True,
            "tangent": "autodiff_consistent",
        },
        "dtype_policy": {"default": "float64", "allowed": ["float32", "float64"]},
        "applicability_domain": {
            "policy": "report_without_silent_fallback",
            "maximum_absolute_strain": 0.03,
            "maximum_peeq": 0.03,
        },
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "weights_file": "weights.safetensors",
        "weights_format": "safetensors",
        "weights_sha256": weights_sha256,
        "training_metadata": {
            "legacy_checkpoint_migrated": True,
            "parameter_count": None,
        },
    }


def convert_legacy_checkpoint(
    checkpoint,
    destination,
    *,
    channels: int,
    model_name: str = "denim-expanded",
    model_version: str = "1.1.0",
    model_revision: str | None = None,
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
) -> Path:
    """Migrate one trusted state-dict checkpoint; never used during a solve."""

    torch = require_torch()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required. Install agentfem-learning[constitutive]."
        ) from exc
    source = Path(checkpoint).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError("Legacy checkpoint does not contain a state_dict.")
    tensors = {
        str(name): value.detach().cpu().contiguous()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    if len(tensors) != len(state):
        raise ValueError("Legacy state_dict contains non-tensor values.")
    weights_path = target / "weights.safetensors"
    save_file(tensors, str(weights_path))
    digest = file_sha256(weights_path)
    manifest = denim_manifest(
        weights_sha256=digest,
        channels=channels,
        model_name=model_name,
        model_version=model_version,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
    )
    manifest["training_metadata"]["parameter_count"] = sum(
        value.numel() for value in tensors.values()
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (target / "model.json").write_text(encoded, encoding="utf-8")
    readme = (
        f"# {model_name}\n\n"
        "Local, offline AgentFEM-learning constitutive model bundle.\n\n"
        f"- Architecture: `denim.v1`\n"
        f"- Model version: `{model_version}`\n"
        f"- Model revision: `{model_revision or 'unspecified'}`\n"
        f"- Dataset: `{dataset_id or 'unspecified'}`\n"
        f"- Dataset revision: `{dataset_revision or 'unspecified'}`\n"
        f"- Memory channels: `{channels}`\n"
        "- Runtime weights: `weights.safetensors`\n\n"
        "The finite-element runtime verifies `model.json` and the weights "
        "checksum and never downloads artifacts during a solve.\n"
    )
    (target / "README.md").write_text(readme, encoding="utf-8")
    checksums = (
        f"{digest}  weights.safetensors\n"
        f"{sha256(encoded.encode('utf-8')).hexdigest()}  model.json\n"
        f"{sha256(readme.encode('utf-8')).hexdigest()}  README.md\n"
    )
    (target / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return target


__all__ = ["convert_legacy_checkpoint", "denim_manifest"]
