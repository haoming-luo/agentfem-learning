# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""DENIM model-bundle loader registered with the generic provider."""

from __future__ import annotations

from .adapter import DenimMaterial
from .architecture import DENIM


def load_denim_v1(bundle, runtime):
    manifest = bundle.manifest
    options = dict(manifest.get("architecture", {}))
    channels = int(options.get("channels", 2))
    hidden = int(options.get("hidden", 24))
    neurons = int(options.get("isotropic_neurons", 12))
    law = DENIM(channels=channels, hidden=hidden, neurons=neurons)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required. Install agentfem-learning[constitutive]."
        ) from exc
    _, device, _ = runtime.resolve()
    state = load_file(str(bundle.weights_path), device=str(device))
    missing, unexpected = law.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DENIM weight keys are incompatible: missing={missing}, unexpected={unexpected}."
        )
    tangent = str(dict(manifest["capabilities"]).get("tangent", "none"))
    return DenimMaterial(
        bundle=bundle,
        law=law,
        runtime=runtime,
        channels=channels,
        tangent_mode=tangent,
        name=str(manifest["model_name"]),
    )


__all__ = ["load_denim_v1"]
