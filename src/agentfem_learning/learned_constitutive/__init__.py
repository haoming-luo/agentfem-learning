# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional learned-constitutive runtimes for AgentFEM."""

from .artifacts import ModelBundle, ModelBundleError, load_model_bundle
from .provider import TORCH_CONSTITUTIVE_PROVIDER, TorchConstitutiveProvider
from .registry import register_architecture, registered_architectures

__all__ = [
    "TORCH_CONSTITUTIVE_PROVIDER",
    "ModelBundle",
    "ModelBundleError",
    "TorchConstitutiveProvider",
    "load_model_bundle",
    "register_architecture",
    "registered_architectures",
]
