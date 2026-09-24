# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional learned-constitutive runtimes for AgentFEM."""

from .artifacts import (
    MODEL_BUNDLE_SCHEMA,
    MODEL_BUNDLE_SCHEMA_VERSION,
    ModelBundle,
    ModelBundleError,
    load_model_bundle,
)
from .provider import TORCH_CONSTITUTIVE_PROVIDER, TorchConstitutiveProvider
from .registry import (
    ArchitectureRegistryError,
    register_architecture,
    registered_architectures,
)

__all__ = [
    "MODEL_BUNDLE_SCHEMA",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "TORCH_CONSTITUTIVE_PROVIDER",
    "ArchitectureRegistryError",
    "ModelBundle",
    "ModelBundleError",
    "TorchConstitutiveProvider",
    "load_model_bundle",
    "register_architecture",
    "registered_architectures",
]
