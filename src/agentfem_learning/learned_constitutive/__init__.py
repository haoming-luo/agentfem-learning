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
from .calibration import (
    CalibrationCheck,
    CalibrationEvidenceReport,
    evaluate_calibration_evidence,
)
from .history import LearnedMaterialHistory, run_small_strain_material_history
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
    "CalibrationCheck",
    "CalibrationEvidenceReport",
    "LearnedMaterialHistory",
    "ModelBundle",
    "ModelBundleError",
    "TorchConstitutiveProvider",
    "evaluate_calibration_evidence",
    "load_model_bundle",
    "register_architecture",
    "registered_architectures",
    "run_small_strain_material_history",
]
