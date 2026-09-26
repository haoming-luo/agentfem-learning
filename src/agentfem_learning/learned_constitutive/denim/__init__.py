# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""DENIM v1 reference architecture."""

from .acceptance import (
    AcceptanceCheck,
    AcceptanceReport,
    evaluate_global_bar,
    evaluate_material_point,
    evaluate_nonproportional_path,
    evaluate_structural_convergence,
    load_denim_v1_acceptance,
    verify_published_evidence,
)
from .architecture import DENIM, GrayboxState, NeuralHardeningLaw, advance
from .export import convert_legacy_checkpoint, denim_manifest
from .nonproportional import (
    PATH_IDENTITY,
    PATH_REFERENCE,
    rotate_path,
    run_nonproportional_validation,
    tension_torsion_circle,
)
from .state_codec import initial_state, pack_state, state_schema, unpack_state

__all__ = [
    "DENIM",
    "PATH_IDENTITY",
    "PATH_REFERENCE",
    "AcceptanceCheck",
    "AcceptanceReport",
    "GrayboxState",
    "NeuralHardeningLaw",
    "advance",
    "convert_legacy_checkpoint",
    "denim_manifest",
    "evaluate_global_bar",
    "evaluate_material_point",
    "evaluate_nonproportional_path",
    "evaluate_structural_convergence",
    "initial_state",
    "load_denim_v1_acceptance",
    "pack_state",
    "rotate_path",
    "run_nonproportional_validation",
    "state_schema",
    "tension_torsion_circle",
    "unpack_state",
    "verify_published_evidence",
]
