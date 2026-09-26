# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""DENIM v1 reference architecture."""

from .acceptance import (
    AcceptanceCheck,
    AcceptanceReport,
    evaluate_global_bar,
    evaluate_material_point,
    load_denim_v1_acceptance,
    verify_published_evidence,
)
from .architecture import DENIM, GrayboxState, NeuralHardeningLaw, advance
from .export import convert_legacy_checkpoint, denim_manifest
from .state_codec import initial_state, pack_state, state_schema, unpack_state

__all__ = [
    "DENIM",
    "AcceptanceCheck",
    "AcceptanceReport",
    "GrayboxState",
    "NeuralHardeningLaw",
    "advance",
    "convert_legacy_checkpoint",
    "denim_manifest",
    "evaluate_global_bar",
    "evaluate_material_point",
    "initial_state",
    "load_denim_v1_acceptance",
    "pack_state",
    "state_schema",
    "unpack_state",
    "verify_published_evidence",
]
