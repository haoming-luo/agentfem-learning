# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""DENIM v1 reference architecture."""

from .architecture import DENIM, GrayboxState, NeuralHardeningLaw, advance
from .export import convert_legacy_checkpoint, denim_manifest
from .state_codec import initial_state, pack_state, state_schema, unpack_state

__all__ = [
    "DENIM",
    "GrayboxState",
    "NeuralHardeningLaw",
    "advance",
    "convert_legacy_checkpoint",
    "denim_manifest",
    "initial_state",
    "pack_state",
    "state_schema",
    "unpack_state",
]
