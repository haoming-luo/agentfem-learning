# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven DENIM state encoding."""

from __future__ import annotations

import numpy as np
from agentfem.constitutive import MaterialStateSchema, MaterialStateVariable

from .architecture import GrayboxState


def state_schema(channels: int, *, version: str = "1.0.0") -> MaterialStateSchema:
    selected = int(channels)
    if selected <= 0:
        raise ValueError("DENIM memory channel count must be positive.")
    return MaterialStateSchema(
        name=f"denim_graybox_ch{selected}",
        version=version,
        variables=(
            MaterialStateVariable("plastic_strain", shape=(6,), description="Plastic strain in tensor-shear Voigt order."),
            MaterialStateVariable("peeq", description="Equivalent plastic strain."),
            MaterialStateVariable("memories", shape=(selected, 6), unit="Pa", description="Objective kinematic memory channels."),
            MaterialStateVariable("previous_flow", shape=(6,), description="Previous plastic flow direction."),
        ),
    )


def unpack_state(values, *, channels: int, torch, dtype, device) -> GrayboxState:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    schema = state_schema(channels)
    if array.ndim != 2 or array.shape[1] != schema.size or not np.all(np.isfinite(array)):
        raise ValueError(f"State must have shape (points, {schema.size}) and be finite.")
    tensor = torch.as_tensor(array, dtype=dtype, device=device)
    offset = 0
    plastic_strain = tensor[:, offset : offset + 6]
    offset += 6
    peeq = tensor[:, offset]
    offset += 1
    memories = tensor[:, offset : offset + channels * 6].reshape(-1, channels, 6)
    offset += channels * 6
    previous_flow = tensor[:, offset : offset + 6]
    return GrayboxState(plastic_strain, peeq, memories, previous_flow)


def pack_state(state: GrayboxState):
    torch = __import__("torch")
    return torch.cat(
        (
            state.plastic_strain,
            state.peeq[..., None],
            state.memories.reshape(state.memories.shape[0], -1),
            state.previous_flow,
        ),
        dim=-1,
    )


def initial_state(count: int, *, channels: int) -> np.ndarray:
    schema = state_schema(channels)
    return np.repeat(schema.initial_state()[None, :], int(count), axis=0)


__all__ = ["initial_state", "pack_state", "state_schema", "unpack_state"]
