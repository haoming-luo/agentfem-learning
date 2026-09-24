# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Lazy PyTorch runtime selection for learned materials."""

from __future__ import annotations

from dataclasses import dataclass


class TorchRuntimeUnavailable(RuntimeError):
    pass


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise TorchRuntimeUnavailable(
            "PyTorch is required for this provider. Install "
            "agentfem-learning[constitutive]."
        ) from exc
    return torch


@dataclass(frozen=True)
class TorchRuntime:
    device: str = "cpu"
    dtype: str = "float64"

    def resolve(self):
        torch = require_torch()
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("Torch constitutive dtype must be float32 or float64.")
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TorchRuntimeUnavailable("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise TorchRuntimeUnavailable("MPS was requested but is unavailable.")
        return torch, device, getattr(torch, self.dtype)

    def summary(self) -> dict[str, object]:
        torch, device, dtype = self.resolve()
        return {
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
        }


__all__ = ["TorchRuntime", "TorchRuntimeUnavailable", "require_torch"]
