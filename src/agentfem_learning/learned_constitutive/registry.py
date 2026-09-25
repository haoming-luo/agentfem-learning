# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Third-party-extensible architecture registry."""

from __future__ import annotations

from collections.abc import Callable

_ARCHITECTURES: dict[str, Callable] = {}


class ArchitectureRegistryError(LookupError):
    """A requested learned-material architecture is not registered."""

    code = "AFM-LEARNING-ARCHITECTURE-001"

    def __init__(self, message: str):
        super().__init__(f"{self.code}: {message}")


def register_architecture(
    architecture_id: str,
    loader: Callable,
    *,
    replace: bool = False,
) -> None:
    selected = str(architecture_id).strip()
    if not selected or not callable(loader):
        raise TypeError("An architecture registration requires an id and callable loader.")
    if selected in _ARCHITECTURES and not replace:
        raise ValueError(f"Architecture {selected!r} is already registered.")
    _ARCHITECTURES[selected] = loader


def architecture_loader(architecture_id: str) -> Callable:
    selected = str(architecture_id).strip()
    try:
        return _ARCHITECTURES[selected]
    except KeyError as exc:
        raise ArchitectureRegistryError(
            f"No learned-constitutive architecture loader for {selected!r}; "
            f"available={registered_architectures()!r}."
        ) from exc


def registered_architectures() -> tuple[str, ...]:
    return tuple(sorted(_ARCHITECTURES))


__all__ = [
    "ArchitectureRegistryError",
    "architecture_loader",
    "register_architecture",
    "registered_architectures",
]
