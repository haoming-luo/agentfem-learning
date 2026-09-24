# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit AgentFEM extension for learned constitutive runtimes."""

from agentfem import extensions

from agentfem_learning import __version__

from .denim.loader import load_denim_v1
from .provider import TORCH_CONSTITUTIVE_PROVIDER
from .registry import register_architecture

register_architecture("denim.v1", load_denim_v1, replace=True)


def _register(context: extensions.ExtensionContext) -> None:
    context.add_learned_constitutive_provider(TORCH_CONSTITUTIVE_PROVIDER)


extension = extensions.Extension(
    spec=extensions.ExtensionSpec(
        name="agentfem-learning.learned-constitutive",
        version=__version__,
        description="Offline, verified PyTorch constitutive model provider.",
        capabilities=(
            "learning.constitutive.small_strain",
            "learning.constitutive.batch",
            "learning.constitutive.autodiff_tangent",
            "learning.constitutive.denim_v1",
        ),
    ),
    register=_register,
)


__all__ = ["extension"]
