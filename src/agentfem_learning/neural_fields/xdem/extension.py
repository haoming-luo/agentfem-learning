"""Explicit AgentFEM extension entry point."""

from agentfem import extensions

from .provider import XDEM_REFERENCE_PROVIDER


def _register(context: extensions.ExtensionContext) -> None:
    context.add_step_provider(XDEM_REFERENCE_PROVIDER)


extension = extensions.Extension(
    spec=extensions.ExtensionSpec(
        name="agentfem-learning.xdem",
        version="0.1.0a1",
        description="Experimental XDEM neural-field provider in AgentFEM-Learning.",
        capabilities=(
            "learning.neural_field.energy",
            "fracture.williams_mode_iii_reference",
            "results.simulation_result",
        ),
    ),
    register=_register,
)


__all__ = ["extension"]
