"""Explicit AgentFEM extension entry point."""

from agentfem import extensions

from .finite_domain_provider import XDEM_FINITE_DOMAIN_PROVIDER
from .provider import XDEM_REFERENCE_PROVIDER
from .vector_provider import XDEM_VECTOR_PROVIDER


def _register(context: extensions.ExtensionContext) -> None:
    context.add_step_provider(XDEM_REFERENCE_PROVIDER)
    context.add_step_provider(XDEM_VECTOR_PROVIDER)
    context.add_step_provider(XDEM_FINITE_DOMAIN_PROVIDER)


extension = extensions.Extension(
    spec=extensions.ExtensionSpec(
        name="agentfem-learning.xdem",
        version="0.1.0a1",
        description="Experimental XDEM neural-field provider in AgentFEM-Learning.",
        capabilities=(
            "learning.neural_field.energy",
            "fracture.williams_mode_iii_reference",
            "fracture.williams_vector_reference",
            "fracture.stress_intensity_extraction",
            "fracture.xdem_d.finite_domain",
            "fracture.xdem_d.multiple_cracks",
            "fracture.xdem_d.per_tip_sif",
            "results.simulation_result",
        ),
    ),
    register=_register,
)


__all__ = ["extension"]
