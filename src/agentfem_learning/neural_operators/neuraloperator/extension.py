"""Explicit NeuralOperator extension entry point."""

from agentfem import extensions

from agentfem_learning import __version__

from .gino_provider import GINO_PROVIDER
from .provider import NEURALOPERATOR_PROVIDER


def _register(context: extensions.ExtensionContext) -> None:
    context.add_step_provider(NEURALOPERATOR_PROVIDER)
    context.add_step_provider(GINO_PROVIDER)


extension = extensions.Extension(
    spec=extensions.ExtensionSpec(
        name="agentfem-learning.neuraloperator",
        version=__version__,
        description="Official FNO/TFNO/GINO integration for AgentFEM field datasets.",
        capabilities=(
            "learning.neural_operator.fno",
            "learning.neural_operator.tfno",
            "learning.neural_operator.gino",
            "learning.geometry_transfer_verification",
            "learning.field_dataset",
            "learning.held_out_field_verification",
            "learning.physics_check_evaluators",
            "learning.resolution_transfer_verification",
            "results.simulation_result",
        ),
    ),
    register=_register,
)


__all__ = ["extension"]
