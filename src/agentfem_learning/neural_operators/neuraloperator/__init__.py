"""Official NeuralOperator FNO/TFNO/GINO bindings for AgentFEM field datasets."""

from .api import (
    NeuralOperatorOutcome,
    NeuralOperatorPredictor,
    NeuralOperatorTrainingOptions,
    load_predictor,
    train_operator,
)
from .checks import (
    OperatorCheck,
    OperatorCheckContext,
    ParameterPathRefinementPlan,
    parameter_path_refinement_plan,
    parameter_path_reliability_check,
)
from .gino import (
    GINOOutcome,
    GINOPredictor,
    GINOTrainingOptions,
    load_gino_predictor,
    train_gino,
)
from .gino_provider import GINO_PROVIDER, GINOStep
from .provider import NEURALOPERATOR_PROVIDER, NeuralOperatorStep

__all__ = [
    "GINO_PROVIDER",
    "NEURALOPERATOR_PROVIDER",
    "GINOOutcome",
    "GINOPredictor",
    "GINOStep",
    "GINOTrainingOptions",
    "NeuralOperatorOutcome",
    "NeuralOperatorPredictor",
    "NeuralOperatorStep",
    "NeuralOperatorTrainingOptions",
    "OperatorCheck",
    "OperatorCheckContext",
    "ParameterPathRefinementPlan",
    "load_gino_predictor",
    "load_predictor",
    "parameter_path_refinement_plan",
    "parameter_path_reliability_check",
    "train_gino",
    "train_operator",
]
