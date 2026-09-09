"""Official NeuralOperator FNO/TFNO/GINO bindings for AgentFEM field datasets."""

from .api import (
    NeuralOperatorOutcome,
    NeuralOperatorPredictor,
    NeuralOperatorTrainingOptions,
    load_predictor,
    train_operator,
)
from .checks import OperatorCheck, OperatorCheckContext
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
    "load_gino_predictor",
    "load_predictor",
    "train_gino",
    "train_operator",
]
