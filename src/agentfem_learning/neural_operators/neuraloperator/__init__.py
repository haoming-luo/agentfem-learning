"""Official NeuralOperator FNO/TFNO binding for AgentFEM field datasets."""

from .api import (
    NeuralOperatorOutcome,
    NeuralOperatorPredictor,
    NeuralOperatorTrainingOptions,
    load_predictor,
    train_operator,
)
from .checks import OperatorCheck, OperatorCheckContext
from .provider import NEURALOPERATOR_PROVIDER, NeuralOperatorStep

__all__ = [
    "NEURALOPERATOR_PROVIDER",
    "NeuralOperatorOutcome",
    "NeuralOperatorPredictor",
    "NeuralOperatorStep",
    "NeuralOperatorTrainingOptions",
    "OperatorCheck",
    "OperatorCheckContext",
    "load_predictor",
    "train_operator",
]
