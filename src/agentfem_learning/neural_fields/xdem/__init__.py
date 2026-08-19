"""XDEM-style neural-field computation for AgentFEM-Learning.

This provider owns PyTorch training and fracture-specific neural
representations. AgentFEM core remains responsible for scientific contracts,
the Step lifecycle, result evidence, and extension discovery.
"""

from ... import __version__
from .provider import XDEM_REFERENCE_PROVIDER, XDEMReferenceStep
from .reference import ReferenceTrainingOptions, WilliamsModeIIINetwork
from .specs import mode_iii_tip_spec

__all__ = [
    "XDEM_REFERENCE_PROVIDER",
    "ReferenceTrainingOptions",
    "WilliamsModeIIINetwork",
    "XDEMReferenceStep",
    "__version__",
    "mode_iii_tip_spec",
]
