"""XDEM-style neural-field computation for AgentFEM-Learning.

This provider owns PyTorch training and fracture-specific neural
representations. AgentFEM core remains responsible for scientific contracts,
the Step lifecycle, result evidence, and extension discovery.
"""

from ... import __version__
from .provider import XDEM_REFERENCE_PROVIDER, XDEMReferenceStep
from .reference import ReferenceTrainingOptions, WilliamsModeIIINetwork
from .specs import mode_iii_tip_spec
from .vector_provider import XDEM_VECTOR_PROVIDER, XDEMVectorStep
from .vector_reference import WilliamsVectorNetwork
from .vector_specs import mode_i_tip_spec, vector_tip_spec

__all__ = [
    "XDEM_REFERENCE_PROVIDER",
    "XDEM_VECTOR_PROVIDER",
    "ReferenceTrainingOptions",
    "WilliamsModeIIINetwork",
    "WilliamsVectorNetwork",
    "XDEMReferenceStep",
    "XDEMVectorStep",
    "__version__",
    "mode_i_tip_spec",
    "mode_iii_tip_spec",
    "vector_tip_spec",
]
