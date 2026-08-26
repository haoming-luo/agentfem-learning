"""XDEM-style neural-field computation for AgentFEM-Learning.

This provider owns PyTorch training and fracture-specific neural
representations. AgentFEM core remains responsible for scientific contracts,
the Step lifecycle, result evidence, and extension discovery.
"""

from ... import __version__
from .finite_domain import (
    RectangularDomain2D,
    StaticXDEMProblem2D,
    UnsupportedFiniteDomainError,
    VectorBoundaryCondition2D,
    displacement_bc,
    finite_domain_spec,
    rectangular_domain,
    static_crack_problem,
    traction_bc,
)
from .finite_domain_provider import (
    XDEM_FINITE_DOMAIN_PROVIDER,
    XDEMFiniteDomainStep,
)
from .finite_domain_solver import (
    FiniteDomainTrainingOutcome,
    FiniteDomainVectorNetwork,
    train_finite_domain,
)
from .provider import XDEM_REFERENCE_PROVIDER, XDEMReferenceStep
from .reference import ReferenceTrainingOptions, WilliamsModeIIINetwork
from .specs import mode_iii_tip_spec
from .tip_reports import (
    MultiTipStressIntensityReport2D,
    TipIntegrationPlan2D,
    stress_intensity_reports,
    tip_integration_plan,
)
from .vector_provider import XDEM_VECTOR_PROVIDER, XDEMVectorStep
from .vector_reference import TorchVectorFractureField, WilliamsVectorNetwork
from .vector_specs import mode_i_tip_spec, vector_tip_spec

__all__ = [
    "XDEM_FINITE_DOMAIN_PROVIDER",
    "XDEM_REFERENCE_PROVIDER",
    "XDEM_VECTOR_PROVIDER",
    "FiniteDomainTrainingOutcome",
    "FiniteDomainVectorNetwork",
    "MultiTipStressIntensityReport2D",
    "RectangularDomain2D",
    "ReferenceTrainingOptions",
    "StaticXDEMProblem2D",
    "TipIntegrationPlan2D",
    "TorchVectorFractureField",
    "UnsupportedFiniteDomainError",
    "VectorBoundaryCondition2D",
    "WilliamsModeIIINetwork",
    "WilliamsVectorNetwork",
    "XDEMFiniteDomainStep",
    "XDEMReferenceStep",
    "XDEMVectorStep",
    "__version__",
    "displacement_bc",
    "finite_domain_spec",
    "mode_i_tip_spec",
    "mode_iii_tip_spec",
    "rectangular_domain",
    "static_crack_problem",
    "stress_intensity_reports",
    "tip_integration_plan",
    "traction_bc",
    "train_finite_domain",
    "vector_tip_spec",
]
