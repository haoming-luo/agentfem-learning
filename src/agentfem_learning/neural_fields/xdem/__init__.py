"""XDEM-style neural-field computation for AgentFEM-Learning.

This provider owns PyTorch training and fracture-specific neural
representations. AgentFEM core remains responsible for scientific contracts,
the Step lifecycle, result evidence, and extension discovery.
"""

from ... import __version__
from .benchmarks import (
    PublishedSIFReference2D,
    center_crack_domain_problem,
    griffith_center_crack_reference,
    two_collinear_cracks_domain_problem,
    two_collinear_cracks_reference,
    xvem_mixed_mode_domain_problem,
    xvem_mixed_mode_reference,
)
from .convergence import (
    XDEMConvergenceCase,
    XDEMMultiAxisConvergenceReport,
    convergence_case,
    run_convergence_slices,
    run_finite_domain_convergence,
)
from .finite_domain import (
    PointDisplacementCondition2D,
    RectangularDomain2D,
    SpatialDisplacementCondition2D,
    SpatialVectorField2D,
    StaticXDEMProblem2D,
    UnsupportedFiniteDomainError,
    VectorBoundaryCondition2D,
    displacement_bc,
    finite_domain_spec,
    point_displacement,
    rectangular_domain,
    spatial_displacement_bc,
    static_crack_problem,
    traction_bc,
    williams_displacement_field,
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
    CrackOpeningSIFReport2D,
    MultiTipStressIntensityReport2D,
    TipIntegrationPlan2D,
    crack_opening_sif_reports,
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
    "CrackOpeningSIFReport2D",
    "FiniteDomainTrainingOutcome",
    "FiniteDomainVectorNetwork",
    "MultiTipStressIntensityReport2D",
    "PointDisplacementCondition2D",
    "PublishedSIFReference2D",
    "RectangularDomain2D",
    "ReferenceTrainingOptions",
    "SpatialDisplacementCondition2D",
    "SpatialVectorField2D",
    "StaticXDEMProblem2D",
    "TipIntegrationPlan2D",
    "TorchVectorFractureField",
    "UnsupportedFiniteDomainError",
    "VectorBoundaryCondition2D",
    "WilliamsModeIIINetwork",
    "WilliamsVectorNetwork",
    "XDEMConvergenceCase",
    "XDEMFiniteDomainStep",
    "XDEMMultiAxisConvergenceReport",
    "XDEMReferenceStep",
    "XDEMVectorStep",
    "__version__",
    "center_crack_domain_problem",
    "convergence_case",
    "crack_opening_sif_reports",
    "displacement_bc",
    "finite_domain_spec",
    "griffith_center_crack_reference",
    "mode_i_tip_spec",
    "mode_iii_tip_spec",
    "point_displacement",
    "rectangular_domain",
    "run_convergence_slices",
    "run_finite_domain_convergence",
    "spatial_displacement_bc",
    "static_crack_problem",
    "stress_intensity_reports",
    "tip_integration_plan",
    "traction_bc",
    "train_finite_domain",
    "two_collinear_cracks_domain_problem",
    "two_collinear_cracks_reference",
    "vector_tip_spec",
    "williams_displacement_field",
    "xvem_mixed_mode_domain_problem",
    "xvem_mixed_mode_reference",
]
