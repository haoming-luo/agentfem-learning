"""Readable AgentFEM neural-field specifications used by the XDEM provider."""

from __future__ import annotations

from agentfem import learning


def mode_iii_tip_spec(
    *,
    radius: float = 1.0,
    tip_core_radius: float = 0.05,
    shear_modulus: float = 1.0,
    boundary_displacement: float = 1.0,
    domain_samples: int = 2048,
    boundary_samples: int = 128,
) -> learning.NeuralFieldSpec:
    """Define a normalized Williams Mode-III crack-tip reference problem.

    The slit annulus has a branch cut on the negative x-axis. Its crack faces
    are traction-free natural boundaries. Exact Williams displacements are
    imposed on the inner and outer circular boundaries, providing an
    independently evaluable manufactured fracture field without claiming
    crack growth or general XDEM coverage.
    """

    radius = float(radius)
    tip_core_radius = float(tip_core_radius)
    shear_modulus = float(shear_modulus)
    boundary_displacement = float(boundary_displacement)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if not 0.0 < tip_core_radius < radius:
        raise ValueError("tip_core_radius must lie strictly between zero and radius.")
    if shear_modulus <= 0.0:
        raise ValueError("shear_modulus must be positive.")
    if boundary_displacement == 0.0:
        raise ValueError("boundary_displacement must be nonzero.")

    displacement = learning.FieldEncoding(
        name="anti_plane_displacement",
        role="output",
        unit="m",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
        components=("W",),
    )
    return learning.NeuralFieldSpec(
        fields=(displacement,),
        objectives=(
            learning.ObjectiveTerm(
                name="anti_plane_strain_energy",
                kind="energy",
                expression="integral(mu/2 * grad(W) dot grad(W), slit_annulus)",
                dependent_fields=(displacement.name,),
                form="variational",
                measure="domain",
                unit="J/m",
                implementation="agentfem_learning.neural_fields.xdem:mode_iii_energy",
            ),
        ),
        conditions=(
            learning.ConditionSpec(
                name="outer_williams_displacement",
                kind="boundary",
                target=displacement.name,
                on="outer_circle",
                value="agentfem_learning.neural_fields.xdem:williams_mode_iii",
                enforcement="penalty",
                implementation="agentfem_learning.neural_fields.xdem:williams_mode_iii",
            ),
            learning.ConditionSpec(
                name="tip_core_williams_displacement",
                kind="boundary",
                target=displacement.name,
                on="tip_core_circle",
                value="agentfem_learning.neural_fields.xdem:williams_mode_iii",
                enforcement="penalty",
                implementation="agentfem_learning.neural_fields.xdem:williams_mode_iii",
            ),
        ),
        representations=(
            learning.NeuralRepresentation(
                name="williams_enriched_displacement",
                fields=(displacement.name,),
                architecture="xdem:williams_mlp",
                features=(
                    "coordinates",
                    "radial_coordinate",
                    "xdem:williams_branch_cut",
                ),
                enrichments=("xdem:williams_mode_iii_tip",),
                implementation=(
                    "agentfem_learning.neural_fields.xdem:WilliamsModeIIINetwork"
                ),
            ),
        ),
        sampling=(
            learning.SamplingPlan(
                name="slit_annulus_energy_points",
                on="domain",
                strategy="xdem:uniform_area",
                count=int(domain_samples),
                seed=2026,
            ),
            learning.SamplingPlan(
                name="circular_boundary_points",
                on="boundary",
                strategy="uniform",
                count=int(boundary_samples),
                seed=2027,
            ),
        ),
        purpose="forward",
        required_checks=(
            "independent_reference_error",
            "condition_error",
            "energy_error",
            "crack_jump_error",
            "crack_face_traction_error",
            "optimization_repeatability",
        ),
        metadata={
            "provider": "agentfem-learning.xdem",
            "problem": "williams_mode_iii_tip",
            "geometry": {
                "kind": "slit_annulus",
                "radius": radius,
                "tip_core_radius": tip_core_radius,
                "crack": "negative_x_axis_branch_cut",
            },
            "material": {"shear_modulus": shear_modulus},
            "loading": {"boundary_displacement": boundary_displacement},
            "maturity": "experimental_reference",
        },
    )


__all__ = ["mode_iii_tip_spec"]
