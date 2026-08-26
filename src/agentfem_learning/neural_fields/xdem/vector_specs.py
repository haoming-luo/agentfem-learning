"""Vector-elastic Williams reference specifications for the XDEM provider."""

from __future__ import annotations

from math import cos, isfinite, sin

from agentfem import fracture, learning


def vector_tip_spec(
    *,
    radius: float = 1.0,
    tip_core_radius: float = 0.05,
    young_modulus: float = 1000.0,
    poisson_ratio: float = 0.25,
    assumption: str = "plane_strain",
    k_i: float = 1.0,
    k_ii: float = 0.0,
    tip: tuple[float, float] = (0.0, 0.0),
    crack_angle: float = 0.0,
    domain_samples: int = 2048,
    boundary_samples: int = 128,
) -> learning.NeuralFieldSpec:
    """Declare one oriented stationary mixed-mode Williams field.

    ``crack_angle`` is the crack-extension direction measured counterclockwise
    from the global x-axis.  Stress-intensity factors retain their conventional
    meaning in this local crack-tip frame while fields are returned in global
    coordinates.
    """

    radius = float(radius)
    core = float(tip_core_radius)
    material = fracture.linear_elastic_fracture_material(
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        assumption=assumption,
    )
    mode_i = float(k_i)
    mode_ii = float(k_ii)
    tip_point = tuple(float(item) for item in tip)
    angle = float(crack_angle)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if not 0.0 < core < radius:
        raise ValueError("tip_core_radius must lie strictly between zero and radius.")
    if core >= 0.2 * radius:
        raise ValueError(
            "The vector reference requires tip_core_radius < 0.2 * radius "
            "so three independent SIF domains remain available."
        )
    if not isfinite(mode_i) or not isfinite(mode_ii):
        raise ValueError("Reference stress-intensity factors must be finite.")
    if mode_i == 0.0 and mode_ii == 0.0:
        raise ValueError("At least one reference stress-intensity factor must be nonzero.")
    if len(tip_point) != 2 or any(not isfinite(item) for item in tip_point):
        raise ValueError("tip must contain two finite coordinates.")
    if not isfinite(angle):
        raise ValueError("crack_angle must be finite.")

    tangent = (cos(angle), sin(angle))
    crack_start = (
        tip_point[0] - radius * tangent[0],
        tip_point[1] - radius * tangent[1],
    )

    cracks = fracture.crack_set(
        fracture.segment(
            "branch_cut",
            start=crack_start,
            end=tip_point,
            metadata={"role": "traction_free_branch_cut"},
        ),
        name="vector_williams_reference_crack",
    )
    integration = learning.IntegrationPlan(
        training=learning.IntegrationRule(
            name="vector_slit_annulus_energy_points",
            role="training",
            strategy="xdem:tensor_midpoint",
            count=int(domain_samples),
            seed=2026,
            implementation="agentfem_learning.neural_fields.xdem:annulus_midpoints",
        ),
        validation=learning.IntegrationRule(
            name="vector_slit_annulus_validation_points",
            role="validation",
            strategy="xdem:tensor_midpoint",
            count=48 * 96,
            seed=3026,
            independent_of=("vector_slit_annulus_energy_points",),
            implementation="agentfem_learning.neural_fields.xdem:annulus_midpoints",
        ),
        refinements=(
            learning.IntegrationRule(
                name="vector_slit_annulus_refined_points",
                role="refinement",
                strategy="xdem:tensor_midpoint",
                count=72 * 144,
                seed=4026,
                independent_of=(
                    "vector_slit_annulus_energy_points",
                    "vector_slit_annulus_validation_points",
                ),
                implementation="agentfem_learning.neural_fields.xdem:annulus_midpoints",
            ),
        ),
        metadata={
            "purpose": "independent_vector_energy_reintegration",
            "coordinate_system": "crack_tip_cartesian",
        },
    )
    displacement = learning.FieldEncoding(
        name="displacement",
        role="output",
        unit="m",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
        components=("U1", "U2"),
    )
    return learning.NeuralFieldSpec(
        fields=(displacement,),
        objectives=(
            learning.ObjectiveTerm(
                name="linear_elastic_strain_energy",
                kind="energy",
                expression=(
                    "integral(mu * epsilon:epsilon + lambda_eff/2 * "
                    "tr(epsilon)^2, slit_annulus)"
                ),
                dependent_fields=(displacement.name,),
                form="variational",
                measure="domain",
                unit="J/m",
                implementation=(
                    "agentfem_learning.neural_fields.xdem:vector_elastic_energy"
                ),
            ),
        ),
        conditions=(
            learning.ConditionSpec(
                name="outer_williams_displacement",
                kind="boundary",
                target=displacement.name,
                on="outer_circle",
                value="agentfem_learning.neural_fields.xdem:williams_vector_field",
                enforcement="penalty",
                implementation=(
                    "agentfem_learning.neural_fields.xdem:williams_vector_field"
                ),
            ),
            learning.ConditionSpec(
                name="tip_core_williams_displacement",
                kind="boundary",
                target=displacement.name,
                on="tip_core_circle",
                value="agentfem_learning.neural_fields.xdem:williams_vector_field",
                enforcement="penalty",
                implementation=(
                    "agentfem_learning.neural_fields.xdem:williams_vector_field"
                ),
            ),
        ),
        representations=(
            learning.NeuralRepresentation(
                name="mixed_mode_williams_enriched_displacement",
                fields=(displacement.name,),
                architecture="xdem:vector_williams_mlp",
                features=(
                    "coordinates",
                    "radial_coordinate",
                    "half_angle_functions",
                ),
                enrichments=(
                    "xdem:williams_mode_i_tip",
                    "xdem:williams_mode_ii_tip",
                ),
                implementation=(
                    "agentfem_learning.neural_fields.xdem:WilliamsVectorNetwork"
                ),
            ),
        ),
        sampling=(
            learning.SamplingPlan(
                name="vector_slit_annulus_energy_points",
                on="domain",
                strategy="xdem:uniform_area",
                count=int(domain_samples),
                seed=2026,
            ),
            learning.SamplingPlan(
                name="vector_circular_boundary_points",
                on="boundary",
                strategy="uniform",
                count=int(boundary_samples),
                seed=2027,
            ),
        ),
        integration=integration,
        purpose="forward",
        required_checks=(
            "independent_reference_error",
            "condition_error",
            "energy_error",
            "crack_jump_error",
            "stress_intensity_error",
            "stress_intensity_path_variation",
            "optimization_repeatability",
        ),
        metadata={
            "provider": "agentfem-learning.xdem",
            "problem": "williams_vector_tip",
            "geometry": {
                "kind": "slit_annulus",
                "radius": radius,
                "tip_core_radius": core,
                "tip": tip_point,
                "crack_angle": angle,
                "coordinate_system": "global_cartesian_with_local_crack_tip_frame",
                "cracks": cracks.summary(),
                "crack_fingerprint": cracks.fingerprint,
            },
            "material": material.summary(),
            "loading": {"K_I": mode_i, "K_II": mode_ii},
            "maturity": "experimental_reference",
        },
    )


def mode_i_tip_spec(**options) -> learning.NeuralFieldSpec:
    """Convenience constructor for the vector Mode-I reference."""

    if "k_ii" in options and float(options["k_ii"]) != 0.0:
        raise ValueError("mode_i_tip_spec fixes k_ii=0; use vector_tip_spec for mixed mode.")
    options["k_ii"] = 0.0
    return vector_tip_spec(**options)


__all__ = ["mode_i_tip_spec", "vector_tip_spec"]
