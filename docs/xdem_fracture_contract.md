# XDEM fracture-field contract

AgentFEM-Learning treats the Extended Deep Energy Method (XDEM) as a neural-
field provider, not as a second fracture-data model. AgentFEM core owns the
scientific assets shared with FEM, XFEM, cohesive, and phase-field consumers.

## Public ownership

Core owns:

- `fracture.segment(...)` and `fracture.crack_set(...)`;
- stable `crack_id` and `tip_id` values, orientation, and content fingerprint;
- `FractureField2D`, the future one-sided field-access protocol;
- ring-resolved `StressIntensityReport` records;
- `IntegrationPlan`, `IntegrationEvidence`, and result trust.

The XDEM provider owns:

- discontinuous and Williams-enriched PyTorch representations;
- automatic differentiation, numerical integration execution, and optimizers;
- device/dtype policy, checkpoints, and compatible warm starts;
- conversion of its trained field into the core result contract.

A research project owns particular crack distances, materials, interaction
coefficients, plots, and conclusions.

## Geometry boundary

The first executable geometry family will be mutually non-intersecting straight
cracks in a homogeneous two-dimensional linear-elastic domain. Curved,
intersecting, branching, boundary-terminating, growing, or three-dimensional
cracks are not silently approximated. They receive an addressable unsupported-
geometry error until another provider declares and verifies them.

Mode is not an input label. Loading, crack orientation, and material response
produce Mode I, Mode II, or mixed-mode results.

## Integration and trust

Every energy solve distinguishes:

1. the integration rule seen by the optimizer;
2. an explicitly independent held-out rule;
3. one or more refinement rules;
4. boundary, jump, traction, and applicable physics-balance checks.

Loss reduction is optimization evidence only. A large held-out gap, failure to
stabilize under refinement, or inconsistent physics balance keeps the result
uncertain even if training converged.

## Discontinuous output

Sampling must retain both one-sided values and an explicit side label. A
visualization writer may duplicate crack-face nodes or use separate face
datasets; it must not average the two traces into one continuous nodal field.

## Promotion sequence

1. analytical and FEM verification of an interaction/domain integral
   (implemented);
2. one Mode-I vector-elastic manufactured XDEM benchmark (implemented on the
   slit annulus; external finite-domain promotion remains);
3. one inclined mixed-mode geometry benchmark (the current mixed-mode
   manufactured field verifies physics and signs, not inclined meshing);
4. multiple separated straight cracks with stable per-tip reports;
5. Campaign, seed ensemble, checkpoint, warm start, and multi-axis convergence.

Crack growth, XDEM-C phase fields, fatigue, inverse identification, and neural-
operator generalization remain later capabilities and do not block this static
foundation.
