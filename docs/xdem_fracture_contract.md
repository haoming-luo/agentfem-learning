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

The first executable geometry family is mutually non-intersecting straight
cracks in a homogeneous two-dimensional linear-elastic domain. Curved,
intersecting, branching, boundary-terminating, growing, or three-dimensional
cracks are not silently approximated. They receive an addressable unsupported-
geometry error until another provider declares and verifies them.

Mode is not an input label. Loading, crack orientation, and material response
produce Mode I, Mode II, or mixed-mode results.

Traction-driven reference problems use point displacement gauges to remove the
three planar rigid-body modes. A gauge is not a concentrated load and does not
replace a physical support. This avoids clamping a complete remote boundary
merely to make the energy minimization unique.

## Integration and trust

Every energy solve distinguishes:

1. the integration rule seen by the optimizer;
2. an explicitly independent held-out rule;
3. one or more refinement rules;
4. boundary, jump, traction, and applicable physics-balance checks.

The executable provider uses deterministic, topology-aware cut-cell rules.
Ordinary cells use midpoint integration. Cells crossed by a straight crack are
clipped into two one-sided polygons and integrated by a positive second-order
triangle rule. A narrow neighboring band and grid-aligned crack cells use a
2-by-2 tensor rule; cells touching a crack tip use a 4-by-4 rule without
silently extending the crack through the cell.
Every point carries one side code per crack, and quadrature weights sum to the
physical domain area. Training, held-out validation, and refinement use
different deterministic grid variants and retain distinct fingerprints. If a
coarse cell intersects more than one crack, the provider fails and requests
integration refinement instead of guessing the topology. A crack-tip-
stratified Monte Carlo rule remains an implementation experiment: fixed random
clouds were observed to admit low training energy with a materially different
held-out energy.

The current default approximation is explicitly additive: a continuous mean field, one
two-sided jump network per crack, and one trainable Williams field per active
tip. A fully internal straight crack uses an LEFM-compatible elliptical jump
closure at its two active tips; a boundary crack retains a separate mouth/tip
contract. Square-root features are not also fed through the mean network. Complete
point gauges are imposed by an exact zero-strain rigid-motion projection rather
than a local penalty. Load-derived nominal SIFs may initialize the Williams
coefficients as a numerical preconditioner, but published target values never
enter training and acceptance still uses independently extracted SIFs.

The published `NN(x, rho)` crack-coordinate architecture and an
AgentFEM-Learning bounded-sheet variant are retained as internal, tested
representation candidates. Neither is the default. In the Griffith diagnostic,
the published distance-decay coordinate produced a large crack-face stress
layer, while smoothing or removing that decay did not improve the independent
SIF checks. These negative results prevent a literature-derived mechanism from
being promoted merely because it trains.

Spatial essential data are serialized as a named field family rather than a
live callback. For a complete rectangular boundary, the trial field is
constructed as `u = u_lift + D u_free`, where `D` vanishes on every prescribed
edge. The default lift is transfinite interpolation of boundary traces. A
declared analytic interior lift is permitted only for an explicitly labelled
patch test; it is recorded in result metadata and is not counted as predictive
validation.

A boundary crack mouth and an active crack tip are different entities. Every
straight crack retains both stable endpoint IDs, while `active_ends` selects
only interior endpoints that receive Williams enrichment and SIF/J integration
domains. A boundary endpoint marked active fails before training.

Loss reduction is optimization evidence only. A large held-out gap, failure to
stabilize under refinement, or inconsistent physics balance keeps the result
uncertain even if training converged.

Every finite-domain result now carries two independent SIF estimates: the
domain interaction integral and a zero-radius extrapolation of two-sided crack
opening displacement. The latter fits the sampled values against `sqrt(r)` and
reports its own residual. Their disagreement is an explicit verification
claim; agreement does not prove correctness, but disagreement prevents a
single extractor from certifying an inconsistent field.

The published-reference gate additionally requires crack-face traction and
bulk-equilibrium residuals below their declared limits. The latest
traction-driven Griffith diagnostic additionally resolves grid-aligned crack
faces and tips. Under the fixed training budget it raised normalized `K_I` to
about 0.69, reduced maximum path variation to about 8.6%, and reduced extractor
disagreement to about 38%. Crack-face traction and bulk equilibrium still
failed their declared limits, so this remains a sharper formulation diagnosis,
not a promotion result. Channel-resolved evidence identifies the continuous
mean field, rather than the Williams term, as the dominant source of remaining
face traction. The public hard-boundary Mode-I/II extended patch passes with
one active tip and is classified as patch evidence. A traction-free
discontinuous-trial-space correction remains the next predictive gate. A
crack-face penalty was tested and rejected as a default because it reduced the
residual by suppressing the physical crack response. Griffith domain-size
convergence and the interacting four-tip case follow only after this gate.

## Discontinuous output

Sampling must retain both one-sided values and an explicit side label. A
visualization writer may duplicate crack-face nodes or use separate face
datasets; it must not average the two traces into one continuous nodal field.

The finite-domain provider writes one unstructured ParaView dataset. Its bulk
sample grid and both coincident crack-face line sets share one file, while
`CrackSide` and `CrackId` preserve the double-valued trace semantics. The
scientific NPZ remains the lossless source; the VTU is the inspection product.

## Convergence gate

For a two-crack problem, convergence is not one loss curve. The promotion
record retains four stable tip identities and audits network architecture,
training and independent integration density, random seed, and SIF/J
integration-ring resolution. Each axis compares its two finest declared
levels, applies the same per-tip contract to `K_I`, `K_II`, and `J`, and
separately limits path variation. Missing or renamed tips make the report
uncertain. Heavy network objects and sample arrays are released after every
case so evidence storage remains bounded.

When a problem factory embeds a published reference, the provider writes a
`published_benchmark.json` artifact and attaches the comparison to
`SimulationResult`. A failed reference remains a first-class failed claim; it
is never converted into an accepted result merely because training completed.

## Promotion sequence

1. analytical and FEM verification of an interaction/domain integral
   (implemented);
2. one Mode-I vector-elastic manufactured XDEM benchmark (implemented on the
   slit annulus) and one public mixed-mode finite-domain extended patch test
   (implemented and accepted, but not predictive validation);
3. one inclined mixed-mode geometry benchmark (implemented for arbitrary tip
   translation and orientation on the manufactured slit annulus; external
   finite-domain geometry remains a promotion gate);
4. multiple separated straight cracks with stable per-tip reports
   (implemented as an experimental joint solver; external interacting-crack
   validation remains);
5. bounded-memory multi-axis convergence and discontinuous ParaView output
   (implemented as infrastructure; numerical promotion campaign remains);
6. Campaign provenance, checkpoint/best state, and explicit warm start with a
   cold-start fallback.

Crack growth, XDEM-C phase fields, fatigue, inverse identification, and neural-
operator generalization remain later capabilities and do not block this static
foundation.
