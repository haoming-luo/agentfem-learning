# AgentFEM-Learning roadmap

AgentFEM-Learning is an optional companion, not the only route from AgentFEM
to machine learning. A user-owned model must remain executable through the
core `model.step(target=spec, executor=...)` boundary.

## Release foundation

1. Keep the Williams Mode-III XDEM case as the provider, packaging, artifact,
   and analytical-evidence regression.
2. Test the wheel against released AgentFEM and current AgentFEM main on Python
   3.11 and 3.12 before publishing.
3. Keep framework dependencies in method extras. Installing the distribution
   alone must not install PyTorch, JAX, DeepXDE, or a neural-operator stack.
4. Record provider and framework versions, architecture configuration,
   scientific specification, state artifact, optimization history, and
   independent checks without pickling a live Python model into the result.
5. Preserve core `CrackSet2D` identity, paired one-sided trace samples, and an
   accepted training/validation/refinement `IntegrationEvidence` record.

## Next providers

### Neural fields

- **Milestone A -- single-crack vector elasticity (foundation implemented):**
  plane stress/strain, Mode-I/II mixtures, a physical jump, differentiable
  Williams enrichment, arbitrary straight-tip orientation, independent energy
  integration, and ring-resolved SIF/J now run through one AgentFEM Step. The
  common interaction integral is checked
  first against exact fields and an ordinary P2 FEM center crack. Remaining
  finite-domain displacement/traction adapter now executes through a separate
  `experimental_solver` provider. The public Benvenuti et al. mixed-mode
  extended patch test now runs with a boundary-terminating crack, one explicit
  active tip, exact spatial Dirichlet enforcement, independently extracted
  `K_I/K_II`, and physics-quality checks. It is patch-test evidence, not a
  predictive finite-domain validation; the traction-driven Griffith limit
  remains the next solver gate. A topology-aware, locally adaptive cut-cell rule now gives every
  training, validation, and refinement point an explicit crack-side identity,
  preserves area, and records independent grid fingerprints.
- **Milestone B -- static multi-crack fields (experimental solver
  implemented):** mutually separated internal straight cracks now enter one
  joint energy solve. Crack and tip identities are stable, integration radii
  are limited by other cracks and domain boundaries, and every tip returns an
  independent ring-resolved SIF/J report. A manufactured disjoint-neighborhood
  test verifies the extractor. The remaining promotion gate is a public
  interacting two-crack benchmark in which every active tip satisfies the
  declared path-variation tolerance. Intersecting, touching, curved, and
  general boundary-terminating geometries continue to fail closed. The narrow
  supported form is one inactive crack mouth on a rectangular boundary plus
  one or more interior active tips declared through stable endpoint identity.
- **Milestone C -- research production (evidence foundation implemented):**
  the provider now emits one ParaView VTU containing the bulk grid and
  duplicated, coincident crack-face traces, so the two displacement values are
  not averaged into one. A memory-bounded multi-axis audit reduces each heavy
  run immediately to scalar metrics and per-tip evidence, and refuses
  acceptance when one required tip disappears. Network, integration density,
  random seed, and extraction-ring slices remain opt-in numerical campaigns;
  their results are not bundled as if they had already converged.
  Checkpoint/best-state and explicit warm start with cold-start fallback remain
  open.
- Add a DeepXDE binding only where `NeuralFieldSpec` can be lowered without
  pretending that arbitrary UFL is a valid strong-form residual.
- Keep each framework's backend selection inside its provider; the core
  contract must not assume PyTorch because DeepXDE also supports TensorFlow,
  JAX, and PaddlePaddle.

### Neural operators

- Begin with a data-processor adapter from `ScientificDataset`,
  `FieldEncoding`, and `ObservationGrid` to explicit keyed input/output
  batches.
- Bind the maintained `neuraloperator` trainer and models rather than cloning
  FNO implementations into AgentFEM-Learning.
- Return held-out field error, boundary/balance error, resolution transfer,
  applicability, and model-state artifacts through AgentFEM evidence.
- Do not route neural operators through `NeuralFieldSpec`: a function-to-
  function map and a per-problem optimized field are different contracts.

## Promotion rule

A provider moves from `experimental_reference` only after it has a readable
case, deterministic small regression, independent scientific check, held-out
or external benchmark, dependency compatibility range, unsupported-case
declaration, and wheel-installed end-to-end test.

For the finite-domain provider, "runs" is deliberately not synonymous with
"validated". Boundary residual, independent integration, and optimization may
pass while one or more SIF rings remain uncertain. That result is useful
diagnostic evidence, not a benchmark pass.

## Current XDEM-D promotion ledger

| Gate | Current state | Promotion evidence |
|---|---|---|
| finite-domain mixed Mode I/II | public X-VEM extended patch test accepted | hard spatial boundary, one active tip, `K_I/K_II`, path, traction and equilibrium checks pass; classified as patch evidence |
| finite center-crack kinematics | public Westergaard extended patch test accepted | both active tips recover normalized `K_I` within 0.1%; equilibrium, face traction, energy refinement and independent SIF agreement pass |
| interacting two-crack field | exact four-tip collinear reference registered | one result must reproduce all four tips; no averaging across tips |
| cut-domain integration | implemented and regression-tested | area-conserving regular/cut/tip cells, one-sided point identity, distinct training/validation/refinement fingerprints |
| multi-axis numerical stability | executable bounded-memory audit | network, quadrature, seed, and ring axes must each pass |
| discontinuous visualization | implemented and regression-tested | one VTU, duplicated crack-face nodes, explicit `CrackSide` and `CrackId` |
| independent SIF cross-check | implemented and regression-tested | interaction integral and zero-radius COD extrapolation must agree within the declared internal contract |
| crack growth | intentionally unopened | all previous gates must pass first |

The earlier traction-driven Griffith truncation is a negative scientific
result, not a completed gate. Exact rigid gauges, additive jump/Williams
channels, deterministic integration, and load-derived initialization removed
several numerical ambiguities, but the converged field still failed SIF,
path-independence, crack-face traction, and bulk-equilibrium acceptance. The
spatial hard-Dirichlet and active-tip milestone is now implemented. The
published X-VEM and exact Westergaard problems are intentionally classified as
extended patch tests because analytic fields supply admissible interior
liftings. The latter recovers both normalized center-crack `K_I` values as
0.99924; equilibrium, crack-face traction, energy refinement, path and
independent COD checks all pass. The separate traction-driven Griffith
prediction remains closed: its analytic two-sheet representation reaches
normalized `K_I` about 0.84 with about 1% path variation, but crack-face
traction remains about 75%. The next predictive milestone is a traction-free
correction that does not suppress the physical crack response, followed by
domain-size convergence and the four-tip collinear-crack reference.

Registering a reference is not a solver pass. The single-crack reference uses
Benvenuti et al.'s public mixed-mode X-VEM problem and preserves its boundary
crack mouth as an inactive endpoint rather than inventing a second tip. The
two-crack values are the exact unbounded-elastic reference tabulated by Liew et
al. at `a/L=0.5`; a finite-domain approximation must demonstrate domain-size
convergence before comparison to that limit.

Official references:

- [Python package entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
- [NeuralOperator user guide](https://neuraloperator.github.io/dev/user_guide/index.html)
- [DeepXDE backend installation](https://deepxde.readthedocs.io/en/stable/user/installation.html)
- [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [Benvenuti et al., mixed-mode X-VEM benchmarks](https://arxiv.org/abs/2111.04150)
- [Westergaard center-crack displacement formulation and verification](https://doi.org/10.1007/s10704-019-00351-3)
- [Liew et al., tabulated two-collinear-crack reference](https://doi.org/10.1002/nme.1786)
- [Wang et al., XDEM graph-field architecture](https://www.nature.com/articles/s41467-026-76748-1)
- [Official XDEM reference implementation](https://github.com/yizheng-wang/XDEM)
