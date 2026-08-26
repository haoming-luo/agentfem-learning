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
  `experimental_solver` provider. An external published finite-domain vector
  benchmark remains the promotion gate; the slit-annulus manufactured
  reference remains `experimental_reference` rather than being relabelled.
- **Milestone B -- static multi-crack fields (experimental solver
  implemented):** mutually separated internal straight cracks now enter one
  joint energy solve. Crack and tip identities are stable, integration radii
  are limited by other cracks and domain boundaries, and every tip returns an
  independent ring-resolved SIF/J report. A manufactured disjoint-neighborhood
  test verifies the extractor. The remaining promotion gate is a public
  interacting two-crack benchmark in which every active tip satisfies the
  declared path-variation tolerance. Intersecting, touching, curved, and
  boundary-terminating geometries continue to fail closed.
- **Milestone C -- research production:** reuse AgentFEM Campaign for seeds and
  parameters; add checkpoint/best state, explicit warm start with cold-start
  fallback, multi-axis convergence, and ParaView output with duplicated
  crack-face topology.
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

Official references:

- [Python package entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
- [NeuralOperator user guide](https://neuraloperator.github.io/dev/user_guide/index.html)
- [DeepXDE backend installation](https://deepxde.readthedocs.io/en/stable/user/installation.html)
- [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
