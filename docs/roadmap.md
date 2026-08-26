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

- **Milestone A -- single-crack vector elasticity:** implement plane stress and
  plane strain, displacement/traction conditions, jump and optional Williams
  enrichment, and Mode-I/mixed-mode public benchmarks. Build the SIF/J
  interaction-integral adapter against analytical and FEM fields before using
  XDEM output as its input.
- **Milestone B -- static multi-crack fields:** support mutually separated
  straight cracks, stable per-tip identities, geometry-aware integration
  radii, independent SIF reports, and one public two-crack benchmark. Reject
  intersecting, touching, curved, and boundary-terminating geometries until a
  separately verified provider supports them.
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

Official references:

- [Python package entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
- [NeuralOperator user guide](https://neuraloperator.github.io/dev/user_guide/index.html)
- [DeepXDE backend installation](https://deepxde.readthedocs.io/en/stable/user/installation.html)
- [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
