# Changelog

## 0.1.0a1 - Unreleased

- Add the first official NeuralOperator provider with FNO and TFNO models,
  deterministic field-dataset splitting, channel normalization, early
  stopping, best-state recovery, held-out physical-field error, safe
  tensor-only checkpoints and reloadable inference.
- Add a shared framework-neutral training ledger so later neural-operator,
  neural-field and user-provider workflows can retain the same bounded
  optimization evidence.
- Add a real AgentFEM steady-heat-to-FNO example using the core
  `ScientificFieldDataset` contract and ordinary `model.step(...)` lifecycle.
- Keep boundary, balance and out-of-distribution checks explicitly
  inconclusive unless a provider computes them; supervised loss alone does not
  promote a learned operator to broad scientific validity.
- Add named, source-fingerprinted `OperatorCheck` evaluators, reject case-ID
  overlap between data partitions, and distinguish genuine spatial-resolution
  transfer from another held-out dataset on the training grid.
- Implement the official NeuralOperator GINO provider with distinct input,
  latent and output geometries; exact-geometry native batches; cross-geometry
  gradient accumulation; training-fitted coordinate transforms; graph-neighbor
  coverage checks; safe reload; and separate geometry- and output-query-transfer
  evidence.
- Add a real varying-width AgentFEM heat-transfer example. Keep new topology,
  ragged point families and optional accelerated neighbor backends outside the
  first validated boundary.
- Add a provider-neutral parameter-path reliability check for geometry and
  other scalar design paths. It records per-output held-out field error,
  worst-case identity and isolated interpolation spikes as ordinary AgentFEM
  verification evidence.
- Make the GINO held-out claim depend on the maximum per-case physical-field
  error and add aggregate, median, 95th-percentile, and per-output metrics.

- Add the independent AgentFEM extension entry point.
- Add a provider-neutral Mode-III `NeuralFieldSpec`.
- Add a PyTorch Williams-enriched deep-energy reference Step.
- Return field, optimization, artifact, and analytical verification evidence
  through `SimulationResult`.
- Add deterministic representation, repeatability, extension, and end-to-end
  tests.
- Establish `agentfem-learning` as the broad companion distribution and place
  the first provider under the explicit `neural_fields.xdem` subdomain.
- Retire the standalone local `agentfem-xdem` project identity; XDEM now lives
  only as a provider subdomain of AgentFEM-Learning.
- Make the clean-room CI environment explicit about AgentFEM's HDF5, MPI, and
  PETSc runtime dependencies while keeping them in one conda-forge stack.
- Add plane-stress and plane-strain vector Williams neural fields with mixed
  Mode-I/II enrichment and paired crack-face output.
- Feed autograd displacement-gradient and stress samples into AgentFEM's common
  ring-resolved SIF/J interaction-integral evidence.
- Add a verified `SimulationResult` path for vector displacement, stress,
  integration evidence, optimization history, and crack-tip quantities.
