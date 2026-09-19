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
- Record deterministic input-point and output-query permutation equivariance
  as a provider-owned verification claim in every GINO result.
- Expose NeuralOperator's maintained compact-support output-GNO weighting
  functions with validated, persisted provider options.
- Add evidence-driven parameter-path refinement plans that select a bounded,
  diverse set of high-risk cases for new trusted-solver evaluations.
- Audit the independent information content of FNO, TFNO and GINO datasets,
  preserve exact replicas as visible evidence, reject contradictory duplicate
  labels, and keep repeated physical inputs atomic during automatic
  partitioning so they cannot leak from training into validation.
- Attach the resulting independent-input and label-consistency audit as an
  `operator_dataset_integrity` verification claim to every maintained neural
  operator result.
- Add an auditable parameter-path refinement transaction: trusted failed-path
  cases move into training, leave the validation path, preserve a minimum
  independent audit set, and retain before/after dataset fingerprints before
  the ordinary provider retrains the model.
- Add a distinct high-fidelity acquisition transaction for cases that have not
  yet been computed: failed path evidence proposes bounded interval midpoints,
  lowers them to AgentFEM Campaign sampling, and verifies returned parameter
  values before merging new scientific fields.
- Add provider-neutral independent-seed field disagreement as an acquisition
  risk signal, while explicitly refusing to present ensemble spread as a
  calibrated physical error bound.

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
