# Changelog

## 0.1.0a1 - Unreleased

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
