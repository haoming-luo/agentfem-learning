# AgentFEM-Learning development contract

- Keep AgentFEM core independent of PyTorch and XDEM-specific architectures.
- Add general learning semantics to AgentFEM before inventing provider-only
  synonyms.
- Every executable provider must enter through an AgentFEM Step provider or
  the framework-neutral executor boundary and return `SimulationResult`.
- Keep neural-field solvers, neural operators, surrogates, and learned
  constitutive laws in distinct subdomains even when they share PyTorch.
- Do not claim general XDEM, crack growth, phase-field fracture, or validation
  from the small Williams Mode-III reference problem.
- A loss decrease is not scientific verification. Preserve independent field,
  boundary, energy, crack-jump, and crack-face-traction checks.
- Never hide physical discontinuities in feature engineering that automatic
  differentiation cannot price.
- Preserve upstream citations and licenses if upstream XDEM code is reused in
  a future provider. The initial reference implementation is independent.
