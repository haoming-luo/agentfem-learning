# Provider architecture

AgentFEM-Learning is one broad companion distribution. XDEM, future DeepXDE
bindings, neural operators, and learned constitutive models remain explicit
subdomains inside it; they do not become separate official repositories merely
because their algorithms differ.

## Ownership

| Layer | Owner |
| --- | --- |
| fields, objectives, conditions, sampling, inverse parameters | AgentFEM core |
| PyTorch module, optimizer, XDEM enrichment, device selection | `agentfem_learning.neural_fields.xdem` |
| Study, Model, Step selection, extension identity | AgentFEM core |
| field samples, optimization history, scientific claims | SimulationResult |

The XDEM subdomain registers a `StepProvider`; it does not register a matrix/form
backend. A deep-energy optimizer solves a field problem and is neither a
surrogate nor a finite-element assembly backend.

The broader distribution shares packaging, extension compatibility, examples,
and result evidence. It does not erase the distinction between neural-field
optimization and future neural-operator training.

The boundary is intentionally asymmetric: AgentFEM core defines the stable
scientific request and result contracts, while this package may evolve with
replaceable frameworks. A user-owned callable can bypass this package entirely
and still enter the same Step and `SimulationResult` lifecycle. Provider code
therefore must not add scientific meanings that are invisible to the core
contract.

## First capability

The reference field is

\[
W(r,\theta)=\bar W\sqrt{r/R}\sin(\theta/2),
\]

on a slit annulus with a branch cut on the negative x-axis. The energy per unit
thickness is

\[
\Pi=\frac{\mu}{2}\int_\Omega \nabla W\cdot\nabla W\,\mathrm d\Omega,
\qquad
\Pi_{\mathrm{ref}}=
\frac{\pi\mu\bar W^2(R-r_c)}{4R}.
\]

The neural representation combines a regular MLP with the leading Williams
term. The regular field is multiplied by a radial envelope that vanishes on
the prescribed inner and outer circles. Consequently it cannot cancel the
boundary condition while the enrichment carries the physical crack jump.

Midpoint tensor-product integration is used for the energy. Small random
point clouds are intentionally avoided because an expressive network can
exploit unsampled regions and report a nonphysical energy below the harmonic
reference.

## Promotion route

1. Keep the Williams field as the packaging and provider regression.
2. Add a reviewed upstream discrete XDEM binding behind a separate provider
   name and reproduce one published Mode-I benchmark.
3. Add SIF extraction with analytical and FEM cross-checks.
4. Add phase-field energy and irreversibility only after AgentFEM's common
   phase-field contract exists.
5. Add crack growth only with path, energy, sampling, and repeatability
   convergence evidence.
