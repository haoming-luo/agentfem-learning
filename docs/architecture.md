# Provider architecture

AgentFEM-Learning is one broad companion distribution. XDEM, future DeepXDE
bindings, neural operators, and learned constitutive models remain explicit
subdomains inside it; they do not become separate official repositories merely
because their algorithms differ.

## Ownership

| Layer | Owner |
| --- | --- |
| fields, objectives, conditions, integration identity, inverse parameters | AgentFEM core |
| crack geometry, tip identity, SIF/J evidence, result trust | AgentFEM core |
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

## First capabilities

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
reference. Training points, held-out validation points, and a refined rule are
separate `IntegrationRule` records. Their objective values are compared in one
`IntegrationEvidence` record attached to `SimulationResult`.

The branch cut is a core `CrackSet2D` asset with stable crack and tip IDs. The
NPZ field contains paired one-sided trace coordinates and side labels, so its
physical jump is not erased by continuous visualization interpolation.

The vector reference uses the same provider lifecycle for two displacement
components. Plane-stress or plane-strain energy is differentiated by PyTorch;
Mode-I and Mode-II Williams bases carry the declared crack jump. Autograd
stress and displacement-gradient samples then enter AgentFEM core's
solver-neutral interaction integral. Therefore the analytical field, ordinary
FEM benchmark, and neural field use the same (K_I/K_{II}/J) conventions and
path-sensitivity report rather than three provider-specific postprocessors.

## Promotion route

1. Keep the Williams field as the packaging, discontinuity, integration, and
   provider regression.
2. Implement the interaction/domain-integral adapter first against analytical
   and ordinary FEM fields, independently of neural optimization.
3. Add one vector-elastic 2D XDEM provider for a published Mode-I benchmark,
   then an inclined mixed-mode benchmark. Review upstream licensing before
   reusing code; keep an independent implementation possible.
4. Add multiple non-intersecting straight cracks only after per-tip SIF and
   unsupported-geometry failures are stable.
5. Add phase-field energy and irreversibility only after AgentFEM's common
   phase-field contract exists.
6. Add crack growth only with path, energy, sampling, and repeatability
   convergence evidence.
