# DENIM material-point example

Prepare a local bundle as described in
[`docs/learned_constitutive.md`](../../docs/learned_constitutive.md), then run:

```bash
python examples/learned_constitutive_denim/case.py \
  --bundle models/denim-expanded
```

The example executes the published 121-step cyclic strain path through the
same batched provider used by finite-element integration points. It writes one
compact JSON result containing stress, PEEQ, diagnostics, fixed model/data
identity, and runtime evidence. It does not download or duplicate the model.

After installing both packages from their wheels, the same local bundle can
enter the ordinary AgentFEM nonlinear Step:

```bash
python examples/learned_constitutive_denim/global_bar.py \
  --bundle models/denim-expanded
```

This second case solves a three-dimensional traction bar with four accepted
load increments. DENIM is evaluated once per MPI rank and Newton iteration as
a batch over local integration points; the public model, loads, Step and
`SimulationResult` lifecycle remain ordinary AgentFEM objects.
