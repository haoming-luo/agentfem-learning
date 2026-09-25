# DENIM learned-constitutive examples

Prepare a local bundle as described in
[`docs/learned_constitutive.md`](../../docs/learned_constitutive.md), then run:

```bash
python examples/learned_constitutive_denim/case.py \
  --bundle models/denim-expanded
```

The example executes the published 121-step cyclic strain path through the
same rank-local batch contract intended for finite-element integration points.
It writes one compact JSON result containing stress, PEEQ, diagnostics, fixed
model/data identity, and runtime evidence. It does not download or duplicate
the model.

The same bundle can drive the ordinary three-dimensional implicit finite-
element lifecycle:

```bash
python examples/learned_constitutive_denim/global_bar.py \
  --bundle models/denim-expanded
```

This path uses AgentFEM's provider-neutral small-strain material Step. DENIM
remains an extension implementation: the global Newton, increment cutback,
state commit/rollback, checkpoint and result evidence belong to AgentFEM.
