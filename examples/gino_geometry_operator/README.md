# GINO across registered geometries

This example solves steady heat transfer with AgentFEM on rectangular plates
of different width, samples named source and temperature fields at registered
physical coordinates, and trains one official NeuralOperator GINO through the
ordinary `model.step(...)` lifecycle.

Training, validation and geometry-test widths are disjoint. The resulting
`geometry_transfer` claim therefore concerns unseen deformations of one
registered rectangular topology. It is not evidence for new topology classes.

```bash
python examples/gino_geometry_operator/case.py \
  --output outputs/gino_geometry_operator \
  --smoke
```

The output keeps all three scientific partitions, reloadable tensor-only GINO
state, training ledger, held-out point fields, geometry transform and a normal
AgentFEM `result.json`. The smoke mode is a workflow check; promotion requires
the full run plus the independent checks listed in
[`docs/gino_provider_contract.md`](../../docs/gino_provider_contract.md).
