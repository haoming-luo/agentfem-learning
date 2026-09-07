# FEM-to-FNO heat operator

This example generates a family of steady AgentFEM heat-conduction solutions,
records the source and temperature fields in a `ScientificFieldDataset`, and
trains the official NeuralOperator FNO provider through the ordinary
`model.step(...)` workflow.

The example deliberately separates three assets:

1. deterministic FEM cases that create scientific evidence;
2. a portable, fingerprinted field dataset;
3. a reloadable learned operator and its held-out verification result.

Run from an environment containing AgentFEM, AgentFEM-Learning, PyTorch, and
the optional `neuraloperator` dependency:

```bash
python examples/fno_heat_operator/case.py --output outputs/fno_heat_operator
```

The learned model is a fast approximation over the sampled heat-source
family. It does not replace the underlying FEM solver or silently extend its
verified applicability domain.
