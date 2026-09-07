# Neural operators in AgentFEM-Learning

## Capability boundary

A neural operator approximates a mapping between functions. AgentFEM keeps
that task separate from a neural-field solver, which optimizes one physical
field problem, and from a parameter-to-quantity surrogate.

```text
FEM cases -> ScientificFieldDataset -> NeuralOperatorSpec
          -> maintained provider -> SimulationResult + reloadable model
```

AgentFEM core owns field meaning, units, geometry policy, case identity,
partitions, provenance and result evidence. AgentFEM-Learning owns maintained
framework bindings and training execution. NeuralOperator owns the FNO/TFNO
architecture and spectral layers. A laboratory-owned model may consume the
same core contracts without using this companion.

## Current provider

`agentfem-learning.neuraloperator` currently supports FNO and TFNO on fixed,
structured observation grids. Every field uses the explicit layout
`(case, channel, spatial...)`. Scalar case parameters may be declared in
`NeuralOperatorSpec.parameter_inputs`; the provider broadcasts them as
constant channels while preserving their named dataset record.

The provider performs:

- deterministic train/validation partitioning;
- per-channel normalization fitted only on training cases;
- reproducible shuffled mini-batches;
- AdamW optimization, early stopping and best-state restoration;
- held-out relative L2 error in physical units;
- optional independent resolution-transfer testing;
- safe tensor-only state persistence and reloadable prediction;
- a bounded `TrainingLedger` and ordinary AgentFEM `SimulationResult`.

Training stays inside the public AgentFEM workflow:

```python
extensions.load_extension("agentfem-learning.neuraloperator")

result = model.step(
    target=operator_spec,
    dataset=field_dataset,
    n_modes=(8, 8),
    hidden_channels=32,
    output="outputs/heat_operator",
).solve_result()
```

The saved tensor state is a deployable artifact rather than a reference to the
training process:

```python
from agentfem_learning.neural_operators.neuraloperator import load_predictor

predictor = load_predictor("outputs/heat_operator/operator_state.pt")
fields = predictor.predict(
    {"heat_source": new_source_fields},
    parameters={
        "source_amplitude": amplitudes,
        "source_center_x": centers,
    },
)
```

Input names, parameter names, output names, channel statistics, architecture,
and the training-dataset fingerprint travel with that state. A missing named
input or parameter therefore fails before inference instead of being matched
by array position alone.

The model artifact is loaded with PyTorch's restricted `weights_only` path.
Provider configuration and scientific metadata are primitive records rather
than a pickled live Python model.

## Evidence is not inferred from loss

The generic provider can compute held-out field error. Boundary error,
conservation or balance error, and out-of-distribution behavior depend on the
physics and must be supplied by a reviewed problem adapter. When these checks
are required but unavailable, the result records an inconclusive claim and
does not silently promote the model.

FNO/TFNO is therefore the first structured-grid route, not a universal finite
operator. Geometry-varying and unstructured finite-element families should
use coordinate-aware operators. The next maintained target is GINO, whose
official formulation maps between arbitrary coordinate meshes and latent
regular grids.

## Storage progression

The first field-dataset backend uses compressed NPZ plus a JSON manifest. It is
portable, lossless and appropriate for tests and laboratory-sized datasets.
Large transient or three-dimensional campaigns will add optional chunked Zarr
storage behind the same `ScientificFieldDataset` contract. Storage choice must
not alter field semantics or trainer code.

## References

- NeuralOperator documentation and maintained model library:
  <https://neuraloperator.github.io/dev/user_guide/index.html>
- Fourier Neural Operator:
  <https://doi.org/10.48550/arXiv.2010.08895>
- Geometry-Informed Neural Operator:
  <https://doi.org/10.48550/arXiv.2309.00583>
- Zarr chunked array specification: <https://zarr.dev/>
- PDEBench: <https://github.com/pdebench/PDEBench>
- The Well: <https://github.com/PolymathicAI/the_well>
