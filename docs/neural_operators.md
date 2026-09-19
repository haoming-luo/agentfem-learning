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
framework bindings and training execution. NeuralOperator owns the
FNO/TFNO/GINO architectures and numerical layers. A laboratory-owned model may consume the
same core contracts without using this companion.

## Current provider

`agentfem-learning.neuraloperator` supports FNO and TFNO on fixed, structured
observation grids. Every field uses the explicit layout
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
- disjoint case-identity enforcement across training, validation, and test data;
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

## Geometry-informed operator

The experimental GINO route learns one operator across a registered family of
coordinate-defined finite-element fields. Its public data layout is
`(case, channel, point)` plus coordinate arrays `(case, point, dimension)`:

```python
operator_spec = learning.NeuralOperatorSpec(
    architecture="gino",
    inputs=(load_encoding,),
    outputs=(response_encoding,),
    boundary_encoding="explicit point coordinates",
    required_checks=(
        "held_out_field_error",
        "geometry_transfer",
        "output_query_transfer",
    ),
)

result = model.step(
    target=operator_spec,
    dataset=training_dataset,
    validation_dataset=validation_dataset,
    test_dataset=unseen_geometry_dataset,
    input_geometry="nodes",
    coordinate_system="cartesian",
    coordinate_unit="m",
    latent_shape=(16, 16),
    n_modes=(8, 8),
    input_radius=0.25,
    output_radius=0.25,
).solve_result()
```

`input_geometry`, the regular `latent_shape`, and optional `output_queries`
have different meanings and remain separate. Physical coordinates are mapped
to one training-fitted unit box; the transform, units, neighborhood radii and
backend are stored with the model. The default `native` neighbor backend is a
reviewed PyTorch fallback and does not require Open3D or `torch-scatter`.

NeuralOperator currently requires every native batch to share geometry. The
provider therefore groups exact geometry fingerprints, uses ordinary batches
inside a group, and accumulates gradients across distinct-geometry
micro-batches. Every case still carries its own geometry identity and no case
is padded.

```python
from agentfem_learning.neural_operators.neuraloperator import load_predictor

predictor = load_predictor("outputs/gino/operator_state.pt")
fields = predictor.predict(
    {"load": load_on_new_geometry},
    input_geometry=new_nodes,
    output_queries=query_points,
)
```

The first provider deliberately requires `registered_mesh_family` field
semantics and rejects padded or masked point clouds. It supports different
coordinates across cases, but does not claim unseen topology generalization,
ragged per-case point counts, or physical validity from training loss alone.
`geometry_transfer` is emitted only for exact geometries absent from training;
its stated domain is registered topology deformation. A changed independent
output query set is reported separately as `output_query_transfer` only when
the corresponding input geometry occurred in training. Thus a finer query grid
is neither mislabeled as a new physical geometry nor confounded with one.

## Evidence is not inferred from loss

The generic provider can compute held-out field error. Boundary error,
conservation or balance error, and out-of-distribution behavior depend on the
physics and must be supplied by a reviewed problem adapter. A named
`OperatorCheck` receives the held-out physical prediction, reference fields,
dataset, specification, and metrics, and must return an AgentFEM
`VerificationClaim`. Its version, callable identity, and source hash are part
of the result's scientific inputs. When required checks are unavailable, the
result records an inconclusive claim and does not silently promote the model.

```python
check = OperatorCheck(
    name="conservation_or_balance_error",
    evaluator=heat_balance_check,
    version="steady-heat-v1",
)

result = model.step(
    target=operator_spec,
    dataset=field_dataset,
    check_evaluators=(check,),
).solve_result()
```

The built-in `resolution_transfer` claim is emitted only when an independent
test dataset actually changes the spatial resolution. Supplying another
dataset on the training resolution remains ordinary held-out testing and does
not satisfy that check.

Geometry operators also need evidence between the chosen training nodes.  A
small mean validation error can hide one severe interpolation failure inside
the parameter domain.  The reusable parameter-path check consumes an
independent, ordered validation slice and records both the worst physical-field
error and isolated interior error spikes:

```python
from agentfem_learning.neural_operators.neuraloperator import (
    parameter_path_reliability_check,
)

path_check = parameter_path_reliability_check(
    "hole_radius",
    output_tolerances={"displacement": 0.08, "von_mises_stress": 0.15},
    maximum_spike_ratio=2.5,
)

result = model.step(
    target=operator_spec,
    dataset=training_dataset,
    validation_dataset=independent_radius_path,
    check_evaluators=(path_check,),
).solve_result()
```

The path must contain at least three distinct parameter values and independent
reference fields.  The resulting claim identifies the worst case and spike
location; it does not infer continuity from optimizer loss or from distance to
the nearest training point.

GINO's provider-owned `held_out_field_error` is conservative as well: its
acceptance value is the maximum per-case relative L2 error.  The global,
median-case, 95th-percentile, and per-output errors remain available as result
quantities, so many easy geometries cannot hide one failed held-out geometry.

For geometry paths that are sensitive to the hard output-neighborhood cutoff,
the provider exposes the compact-support kernels maintained by NeuralOperator:

```python
model.step(
    target=operator_spec,
    dataset=training_dataset,
    output_weighting_function="half_cos",
    output_weighting_scale=1.0,
)
```

This changes only the output GNO quadrature weighting. It must be selected by
held-out and path evidence; AgentFEM-Learning does not silently change the
architecture after seeing validation results.

When a path fails, the same evidence can produce a bounded simulator-sampling
plan:

```python
from agentfem_learning.neural_operators.neuraloperator import (
    parameter_path_refinement_plan,
)

plan = parameter_path_refinement_plan(
    path_claim,
    existing_values=training_dataset.parameters["hole_radius"],
    maximum_candidates=3,
)
```

The plan balances measured field risk with distance from existing samples. It
does not synthesize labels or mutate the dataset: the project evaluates those
parameters with AgentFEM or another declared reference solver, appends the new
cases, and reruns the same independent path check.

FNO/TFNO remains the structured-grid route rather than a universal finite
operator. GINO is the coordinate-aware route under the
[geometry-informed provider contract](gino_provider_contract.md). Both remain
experimental until their independent promotion evidence is complete.

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
