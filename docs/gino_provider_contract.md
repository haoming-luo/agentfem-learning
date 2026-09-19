# Geometry-informed neural-operator contract

## Purpose

The first NeuralOperator provider learns maps on registered structured grids.
The experimental geometry-aware provider uses GINO for finite-element families in
which coordinates matter to the operator.  It must not present a padded FNO
array as support for arbitrary meshes.

The public scientific path remains:

```text
AgentFEM cases -> scientific geometry/field data -> NeuralOperatorSpec
               -> GINO provider -> SimulationResult + reloadable model
```

AgentFEM owns physical field meaning, units, parameter identity, geometry and
case fingerprints, partitions, evidence and result custody.  The companion
owns conversion to NeuralOperator tensors, geometry grouping, training and
reloadable inference.  NeuralOperator owns the GINO architecture.

## Three geometries, three meanings

GINO uses three coordinate objects that must remain distinct:

1. `input_geometry`: coordinates carrying the input function;
2. `latent_grid`: a declared regular grid on which the FNO blocks operate;
3. `output_queries`: coordinates at which output fields are requested.

Each object needs dimension, coordinate-system, unit and content identity.
Changing any one invalidates cached training and inference artifacts.

## First supported family

The implemented first slice accepts registered finite-element families
with fixed point count and channel meaning, while coordinates may vary by
case.  Cases are grouped by an exact geometry fingerprint.  Cases sharing one
geometry may form a mini-batch; distinct geometries execute as separate
micro-batches and accumulate gradients before one optimizer step.

This design follows the current upstream GINO constraint: input and output
geometry must be shared within a native batch.  A family containing one
geometry per case therefore remains correct, but may train with micro-batch
size one.

The first slice supports:

- explicit input coordinates and output query coordinates;
- one declared latent regular grid;
- named point fields and scalar case parameters;
- geometry-aware train, validation and test partitions;
- held-out geometry evidence distinguished from ordinary held-out field error;
- output-query transfer isolated on input geometries already present in training;
- prediction on a new registered geometry without changing field names;
- exact recording of geometry, neighborhood radii and neighbor-search backend;
- safe model-state persistence and the ordinary AgentFEM result lifecycle.

The implementation uses the upstream pure-PyTorch neighbor fallback by
default. Open3D and `torch-scatter` remain disabled until their optional
dependency combinations pass installed-wheel evidence; selecting them now
fails before training rather than changing execution silently.

The output GNO may use NeuralOperator's maintained compact-support weighting
functions through `output_weighting_function`. Supported values are `bump`,
`half_cos`, `quadr`, `quartic`, and `octic`; `None` preserves the unweighted
upstream path. The selected function and scale are part of the saved model and
geometry configuration. They are explicit numerical choices, not an automatic
accuracy claim.

Every training result also checks point-order semantics. It reverses input
points, cyclically permutes output queries, restores the output order, and
records `permutation_equivariance` as an ordinary verification claim. This
checks a numerical invariant of the coordinate-defined operator; it does not
replace physical held-out evidence.

## Deliberate rejection boundary

Variable point counts are not represented by padding and a mask merely to
obtain a rectangular NPZ tensor.  Until a case-indexed ragged storage backend
exists, such families must fail before training with a specific capability
message.  Mixed spatial dimensions, changing channel semantics and unlabeled
coordinate systems also fail closed. Reordering invariance is measured rather
than assumed from the registered-family declaration.

The provider must not claim topology generalization solely because coordinates
change.  Held-out evidence distinguishes at least:

- parameter interpolation on a known geometry;
- a new deformation of a registered topology;
- a new discretization resolution;
- a new topology or geometric class.

Aggregate held-out error is not sufficient evidence for a continuous design
space.  The provider-owned held-out claim accepts against the maximum per-case
relative L2 error and reports the global, median, 95th-percentile, and
per-output metrics separately.  A promoted varying-geometry case must also
evaluate one or more dense, independent parameter paths.  AgentFEM-Learning's
`parameter_path_reliability_check(...)` reports the worst per-case physical
field error and rejects isolated interior error spikes even when the global
error limit still passes.  Training nodes alone do not constitute a path
audit.

## Dependency policy

GINO remains an optional method extra.  PyTorch, NeuralOperator and optional
neighbor-search accelerators do not enter AgentFEM core.  The result records
whether neighborhood construction used Open3D, `torch-scatter`, or a reviewed
fallback.  A missing accelerator may change performance, never physical
meaning or acceptance thresholds.

## Promotion evidence

The provider remains experimental until all of the following pass from an
installed wheel:

1. identity mapping on an irregular point cloud;
2. one fixed-geometry FEM operator with held-out parameters;
3. one geometry-varying family with disjoint held-out geometries;
4. independent output-query and resolution transfer;
5. permutation/reordering invariance where the declared representation
   requires it;
6. geometry, boundary or balance checks supplied by the problem adapter;
7. cold reload reproducing predictions within a declared tolerance;
8. CPU dependency and memory bounds, plus one accelerated backend smoke test;
9. dense parameter-path evidence without an unresolved interior error spike;
10. point-order permutation evidence recorded in every training result.

## References

- NeuralOperator GINO documentation and maintained implementation:
  <https://neuraloperator.github.io/dev/modules/generated/neuralop.models.GINO.html>
- Geometry-Informed Neural Operator:
  <https://doi.org/10.48550/arXiv.2309.00583>
