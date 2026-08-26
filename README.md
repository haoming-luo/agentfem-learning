# AgentFEM-Learning

`agentfem-learning` is the optional official companion for reviewed
scientific-learning integrations around AgentFEM. AgentFEM itself remains
usable with any user-owned model and does not require this package. The
companion provides batteries-included providers, reference implementations,
examples, and benchmark evidence without moving PyTorch or method-specific
architectures into the finite-element core.

This is the single official learning companion. Individual methods such as
XDEM are subdomains, not separate top-level projects:

```text
AgentFEM core      scientific contracts, execution lifecycle, results
AgentFEM-Learning maintained method providers, examples, benchmark evidence
user packages      laboratory-owned or private models using the same contract
```

The first provider family is `neural_fields.xdem`, proving one complete workflow:

```text
NeuralFieldSpec -> AgentFEM Step Provider -> PyTorch energy optimization
                -> SimulationResult -> field artifact + verification evidence
```

The XDEM providers are experimental. The packaging regression is a normalized
Williams Mode-III field. The first vector-elastic reference adds plane stress,
plane strain, Mode I/II mixtures, arbitrary crack-tip translation and
orientation, two-sided displacement output, and common ring-resolved SIF/J
evidence on the same slit annulus. An experimental finite-domain XDEM-D
provider additionally accepts rectangular domains, constant displacement and
traction boundaries, spatial hard Dirichlet data, multiple separated straight
cracks, and an inactive boundary crack mouth with explicit active-tip identity.
The public mixed-mode X-VEM extended patch test is accepted; it verifies exact
boundary enforcement, enrichment reproduction, and independent SIF/J
extraction, not predictive crack interaction. These capabilities do not claim
curved or intersecting cracks, crack growth, phase-field fracture, or external
multi-crack validation.

## What the first reference proves

- residual-only PINN vocabulary is not required for an energy method;
- the physical energy, conditions, sampling, and Williams enrichment remain
  visible in an AgentFEM `NeuralFieldSpec`;
- the executable trainer is selected through AgentFEM's ordinary
  `model.step(...)` provider mechanism;
- loss history, predicted field, model state, software identity, and
  independent analytical checks return in one `SimulationResult`;
- the crack jump exists only on the declared branch cut—feature engineering is
  not allowed to introduce autograd-invisible internal discontinuities.

The accepted reference checks cover displacement, prescribed boundaries,
strain energy, crack opening jump, independent integration consistency, and,
for vector elasticity, (K_I), (K_{II}), (J), and extraction-domain
variation. The field artifact preserves paired one-sided crack-face samples
instead of averaging the discontinuity away.

## Why a companion project

Use AgentFEM core directly when a laboratory already owns a model:

```python
step = model.step(target=spec, executor=my_solver)
```

Use AgentFEM-Learning when a maintained method binding, dependency policy,
standard artifacts, readable example, and independent benchmark are valuable.
The repository is broad; its scientific subdomains remain narrow:

```text
agentfem_learning
  neural_fields
    xdem
  neural_operators       # future, distinct from neural-field solvers
  learned_constitutive   # future
```

See the [development roadmap](docs/roadmap.md) for the evidence required before
new providers or maturity claims are added. The method-neutral fracture/XDEM
boundary is specified in the
[fracture-field contract](https://github.com/haoming-luo/agentfem-learning/blob/main/docs/xdem_fracture_contract.md).

## Installation during development

The vector provider targets the next AgentFEM release containing the common
LEFM interaction-integral contract (`agentfem>=0.2.6`). Until that release is
published, keep `agentfem` and `agentfem-learning` as sibling repositories and
use the current AgentFEM source tree for development.

For a conda-forge FEniCSx environment, install PyTorch from conda-forge so it
shares the environment's OpenMP runtime:

```bash
mamba install -n fenicsx-env -c conda-forge pytorch
conda activate fenicsx-env
python -m pip install -e ../agentfem
python -m pip install -e '.[xdem]' --no-deps
```

Run these commands from this repository root with the AgentFEM source tree in
the sibling `../agentfem` directory. The temporary `--no-deps` prevents pip
from searching PyPI for the not-yet-released core compatibility floor.

Do not use `KMP_DUPLICATE_LIB_OK` to conceal duplicate OpenMP runtimes. On
Apple Silicon the conda-forge build also exposes the PyTorch MPS device. The
reference case defaults to reproducible CPU `float64`; use
`--device mps --dtype float32` when faster exploratory training is preferred.

## Run the reference cases

```bash
python examples/mode_iii_tip/case.py --output outputs/mode_iii_tip
python examples/vector_mixed_mode_tip/case.py
python examples/finite_domain_benchmarks/case.py --case xvem
python examples/finite_domain_benchmarks/case.py --case center
python examples/finite_domain_benchmarks/case.py --case two
```

The `xvem` command is a public extended patch test and should be accepted. The
`center` and `two` commands remain predictive promotion candidates: they may
finish successfully while `published_benchmark.json` correctly says `failed`.
Promotion requires the full convergence ledger in the roadmap.

The four required numerical axes use one bounded-memory entry point:

```python
report = xdem.run_finite_domain_convergence(
    specification,
    training_options,
    network_layers=((32, 32), (48, 48, 48)),
    integration_counts=(2048, 4096),
    seeds=(2026, 2027),
    ring_resolutions=((20, 80), (32, 128)),
    output="outputs/xdem_convergence.json",
)
```

Cases run sequentially by design. Each trained network is reduced to metrics
and per-tip evidence and then released before the next case starts; parallel
campaign execution is deferred until it has an explicit device-memory budget.

The installed-package acceptance gate consumes that ordinary result rather
than running a private test path:

```bash
python acceptance_gate.py outputs/mode_iii_tip/result.json \
  --report outputs/extension-acceptance.json
```

It proves that an installed extension was discovered through the public entry
point, added its declared providers without modifying AgentFEM core, and returned an
integrity-checked `SimulationResult`.

The vector-elastic model is equally short:

```python
from math import pi

from agentfem_learning.neural_fields.xdem import vector_tip_spec

result = model.step(
    target=vector_tip_spec(
        assumption="plane_stress",
        k_i=1.0,
        k_ii=0.5,
        crack_angle=pi / 6,
    ),
    epochs=500,
    output="outputs/vector_mixed_mode_tip",
).solve_result()
```

Its result contains `U` and `S` field records, paired crack-face traces, and a
`stress_intensity.json` artifact generated through AgentFEM's common LEFM
interaction-integral contract.

The finite-domain public language remains problem-oriented:

```python
from agentfem import fracture
from agentfem_learning.neural_fields import xdem

cracks = fracture.crack_set(
    fracture.segment("lower", start=(-1.0, -0.5), end=(1.0, -0.5)),
    fracture.segment("upper", start=(-1.0, 0.5), end=(1.0, 0.5)),
)
problem = xdem.static_crack_problem(
    domain=xdem.rectangular_domain((-3.0, 3.0, -2.0, 2.0)),
    material=material,
    cracks=cracks,
    conditions=(
        xdem.displacement_bc("fixed_bottom", "bottom", (0.0, 0.0)),
        xdem.traction_bc("pull_top", "top", (0.0, 1.0e6)),
    ),
)
result = model.step(
    target=xdem.finite_domain_spec(problem),
    output="outputs/two_cracks",
).solve_result()
```

For remote-traction benchmarks, use `xdem.point_displacement(...)` only as a
minimal rigid-body gauge, or start from
`xdem.center_crack_domain_problem(...)` and
`xdem.two_collinear_cracks_domain_problem(...)`. These factories embed the
published normalization and make the provider write
`published_benchmark.json`; they do not pre-approve the numerical result.

The result keeps four stable tip identities and four independent
`K_I/K_II/J` ring reports. It also writes one ParaView VTU with duplicated
coincident crack faces, preserving both one-sided displacement values through
explicit `CrackSide` and `CrackId` arrays. `experimental_solver` means that the
workflow is executable; it is not promoted to an externally validated
interacting-crack solver until the public benchmark and multi-axis convergence
gates in the roadmap pass.

The user-facing model remains short:

```python
from agentfem import extensions, models, studies
from agentfem_learning.neural_fields.xdem import mode_iii_tip_spec

extensions.load_extension("agentfem-learning.xdem")

model = models.create(
    study=studies.static_solid(
        dimension=2,
        assumption="plane_strain",
        name="mode_iii_neural_field",
    ),
    name="williams_mode_iii_tip",
)

step = model.step(
    target=mode_iii_tip_spec(),
    epochs=500,
    output="outputs/mode_iii_tip",
)
result = step.solve_result()
```

The output directory contains:

- `result.json`: portable AgentFEM result and verification manifest;
- `mode_iii_field.npz`: coordinates, prediction, and analytical reference;
- `crack_geometry.json`: stable crack and crack-tip identity;
- `integration_evidence.json`: training, held-out, and refined integration;
- `model_state.pt`: PyTorch state dictionary and scientific specification.

## Architectural boundary

AgentFEM owns scientific meaning, the user-executor boundary, provider
discovery, result evidence, and the human/agent workflow. This companion's
XDEM subdomain owns PyTorch training and XDEM-style representations. The public
XDEM research repository is not copied or vendored into this package.

A later provider may bind reviewed upstream implementations for general-domain
discrete fracture, phase-field fracture, and crack propagation. Each
capability must receive its own benchmark and maturity label before being
advertised.

## Upstream method and attribution

The Extended Deep Energy Method repository describes discontinuity and
Williams crack-tip enrichments for discrete fracture as well as continuous
phase-field variants:

- [Yizheng Wang et al., XDEM repository](https://github.com/yizheng-wang/XDEM)
- *Towards Unified AI-Driven Fracture Mechanics: The Extended Deep Energy
  Method (XDEM)*, arXiv:2511.05888 (2025)

This provider is an independent AgentFEM integration and is not presented as an
official upstream distribution.
