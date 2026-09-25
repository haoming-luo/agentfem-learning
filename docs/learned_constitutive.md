# Learned constitutive provider

`agentfem-learning[constitutive]` is the maintained PyTorch runtime for
AgentFEM's framework-neutral learned-material contract. It is optional and does
not affect XDEM or neural-operator installations.

The public loading path is architecture-neutral:

```text
LearnedConstitutiveSpec
  -> verified local model bundle
  -> architecture registry
  -> cached batch material
  -> AgentFEM quadrature transaction
```

One explicit activation makes the provider available to the ordinary
AgentFEM material and Step workflow:

```python
from agentfem import extensions, materials

extensions.load_extension("agentfem-learning.learned-constitutive")
material = materials.learned(spec)
```

`spec` is an AgentFEM `LearnedConstitutiveSpec`; it remains independent of
PyTorch and may be serialized before this optional package is installed.

Every bundle contains:

```text
model.json
weights.safetensors
SHA256SUMS
README.md
```

The manifest uses `agentfem.learned_constitutive.bundle` version `1.0.0`.
Unknown schemas, versions, weights, or checksums fail before inference.

`model.json` declares kinematics, stress/tangent conventions, named
parameters, versioned state layout, required inputs, capabilities, dtype
policy, applicability domain, dataset identity, and the weights digest. The
provider never infers an architecture from a filename and never downloads
during a solve.

## DENIM v1

DENIM is the first registered architecture, not a special AgentFEM workflow.
Its discrete update keeps small-strain J2 kinematics explicit and uses compact
neural functions for hardening and recovery. The memory-channel count comes
from the manifest; the state size is `13 + 6 * channels`.

The published fixed identities used for acceptance are:

- model: `HaomingLuo/AgentFEM-DENIM` at
  `5629df0a23a3d1ed43e9de2150e3d33cb979fdc1`;
- dataset: `HaomingLuo/AgentFEM-Material-Loading-Memory` at
  `c84f416e5a71daa157e406c50afc3fc73509b9ca`.

Prepare those files explicitly, then migrate the trusted legacy state dict:

```python
from agentfem_learning.learned_constitutive.denim import convert_legacy_checkpoint

convert_legacy_checkpoint(
    "prepared/denim-expanded.pt",
    "models/denim-expanded",
    channels=2,
    model_revision="5629df0a23a3d1ed43e9de2150e3d33cb979fdc1",
    dataset_id="HaomingLuo/AgentFEM-Material-Loading-Memory",
    dataset_revision="c84f416e5a71daa157e406c50afc3fc73509b9ca",
)
```

The migration uses `torch.load(..., weights_only=True)` and writes only a
`safetensors` state dict. The formal runtime will not deserialize the legacy
checkpoint.

## Runtime and trust boundary

- default device: CPU;
- default precision: float64;
- one loaded model per bundle/device/dtype in a Step lifecycle;
- direct rank-local batch inference;
- fixed-old-state automatic-differentiation tangent;
- separate elastic, isotropic-hardening, kinematic-hardening, plastic-work,
  and modeled-dissipation channels plus yield, incompressibility, PEEQ,
  finite-state, and applicability diagnostics;
- no network access and no silent high-fidelity fallback.

The fixed 121-step reference path after bundle migration gives approximately
`292.547 MPa` maximum absolute stress and `0.00952709` final PEEQ. These values
are a software regression, not a declaration that every finite-element result
using the model is validated.

The current acceptance boundary is the framework-neutral material-point and
rank-local batch contract. Promotion to an implicit finite-element capability
requires a core AgentFEM Step provider, installed-wheel structural regression,
and serial/two-rank agreement; the companion does not infer that capability
from a passing tangent check alone.

## References

- [AgentFEM-DENIM fixed model revision](https://huggingface.co/HaomingLuo/AgentFEM-DENIM/tree/5629df0a23a3d1ed43e9de2150e3d33cb979fdc1)
- [AgentFEM material-loading-memory fixed dataset revision](https://huggingface.co/datasets/HaomingLuo/AgentFEM-Material-Loading-Memory/tree/c84f416e5a71daa157e406c50afc3fc73509b9ca)
- [PyTorch function transforms and Jacobians](https://docs.pytorch.org/docs/stable/func.api.html)
- [Safetensors format](https://huggingface.co/docs/safetensors/)
