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

Prepare those files explicitly, then authenticate and migrate the trusted
legacy state dict:

```bash
hf download HaomingLuo/AgentFEM-DENIM denim-expanded.pt \
  --revision 5629df0a23a3d1ed43e9de2150e3d33cb979fdc1 \
  --local-dir prepared
python examples/learned_constitutive_denim/prepare_bundle.py \
  prepared/denim-expanded.pt models/denim-expanded
```

The preparation command rejects any checkpoint whose SHA-256 differs from the
published fixed asset. The equivalent low-level Python API is:

```python
from agentfem_learning.learned_constitutive.denim import convert_legacy_checkpoint

convert_legacy_checkpoint(
    "prepared/denim-expanded.pt",
    "models/denim-expanded",
    channels=2,
    model_revision="5629df0a23a3d1ed43e9de2150e3d33cb979fdc1",
    dataset_id="HaomingLuo/AgentFEM-Material-Loading-Memory",
    dataset_revision="c84f416e5a71daa157e406c50afc3fc73509b9ca",
    expected_checkpoint_sha256="db9b7ee5425f50ef6fdbefb04757bd13797769da7ca022a0b4ad83033b304888",
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

The fixed 121-step reference path gives approximately `292.547 MPa` maximum
absolute stress and `0.00952709` final PEEQ. The fixed three-dimensional
symmetry-bar regression reaches plastic flow under `0.004` prescribed axial
displacement through AgentFEM's ordinary global Newton, quadrature-state and
result lifecycle. Both records are evaluated by the acceptance contract that
ships in the wheel.

Published comparison evidence is authenticated separately. It records
material-path and structural-reaction errors together with the exact scope of
the reference comparison. This separation is deliberate:

```text
fixed software Golden       -> is this exact integration still reproducible?
published scientific receipt -> what comparison evidence currently exists?
declared limitations         -> what has not been established?
```

Passing the first two questions does not declare every finite-element result
or every material represented by DENIM to be validated. See the
[material-memory laboratory](material_memory_and_denim_lab.md) for a
beginner-readable explanation.

## References

- [AgentFEM-DENIM fixed model revision](https://huggingface.co/HaomingLuo/AgentFEM-DENIM/tree/5629df0a23a3d1ed43e9de2150e3d33cb979fdc1)
- [AgentFEM material-loading-memory fixed dataset revision](https://huggingface.co/datasets/HaomingLuo/AgentFEM-Material-Loading-Memory/tree/c84f416e5a71daa157e406c50afc3fc73509b9ca)
- [PyTorch function transforms and Jacobians](https://docs.pytorch.org/docs/stable/func.api.html)
- [Safetensors format](https://huggingface.co/docs/safetensors/)
