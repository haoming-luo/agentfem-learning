# Material memory and DENIM laboratory

This note explains the idea without assuming prior machine-learning or
plasticity training.

## 1. Why a material needs memory

An ideal spring responds only to its current deformation. Metal plasticity is
different: two samples at the same current strain can carry different stresses
because one has never yielded and the other has already been loaded, unloaded,
and reversed. The missing information is the material's **internal state**.

AgentFEM separates three things:

```text
current strain + old material state
                 |
                 v
          constitutive update
                 |
                 v
new stress + new state + tangent + diagnostics
```

The finite-element solver owns the global displacement equilibrium. The
material model owns the local stress update. The state container owns the
history and supports trial, commit, rollback, checkpoint, and restart.

## 2. What DENIM learns

DENIM keeps a recognizable small-strain J2-plasticity skeleton. Its compact
neural functions describe hardening, recovery, and memory effects inside that
structure. It is therefore not a black box that directly maps a mesh to a
finished answer.

The fixed v1 model receives:

- the new total strain;
- the previous internal state;
- Young's modulus, Poisson ratio, and initial yield stress.

It returns stress, plastic strain, equivalent plastic strain (PEEQ), two
memory channels, the previous flow direction, an automatic-differentiation
tangent, energy terms, and diagnostics. The number of memory channels is read
from the model manifest rather than hard-coded into AgentFEM.

## 3. Why trial, commit, and rollback matter

A Newton iteration tests a possible displacement before the global structure
has converged. Its material state is only a **trial**. If the iteration or load
increment fails, AgentFEM restores the last committed state. If it converges,
all integration points commit together. Without this transaction, a failed
iteration would silently age or plastically deform the material and corrupt
the next attempt.

## 4. From one point to a structure

The material-point example asks: “Given this strain history, what does one
small piece of material do?” The global bar asks: “Can many integration points
using the same update reach structural equilibrium through Newton iterations?”

The fixed material-point Golden checks exact identity and stable response. The
bar uses a `0.004` axial displacement so PEEQ must become positive; a smaller
purely elastic load would not prove that material memory was used.

## 5. Run the laboratory

Prepare the fixed model once:

```bash
hf download HaomingLuo/AgentFEM-DENIM denim-expanded.pt \
  --revision 5629df0a23a3d1ed43e9de2150e3d33cb979fdc1 \
  --local-dir prepared
python examples/learned_constitutive_denim/prepare_bundle.py \
  prepared/denim-expanded.pt models/denim-expanded
```

Run the point and structure:

```bash
python examples/learned_constitutive_denim/case.py \
  --bundle models/denim-expanded --output outputs/denim-point.json
python examples/learned_constitutive_denim/global_bar.py \
  --bundle models/denim-expanded --output outputs/denim-bar.json
python examples/learned_constitutive_denim/nonproportional_path.py \
  --bundle models/denim-expanded \
  --output outputs/denim-nonproportional.json
python examples/learned_constitutive_denim/global_cantilever_convergence.py \
  --bundle models/denim-expanded \
  --output outputs/denim-structural-convergence.json
```

An independently published coupon can be checked without pretending it is
training data:

```bash
python examples/learned_constitutive_denim/experimental_baseline.py \
  --bundle models/denim-expanded \
  --csv HEM320C-LP8_Specimen_1_processed_data.csv \
  --maximum-points 121 \
  --output outputs/denim-experimental-pilot.json
```

This coupon prescribes axial strain while the transverse and shear stresses
remain zero. That is different from artificially locking all lateral strains.
The command checks the source SHA-256 and writes bounded metrics by default;
add `--include-curves` only for a local diagnostic file.

Authenticate the separately published comparison receipt and combine all
reports:

```bash
python examples/learned_constitutive_denim/verify_acceptance.py \
  --material-point outputs/denim-point.json \
  --global-bar outputs/denim-bar.json \
  --nonproportional outputs/denim-nonproportional.json \
  --structural-convergence outputs/denim-structural-convergence.json \
  --published-evidence evidence/denim_v1/published_deployment_validation.json \
  --output outputs/denim-acceptance.json
```

## 6. How to read the result

- `accepted: true` for the point means the fixed asset and response match the
  software Golden.
- positive `maximum_peeq` in the bar proves that the global solve crossed into
  plastic flow.
- `applicability_counts` must contain only `in_domain` points.
- the non-proportional gate must retain exact path knots, converge under
  refinement, and transform covariantly after a rigid rotation.
- the structural gate checks converged reactions on both mesh and increment
  axes; its clamp-sensitive maximum PEEQ remains a diagnostic.
- the published receipt states comparison errors and its narrower reference
  scope.
- none of these checks proves experimental calibration for an arbitrary alloy
  or reliability under temperature, creep, fatigue, cracks, finite strain, or
  unseen long histories.
- the external S355J2+M pilot is intentionally rejected: the stress-history
  error is about 24.46% and the fixed asset exits its declared applicability
  domain before the reduced test ends.

That last distinction is part of the product: AgentFEM records what is known,
what passed, and what remains unproven instead of turning one successful run
into a broad claim.
