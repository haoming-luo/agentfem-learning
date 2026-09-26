# DENIM v1 acceptance evidence

This directory preserves the small published comparison receipt used by the
installed DENIM acceptance contract. Its byte-level SHA-256 is checked before
the metrics are consumed.

Source:

- repository: `HaomingLuo/AgentFEM-DENIM`;
- revision: `5629df0a23a3d1ed43e9de2150e3d33cb979fdc1`;
- path: `artifacts/deployment_validation.json`;
- SHA-256: `f04ca496825ec0c84c11cc284d640aa0e195a21c89cd262065ddaddb9fedd2dc`.

This is supporting scientific evidence with a declared comparison scope. The
material-point and global-bar Goldens are separate software regressions. None
of these records turns DENIM into a universally validated material model.

Two locally regenerated, fixed-asset receipts extend that evidence:

- `nonproportional_validation.json` records nested increment refinement,
  rigid-rotation covariance and constitutive residuals on an independently
  published loading-path topology;
- `structural_convergence.json` records three mesh levels and three increment
  levels for a three-dimensional plastic cantilever, sharing one reference
  case so only five unique solves are required.

The non-proportional source describes OFHC copper, whereas the fixed DENIM
asset has different material parameters. The source is therefore used only
for path topology. Neither receipt is presented as experimental calibration.

`experimental_pilot.json` is the first comparison with an independently
published laboratory coupon. It uses Hartloper et al.'s S355J2+M LP8 history,
identified by DOI, repository revision, Git blob and file SHA-256. The local
mixed-control solve reproduces the coupon boundary semantics to sub-pascal
stress residual, but the fixed synthetic DENIM v1 asset fails the declared 5%
stress-history threshold and leaves its applicability domain before completing
the reduced experiment. The negative result is retained deliberately: DENIM
v1 is **not** promoted as a calibrated S355J2+M material.

The repository stores only bounded metrics and identities, not the third-party
experimental curve. Users obtain the data from its published source under
CC BY 4.0 and can regenerate the receipt with
`examples/learned_constitutive_denim/experimental_baseline.py`.
