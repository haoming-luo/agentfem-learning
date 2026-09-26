"""Compare the fixed DENIM v1 asset with one independent coupon history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from agentfem import extensions, materials
from case import specification

from agentfem_learning.learned_constitutive import (
    prefix_uniaxial_history,
    read_uniaxial_csv,
    run_small_strain_material_history,
    simplify_uniaxial_history,
    truncate_to_strain_domain,
    uniaxial_stress_free_path,
    uniaxial_stress_metrics,
)
from agentfem_learning.learned_constitutive.provider import TORCH_CONSTITUTIVE_PROVIDER

SOURCE_IDENTIFIER = "doi:10.5281/zenodo.6965147"
SOURCE_REPOSITORY = "https://github.com/ahartloper/rlmtp"
SOURCE_REVISION = "6ad0092094d0de3d00cb573bbb8ff22d2fb338a7"
SOURCE_BLOB_SHA = "3ee1d5db5ed8501a8701b07d39f4a83d03623ab4"
SOURCE_PATH = "Examples/HEM320C-LP8_Specimen_1_processed_data.csv"
PILOT_SAMPLE_ID = "S355J2+M-HEM320-flange-LP8-S1"
PILOT_SHA256 = "db0cfeddfa2dd2ba9f1aee65a68163934bf213b8d16f59634d87df6f1ad40498"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experimental-baseline.json"))
    parser.add_argument("--maximum-points", type=int, default=181)
    parser.add_argument(
        "--include-curves",
        action="store_true",
        help="Include reduced measured/predicted arrays in the local result.",
    )
    args = parser.parse_args()

    measured = read_uniaxial_csv(
        args.csv,
        sample_id=PILOT_SAMPLE_ID,
        source_identifier=SOURCE_IDENTIFIER,
        license="CC-BY-4.0",
        coordinate_column="Time[s]",
        strain_column="e_true",
        stress_column="Sigma_true",
        stress_scale=1.0e6,
        metadata={
            "material": "S355J2+M",
            "source_product": "HEM320 flange",
            "load_protocol": "LP8",
            "data_role": "external_pilot_not_calibration_partition",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_blob_sha": SOURCE_BLOB_SHA,
            "source_path": SOURCE_PATH,
        },
    )
    if measured.source_sha256 != PILOT_SHA256:
        raise ValueError(
            "Pilot experimental checksum mismatch: expected "
            f"{PILOT_SHA256}, found {measured.source_sha256}."
        )
    bounded = truncate_to_strain_domain(measured, 0.025)
    reduced = simplify_uniaxial_history(
        bounded,
        maximum_points=args.maximum_points,
        geometric_tolerance=1.0e-4,
    )

    extensions.load_extension("agentfem-learning.learned-constitutive")
    spec = specification(args.bundle.resolve())
    material = materials.learned(spec)
    response = run_small_strain_material_history(
        material,
        uniaxial_stress_free_path(reduced),
        control_tolerance=1.0e-9,
        failure_policy="return_partial",
    )
    compared = prefix_uniaxial_history(reduced, response.path.point_count)
    predicted = response.stress[:, 0, 0]
    metrics = uniaxial_stress_metrics(compared, predicted)
    threshold = 0.05
    accepted = response.completed and metrics["stress_range_normalized_rmse"] <= threshold
    record = {
        "schema": "agentfem-learning.experimental-denim-baseline.v1",
        "status": "accepted" if accepted else "baseline_rejected",
        "promotion_status": "not_established",
        "promotion_reason": (
            "This is one independently published pilot coupon, not disjoint "
            "fit, validation and held-out sample partitions."
        ),
        "source": reduced.summary(),
        "compared_source_prefix": compared.summary(),
        "model": {
            "specification": spec.summary(),
            "runtime": TORCH_CONSTITUTIVE_PROVIDER.evidence(spec),
        },
        "control": {
            "kind": "axial_strain_with_zero_transverse_and_shear_stress",
            "maximum_stress_control_residual_pa": response.diagnostics[-1][
                "maximum_stress_control_residual"
            ],
            "completed": response.completed,
            "failure": response.failure,
        },
        "metrics": metrics,
        "acceptance": {
            "metric": "stress_range_normalized_rmse",
            "value": metrics["stress_range_normalized_rmse"],
            "maximum": threshold,
            "accepted": accepted,
        },
    }
    if args.include_curves:
        record["curves"] = {
            "predicted_axial_stress_pa": np.asarray(predicted).tolist(),
            "measured_axial_stress_pa": compared.stress.tolist(),
            "axial_strain": compared.strain.tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": record["schema"],
                "status": record["status"],
                "promotion_status": record["promotion_status"],
                "source": record["source"],
                "control": record["control"],
                "metrics": record["metrics"],
                "acceptance": record["acceptance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
