"""Prepare the fixed published DENIM v1 checkpoint for offline execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentfem_learning.learned_constitutive.denim import (
    convert_legacy_checkpoint,
    load_denim_v1_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    contract = load_denim_v1_acceptance()
    identity = contract["identity"]
    bundle = convert_legacy_checkpoint(
        args.checkpoint,
        args.destination,
        channels=2,
        model_name=identity["model_name"],
        model_version=identity["model_version"],
        model_revision=identity["model_revision"],
        dataset_id=identity["dataset_repository"],
        dataset_revision=identity["dataset_revision"],
        expected_checkpoint_sha256=identity["legacy_checkpoint_sha256"],
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "bundle": str(bundle),
                "model_revision": identity["model_revision"],
                "dataset_revision": identity["dataset_revision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
