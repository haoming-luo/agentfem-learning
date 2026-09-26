"""Validate DENIM on a nested non-proportional tension-torsion path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentfem import extensions, materials
from case import specification

from agentfem_learning.learned_constitutive.denim import (
    evaluate_nonproportional_path,
    run_nonproportional_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("denim-nonproportional-validation.json"),
    )
    args = parser.parse_args()
    extensions.load_extension("agentfem-learning.learned-constitutive")
    spec = specification(args.bundle.resolve())
    material = materials.learned(spec)
    record = run_nonproportional_validation(material)
    record["specification_fingerprint"] = spec.fingerprint
    record["runtime"] = material.implementation.runtime_evidence()
    record["acceptance"] = evaluate_nonproportional_path(record).summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not record["acceptance"]["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
