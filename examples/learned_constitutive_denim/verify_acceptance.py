"""Verify material, structure, and published DENIM evidence separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentfem_learning.learned_constitutive.denim import (
    evaluate_global_bar,
    evaluate_material_point,
    evaluate_nonproportional_path,
    evaluate_structural_convergence,
    verify_published_evidence,
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-point", type=Path)
    parser.add_argument("--global-bar", type=Path)
    parser.add_argument("--nonproportional", type=Path)
    parser.add_argument("--structural-convergence", type=Path)
    parser.add_argument("--published-evidence", type=Path)
    parser.add_argument("--output", type=Path, default=Path("denim-acceptance.json"))
    args = parser.parse_args()
    reports = []
    if args.material_point:
        reports.append(evaluate_material_point(_read(args.material_point)).summary())
    if args.global_bar:
        reports.append(evaluate_global_bar(_read(args.global_bar)).summary())
    if args.nonproportional:
        reports.append(evaluate_nonproportional_path(_read(args.nonproportional)).summary())
    if args.structural_convergence:
        reports.append(
            evaluate_structural_convergence(
                _read(args.structural_convergence)
            ).summary()
        )
    if args.published_evidence:
        reports.append(verify_published_evidence(args.published_evidence).summary())
    if not reports:
        parser.error("select at least one evidence input")
    record = {
        "schema": "agentfem-learning.denim-acceptance-suite.v1",
        "accepted": all(report["accepted"] for report in reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not record["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
