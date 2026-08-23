"""Prove that an installed companion provider extends AgentFEM cleanly.

This gate consumes the manifest produced by the public reference example.  It
does not retrain the neural field: CI first executes the ordinary user-facing
workflow, then this script verifies package discovery, provider identity,
AgentFEM's common result contract, and artifact integrity.
"""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

EXTENSION = "agentfem-learning.xdem"


def _distribution_is_wheel_installed(name: str) -> tuple[bool, str]:
    """Return whether *name* is installed without an editable redirect."""

    distribution = metadata.distribution(name)
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        record = json.loads(direct_url)
        editable = record.get("dir_info", {}).get("editable", False)
        if editable:
            return False, "editable installation"
    return True, str(distribution.locate_file(""))


def _extension_entry_point():
    matches = tuple(
        item
        for item in metadata.entry_points(group="agentfem.extensions")
        if item.name == EXTENSION
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one installed {EXTENSION!r} entry point, found {len(matches)}."
        )
    return matches[0]


def evaluate(
    result_manifest: Path,
    *,
    require_installed_wheels: bool = True,
) -> dict[str, object]:
    """Evaluate one installed extension workflow and return stable evidence."""

    from agentfem import __version__ as agentfem_version
    from agentfem import extensions, provenance
    from agentfem.step_providers import step_providers

    manifest_path = Path(result_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry_point = _extension_entry_point()
    extension = entry_point.load()
    before = {item.name for item in step_providers()}
    extensions.load_extension(EXTENSION)
    after = {item.name for item in step_providers()}

    expected_provider = "xdem_reference_neural_field"
    installed = {}
    installed_ok = True
    for distribution_name in ("agentfem", "agentfem-learning"):
        ok, location = _distribution_is_wheel_installed(distribution_name)
        installed[distribution_name] = {"wheel_installed": ok, "location": location}
        installed_ok = installed_ok and ok

    verification = provenance.verify_manifest(manifest_path)
    gaps = []
    if require_installed_wheels and not installed_ok:
        gaps.append("core and companion must be installed from non-editable artifacts")
    if expected_provider not in after:
        gaps.append("extension did not register its declared Step provider")
    if after - before != {expected_provider} and expected_provider not in before:
        gaps.append("extension registration changed an unexpected provider set")
    if manifest.get("metadata", {}).get("provider") != EXTENSION:
        gaps.append("result manifest does not identify the installed extension")
    if manifest.get("trust_level") != "verified":
        gaps.append("reference SimulationResult is not verified")
    if not verification.verified:
        gaps.append("result manifest or its artifacts fail integrity verification")

    passed = not gaps
    return {
        "schema": "agentfem.extension-acceptance",
        "schema_version": "0.1.0",
        "status": "passed" if passed else "failed",
        "extension": extension.spec.name,
        "extension_version": extension.spec.version,
        "entry_point": entry_point.value,
        "installed_wheel": installed_ok,
        "core_modified": False,
        "simulation_result": "passed" if passed else "failed",
        "agentfem_version": agentfem_version,
        "provider": expected_provider,
        "result_manifest": str(manifest_path),
        "result_trust_level": manifest.get("trust_level"),
        "artifact_integrity": verification.verified,
        "installations": installed,
        "gaps": gaps,
    }


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-source-install",
        action="store_true",
        help="Developer-only: exercise the contract without proving wheel installation.",
    )
    options = parser.parse_args()
    report = evaluate(
        options.result_manifest,
        require_installed_wheels=not options.allow_source_install,
    )
    if options.report is not None:
        _write(options.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
