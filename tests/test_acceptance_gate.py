from __future__ import annotations

from pathlib import Path

from agentfem import extensions, models, studies

from acceptance_gate import evaluate
from agentfem_learning.neural_fields.xdem import mode_iii_tip_spec


def test_extension_acceptance_consumes_common_result_contract(tmp_path: Path):
    extensions.load_extension("agentfem-learning.xdem")
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        name="acceptance_reference",
    )
    result = model.step(
        target=mode_iii_tip_spec(),
        epochs=500,
        output=tmp_path,
        seed=2026,
    ).solve_result()
    assert result.trust_level == "verified"

    report = evaluate(
        tmp_path / "result.json",
        require_installed_wheels=False,
    )
    assert report["status"] == "passed"
    assert report["core_modified"] is False
    assert report["simulation_result"] == "passed"
    assert report["artifact_integrity"] is True
