from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from agentfem import extensions, models, provenance, studies
from agentfem.step_providers import step_providers

from agentfem_learning.neural_fields.xdem import mode_iii_tip_spec
from agentfem_learning.neural_fields.xdem.extension import extension
from agentfem_learning.neural_fields.xdem.reference import (
    ReferenceTrainingOptions,
    train_mode_iii_reference,
)


def _activate_extension():
    if any(item.name == "xdem_reference_neural_field" for item in step_providers()):
        return
    context = extensions.ExtensionContext(extension.spec)
    extension.register(context)
    context.commit()


def test_reference_training_is_repeatable_for_one_seed():
    spec = mode_iii_tip_spec(domain_samples=256, boundary_samples=32)
    options = ReferenceTrainingOptions(
        adam_epochs=20,
        lbfgs_steps=0,
        hidden_layers=(8,),
        seed=13,
    )

    first = train_mode_iii_reference(spec, options)
    second = train_mode_iii_reference(spec, options)

    np.testing.assert_allclose(first.losses, second.losses, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first.prediction, second.prediction, rtol=0.0, atol=0.0)


def test_neural_field_step_returns_verified_simulation_result(tmp_path, monkeypatch):
    _activate_extension()
    monkeypatch.chdir(tmp_path)
    output = Path("relative-result")
    study = studies.static_solid(
        dimension=2,
        assumption="plane_strain",
        name="mode_iii_neural_field",
    )
    model = models.create(study=study, name="mode_iii_reference")
    spec = mode_iii_tip_spec()

    step = model.step(
        target=spec,
        epochs=500,
        output=output,
        seed=2026,
    )
    result = step.solve_result()
    output = tmp_path / output

    assert result.trust_level == "verified"
    assert result.quantity("relative_l2_error") < 0.08
    assert result.quantity("relative_energy_error") < 0.10
    assert result.quantity("crack_jump_relative_error") < 0.10
    assert result.field("W") is None
    assert (output / "mode_iii_field.npz").is_file()
    assert (output / "model_state.pt").is_file()
    manifest_path = output / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["trust_level"] == "verified"
    assert manifest["metadata"]["provider"] == "agentfem-learning.xdem"
    assert manifest["artifacts"]["neural_field"] == "mode_iii_field.npz"
    assert provenance.verify_manifest(manifest_path).verified is True
    assert step.solve_result() is result
