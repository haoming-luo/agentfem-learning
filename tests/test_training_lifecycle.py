from __future__ import annotations

import pytest

from agentfem_learning.training import TrainingEpoch, TrainingLedger


def test_training_ledger_keeps_best_epoch_and_round_trips_to_json(tmp_path):
    ledger = TrainingLedger(
        seed=12,
        device="cpu",
        dtype="float32",
        framework="reference",
        framework_version="1.0",
    )
    ledger.append(TrainingEpoch(1, 1.0, 1.2, 1.0e-3))
    ledger.append(TrainingEpoch(2, 0.5, 0.6, 1.0e-3))
    ledger.append(TrainingEpoch(3, 0.4, 0.8, 1.0e-3))
    path = ledger.write(tmp_path / "training.json")

    assert ledger.best_epoch == 2
    assert ledger.best.validation_loss == 0.6
    assert ledger.converged is True
    assert path.is_file()
    assert '"best_epoch": 2' in path.read_text(encoding="utf-8")


def test_training_ledger_rejects_nonfinite_and_noncontiguous_records():
    with pytest.raises(ValueError, match="finite"):
        TrainingEpoch(1, float("inf"), 1.0, 1.0e-3)
    ledger = TrainingLedger(1, "cpu", "float32", "reference", "1.0")
    ledger.append(TrainingEpoch(1, 1.0, 1.0, 1.0e-3))
    with pytest.raises(ValueError, match="contiguous"):
        ledger.append(TrainingEpoch(3, 0.5, 0.5, 1.0e-3))
