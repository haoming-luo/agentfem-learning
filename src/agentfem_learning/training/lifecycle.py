"""Small evidence records shared by AgentFEM-Learning providers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class TrainingEpoch:
    """One optimizer epoch reduced to portable scalar evidence."""

    epoch: int
    training_loss: float
    validation_loss: float
    learning_rate: float

    def __post_init__(self) -> None:
        if int(self.epoch) < 1:
            raise ValueError("TrainingEpoch.epoch must be positive.")
        for name in ("training_loss", "validation_loss", "learning_rate"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"TrainingEpoch.{name} must be finite and non-negative.")


@dataclass
class TrainingLedger:
    """Bounded training history and early-stopping decision evidence."""

    seed: int
    device: str
    dtype: str
    framework: str
    framework_version: str
    records: list[TrainingEpoch] = field(default_factory=list)
    best_epoch: int | None = None
    stopped_early: bool = False
    stop_reason: str | None = None

    def append(self, record: TrainingEpoch, *, minimum_delta: float = 0.0) -> bool:
        """Append one epoch and return whether it becomes the selected best."""

        if minimum_delta < 0.0:
            raise ValueError("minimum_delta must be non-negative.")
        if self.records and record.epoch != self.records[-1].epoch + 1:
            raise ValueError("Training epochs must be contiguous and increasing.")
        self.records.append(record)
        if self.best_epoch is None:
            self.best_epoch = record.epoch
            return True
        current_best = self.records[self.best_epoch - 1]
        if record.validation_loss < current_best.validation_loss - minimum_delta:
            self.best_epoch = record.epoch
            return True
        return False

    @property
    def best(self) -> TrainingEpoch:
        if self.best_epoch is None:
            raise RuntimeError("The training ledger has no epochs.")
        return self.records[self.best_epoch - 1]

    @property
    def converged(self) -> bool:
        return (
            bool(self.records) and self.best.validation_loss < self.records[0].validation_loss
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem-learning.training-ledger",
            "schema_version": "0.1.0",
            "seed": int(self.seed),
            "device": self.device,
            "dtype": self.dtype,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "epochs_completed": len(self.records),
            "best_epoch": self.best_epoch,
            "best_training_loss": None if not self.records else self.best.training_loss,
            "best_validation_loss": None if not self.records else self.best.validation_loss,
            "stopped_early": bool(self.stopped_early),
            "stop_reason": self.stop_reason,
            "converged": self.converged,
            "records": [asdict(item) for item in self.records],
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output


__all__ = ["TrainingEpoch", "TrainingLedger"]
