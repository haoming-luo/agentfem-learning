"""Named scientific checks for learned field operators."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256

from agentfem import verification


@dataclass(frozen=True)
class OperatorCheckContext:
    """Physical held-out data made available to one reviewed check."""

    specification: object
    training_dataset: object
    validation_dataset: object
    predictions: Mapping[str, object]
    references: Mapping[str, object]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class OperatorCheck:
    """A named, fingerprinted evaluator that returns one verification claim.

    The name is part of ``NeuralOperatorSpec.required_checks``.  Keeping the
    callable behind this explicit record lets a project provide physics-aware
    boundary, balance, or applicability checks without teaching the generic
    trainer the equations of every analysis family.
    """

    name: str
    evaluator: Callable[[OperatorCheckContext], verification.VerificationClaim]
    version: str = "1"
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        version = str(self.version).strip()
        if not name or not version:
            raise ValueError("OperatorCheck name and version must not be empty.")
        if not callable(self.evaluator):
            raise TypeError("OperatorCheck.evaluator must be callable.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def evaluate(
        self, context: OperatorCheckContext
    ) -> verification.VerificationClaim:
        claim = self.evaluator(context)
        if not isinstance(claim, verification.VerificationClaim):
            raise TypeError(
                f"Operator check {self.name!r} must return VerificationClaim."
            )
        if claim.name != self.name:
            raise ValueError(
                f"Operator check {self.name!r} returned claim {claim.name!r}."
            )
        return claim

    def summary(self) -> dict[str, object]:
        return {
            "kind": "operator_check",
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "evaluator": _callable_identity(self.evaluator),
            "metadata": dict(self.metadata),
        }


def _callable_identity(function) -> dict[str, object]:
    identity = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(
            function,
            "__qualname__",
            getattr(function, "__name__", None),
        ),
    }
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError):
        source = None
    if source is not None:
        identity["source_sha256"] = sha256(source).hexdigest()
    return identity


__all__ = ["OperatorCheck", "OperatorCheckContext"]
