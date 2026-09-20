"""Lower trusted AgentFEM Campaign fields into neural-operator datasets."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agentfem import datasets, learning


@dataclass(frozen=True)
class NeuralOperatorCampaignAdapter:
    """Bind one neural-operator contract to Campaign field extraction.

    The extractor remains problem-owned because only the scientific adapter
    knows how a solver result becomes the declared physical fields.  Campaign
    still owns accepted case identity and provenance; AgentFEM core owns field
    assembly; this companion validates the result against the operator spec.
    """

    specification: learning.NeuralOperatorSpec
    extract: Callable[[object, object], datasets.FieldCaseData | Mapping[str, object]]
    coordinate_names: tuple[str, ...] = ()
    name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.specification, learning.NeuralOperatorSpec):
            raise TypeError("specification must be an AgentFEM NeuralOperatorSpec.")
        if not callable(self.extract):
            raise TypeError("extract must be callable.")
        coordinate_names = tuple(str(item).strip() for item in self.coordinate_names)
        if any(not item for item in coordinate_names):
            raise ValueError("coordinate_names must not contain empty names.")
        if len(set(coordinate_names)) != len(coordinate_names):
            raise ValueError("coordinate_names must be unique.")
        object.__setattr__(self, "coordinate_names", coordinate_names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def scientific_contract(self) -> dict[str, object]:
        """Stable semantics used to reject incompatible later acquisitions."""

        record = {
            "kind": "neural_operator_campaign_fields",
            "operator_specification": self.specification.summary(),
            "coordinate_names": self.coordinate_names,
        }
        return _json_safe(record)

    def assemble(
        self,
        report,
        *,
        allow_partial: bool = False,
        minimum_cases: int = 1,
        minimum_trust_level: str | None = None,
        quality: str | None = None,
    ) -> datasets.ScientificFieldDataset:
        """Create and validate one operator dataset from accepted Campaign cases."""

        assembler = datasets.FieldDatasetAssembler(
            encodings=(*self.specification.inputs, *self.specification.outputs),
            extract=self.extract,
            parameter_names=self.specification.parameter_inputs,
            name=self.name,
            metadata={
                **self.metadata,
                "scientific_contract": self.scientific_contract,
            },
        )
        dataset = report.require_field_dataset(
            assembler,
            allow_partial=allow_partial,
            minimum_cases=minimum_cases,
            minimum_trust_level=minimum_trust_level,
            quality=quality,
        )
        self.validate(dataset)
        return dataset

    def validate(self, dataset) -> None:
        """Fail early when a dataset no longer represents this operator."""

        if not isinstance(dataset, datasets.ScientificFieldDataset):
            raise TypeError("dataset must be a ScientificFieldDataset.")
        expected_encodings = _json_safe(
            [
                item.summary()
                for item in (*self.specification.inputs, *self.specification.outputs)
            ]
        )
        if _json_safe(list(dataset.encodings)) != expected_encodings:
            raise ValueError("Dataset fields do not match the neural-operator specification.")
        missing_coordinates = set(self.coordinate_names).difference(dataset.coordinates)
        if missing_coordinates:
            raise ValueError(
                "Dataset is missing declared operator coordinates: "
                f"{tuple(sorted(missing_coordinates))!r}."
            )
        if set(dataset.parameters) != set(self.specification.parameter_inputs):
            raise ValueError(
                "Dataset parameters must match NeuralOperatorSpec.parameter_inputs exactly."
            )
        if dataset.metadata.get("scientific_contract") != self.scientific_contract:
            raise ValueError("Dataset scientific contract does not match this adapter.")


def _json_safe(value):
    return json.loads(json.dumps(value, sort_keys=True))


__all__ = ["NeuralOperatorCampaignAdapter"]
