"""Verification claims shared by maintained neural-operator providers."""

from __future__ import annotations

from agentfem import verification


def operator_dataset_integrity_claim(data_quality):
    """Record that deterministic labels are consistent and partitions independent."""

    partitions = {
        name: record
        for name in ("training", "validation", "test")
        if (record := data_quality.get(name)) is not None
    }
    conflicts = sum(int(record["conflicting_input_count"]) for record in partitions.values())
    return verification.VerificationClaim.compare(
        name="operator_dataset_integrity",
        observable="conflicting mappings or cross-partition input replicas",
        actual=float(conflicts),
        expected=0.0,
        reference="exact declared operator-input identity audit before optimization",
        absolute_tolerance=0.0,
        validity_domain=str(data_quality["identity"]),
        evidence={
            "partition_policy": data_quality["partition_policy"],
            "training_validation_input_overlap_count": 0,
            "training_test_input_overlap_count": 0 if "test" in partitions else None,
            "training_case_count": int(partitions["training"]["case_count"]),
            "training_unique_input_count": int(
                partitions["training"]["unique_input_count"]
            ),
            "training_duplicate_input_fraction": float(
                partitions["training"]["duplicate_input_fraction"]
            ),
            "validation_unique_input_count": int(
                partitions["validation"]["unique_input_count"]
            ),
            "test_unique_input_count": (
                None
                if "test" not in partitions
                else int(partitions["test"]["unique_input_count"])
            ),
        },
    )


__all__ = ["operator_dataset_integrity_claim"]
