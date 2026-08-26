from __future__ import annotations

import json

import pytest
import torch

from agentfem_learning.neural_fields.xdem import (
    WilliamsModeIIINetwork,
    mode_iii_tip_spec,
)
from agentfem_learning.neural_fields.xdem.reference import _resolve_device


def test_mode_iii_spec_is_machine_readable_and_explicitly_bounded():
    spec = mode_iii_tip_spec(domain_samples=256, boundary_samples=32)

    assert spec.metadata["provider"] == "agentfem-learning.xdem"
    assert spec.objective_kinds == ("energy",)
    assert spec.representations[0].architecture == "xdem:williams_mlp"
    assert spec.metadata["maturity"] == "experimental_reference"
    assert spec.integration.validation.independent_of == (
        "slit_annulus_energy_points",
    )
    assert spec.integration.refinements[0].count > spec.integration.validation.count
    geometry = spec.metadata["geometry"]
    assert geometry["cracks"]["tip_ids"] == [
        "branch_cut:start",
        "branch_cut:end",
    ]
    assert len(geometry["crack_fingerprint"]) == 64
    json.dumps(spec.summary())


def test_williams_representation_has_only_the_physical_branch_cut():
    network = WilliamsModeIIINetwork(
        radius=1.0,
        tip_core_radius=0.05,
        boundary_displacement=1.0,
        hidden_layers=(8,),
    )
    epsilon = 1.0e-8
    points = torch.tensor(
        [
            [0.5, epsilon],
            [0.5, -epsilon],
            [-0.5, epsilon],
            [-0.5, -epsilon],
        ],
        dtype=torch.float64,
    )
    values = network(points).detach().reshape(-1)

    intact_jump = torch.abs(values[0] - values[1])
    crack_jump = torch.abs(values[2] - values[3])
    assert intact_jump < 1.0e-5
    assert crack_jump > 1.0e-2


def test_device_policy_keeps_double_precision_on_cpu():
    assert _resolve_device("auto", dtype="float64").type == "cpu"
    with pytest.raises(ValueError, match="does not support float64"):
        _resolve_device("mps", dtype="float64")
