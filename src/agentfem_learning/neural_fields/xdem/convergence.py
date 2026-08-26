"""Memory-bounded multi-axis convergence evidence for finite-domain XDEM."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class XDEMConvergenceCase:
    axis: str
    level: str
    controls: Mapping[str, object]
    metrics: Mapping[str, float]
    tips: Mapping[str, Mapping[str, float]]

    def summary(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "level": self.level,
            "controls": dict(self.controls),
            "metrics": dict(self.metrics),
            "tips": {name: dict(values) for name, values in self.tips.items()},
        }


@dataclass(frozen=True)
class XDEMMultiAxisConvergenceReport:
    cases: tuple[XDEMConvergenceCase, ...]
    relative_tolerance: float = 0.08
    path_variation_tolerance: float = 0.08
    required_tip_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("XDEMMultiAxisConvergenceReport requires cases.")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(
            self, "required_tip_ids", tuple(str(item) for item in self.required_tip_ids)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(case.axis for case in self.cases))

    def axis_status(self, axis: str) -> dict[str, object]:
        selected = [case for case in self.cases if case.axis == str(axis)]
        if len(selected) < 2:
            return {"axis": axis, "status": "insufficient", "maximum_change": None}
        left, right = selected[-2:]
        changes = []
        common = set(left.tips).intersection(right.tips)
        expected = set(self.required_tip_ids) or set(left.tips).union(right.tips)
        for tip_id in common:
            for name in ("k_i", "k_ii", "j_integral"):
                a = float(left.tips[tip_id][name])
                b = float(right.tips[tip_id][name])
                changes.append(abs(b - a) / max(abs(a), abs(b), 1.0e-14))
        maximum = max(changes, default=float("inf"))
        maximum_path_variation = max(
            (
                float(case.tips[tip_id]["path_variation"])
                for case in (left, right)
                for tip_id in common
            ),
            default=float("inf"),
        )
        complete = common == expected and set(left.tips) == set(right.tips)
        accepted = (
            complete
            and maximum <= self.relative_tolerance
            and maximum_path_variation <= self.path_variation_tolerance
        )
        return {
            "axis": axis,
            "status": "accepted" if accepted else "uncertain",
            "maximum_change": maximum,
            "maximum_path_variation": maximum_path_variation,
            "compared_levels": (left.level, right.level),
            "tip_ids": tuple(sorted(common)),
            "required_tip_ids": tuple(sorted(expected)),
            "complete_tip_set": complete,
        }

    @property
    def status(self) -> str:
        states = [self.axis_status(axis)["status"] for axis in self.axes]
        return "accepted" if states and all(item == "accepted" for item in states) else "uncertain"

    def summary(self) -> dict[str, object]:
        return {
            "kind": "xdem_multi_axis_convergence",
            "schema_version": "0.1.0",
            "status": self.status,
            "relative_tolerance": self.relative_tolerance,
            "path_variation_tolerance": self.path_variation_tolerance,
            "required_tip_ids": self.required_tip_ids,
            "axes": [self.axis_status(axis) for axis in self.axes],
            "cases": [case.summary() for case in self.cases],
            "metadata": dict(self.metadata),
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output


def convergence_case(axis: str, level: str, controls, outcome) -> XDEMConvergenceCase:
    """Reduce one heavy training outcome to small immutable evidence."""

    tips = {
        item.tip_id: {
            "k_i": float(item.k_i),
            "k_ii": float(item.k_ii),
            "j_integral": float(item.j_integral),
            "path_variation": float(item.path_variation),
        }
        for item in outcome.stress_intensity.reports
    }
    return XDEMConvergenceCase(
        axis=str(axis),
        level=str(level),
        controls=dict(controls),
        metrics={name: float(value) for name, value in outcome.metrics.items()},
        tips=tips,
    )


def run_convergence_slices(
    slices: Mapping[str, tuple[tuple[str, Mapping[str, object]], ...]],
    *,
    runner: Callable[[Mapping[str, object]], object],
    relative_tolerance: float = 0.08,
    path_variation_tolerance: float = 0.08,
    required_tip_ids: tuple[str, ...] = (),
    metadata: Mapping[str, object] | None = None,
) -> XDEMMultiAxisConvergenceReport:
    """Run one-factor slices sequentially and discard heavy models promptly.

    ``runner`` owns how controls map to a provider invocation.  Only scalar
    metrics and per-tip reports survive each case, keeping memory approximately
    constant even when network, quadrature, seed, and integration-ring axes are
    audited together.
    """

    records = []
    for axis, levels in slices.items():
        for level, controls in levels:
            outcome = runner(dict(controls))
            records.append(convergence_case(axis, level, controls, outcome))
            del outcome
            gc.collect()
    report = XDEMMultiAxisConvergenceReport(
        tuple(records),
        relative_tolerance=float(relative_tolerance),
        path_variation_tolerance=float(path_variation_tolerance),
        required_tip_ids=tuple(required_tip_ids),
        metadata=metadata or {},
    )
    if not all(
        bool(np.isfinite(value))
        for case in report.cases
        for value in case.metrics.values()
    ):
        raise ValueError("Convergence evidence contains a non-finite metric.")
    return report


def run_finite_domain_convergence(
    spec,
    options,
    *,
    network_layers=((32, 32), (48, 48, 48)),
    integration_counts=(2048, 4096),
    seeds=(2026, 2027),
    ring_resolutions=((20, 80), (32, 128)),
    relative_tolerance: float = 0.08,
    path_variation_tolerance: float = 0.08,
    output: str | Path | None = None,
) -> XDEMMultiAxisConvergenceReport:
    """Run the four finite-domain XDEM promotion axes sequentially.

    The routine intentionally avoids case-level concurrency. One network is
    trained, reduced to immutable scalar/tip evidence, deleted, and collected
    before the next case starts. Campaign-level parallelism can be layered on
    later with an explicit memory budget.
    """

    from .finite_domain import finite_domain_spec
    from .finite_domain_solver import problem_from_spec, train_finite_domain

    problem = problem_from_spec(spec)
    boundary_count = next(
        item.count for item in spec.sampling if item.on == "boundary"
    )
    jobs: dict[tuple[str, str], tuple[object, object, dict[str, object]]] = {}
    slices: dict[str, tuple[tuple[str, Mapping[str, object]], ...]] = {}

    def register(axis, levels):
        records = []
        for label, controls, selected_spec, selected_options, tip_options in levels:
            key = (str(axis), str(label))
            jobs[key] = (selected_spec, selected_options, tip_options)
            records.append((str(label), {"axis": str(axis), **dict(controls)}))
        slices[str(axis)] = tuple(records)

    register(
        "network",
        (
            (
                "x".join(str(width) for width in layers),
                {"hidden_layers": tuple(layers)},
                spec,
                replace(options, hidden_layers=tuple(layers)),
                {},
            )
            for layers in network_layers
        ),
    )
    register(
        "integration",
        (
            (
                str(count),
                {"domain_samples": int(count)},
                finite_domain_spec(
                    problem,
                    domain_samples=int(count),
                    boundary_samples=boundary_count,
                ),
                options,
                {},
            )
            for count in integration_counts
        ),
    )
    register(
        "seed",
        (
            (
                str(seed),
                {"seed": int(seed)},
                spec,
                replace(options, seed=int(seed)),
                {},
            )
            for seed in seeds
        ),
    )
    register(
        "rings",
        (
            (
                f"{int(radial)}x{int(angular)}",
                {"radial_count": int(radial), "angular_count": int(angular)},
                spec,
                options,
                {"radial_count": int(radial), "angular_count": int(angular)},
            )
            for radial, angular in ring_resolutions
        ),
    )

    def runner(controls):
        key = (str(controls["axis"]), _level_for_controls(slices, controls))
        selected_spec, selected_options, tip_options = jobs[key]
        return train_finite_domain(
            selected_spec,
            selected_options,
            tip_plan_options=tip_options,
        )

    report = run_convergence_slices(
        slices,
        runner=runner,
        relative_tolerance=relative_tolerance,
        path_variation_tolerance=path_variation_tolerance,
        required_tip_ids=problem.tip_ids,
        metadata={
            "problem": problem.name,
            "scientific_fingerprint": problem.fingerprint,
            "execution_policy": "sequential_memory_bounded",
        },
    )
    if output is not None:
        report.write(output)
    return report


def _level_for_controls(slices, controls) -> str:
    axis = str(controls["axis"])
    for level, expected in slices[axis]:
        if dict(expected) == dict(controls):
            return str(level)
    raise KeyError(f"Unknown convergence controls for axis {axis!r}.")


__all__ = [
    "XDEMConvergenceCase",
    "XDEMMultiAxisConvergenceReport",
    "convergence_case",
    "run_convergence_slices",
    "run_finite_domain_convergence",
]
