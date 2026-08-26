"""Finite-domain scientific contracts for stationary two-dimensional XDEM-D.

The objects in this module describe the problem that a neural-field provider
must solve.  They deliberately contain no PyTorch architecture or optimizer
choice: domain, boundary conditions, material, and crack geometry are the
stable scientific inputs; a provider is free to lower them to an appropriate
representation after it has declared and verified that capability.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite

import numpy as np
from agentfem import fracture, learning

_BOUNDARIES = {"left", "right", "bottom", "top"}


class UnsupportedFiniteDomainError(ValueError):
    """Fail-closed error for finite-domain XDEM-D inputs."""

    code = "AFL-XDEM-DOMAIN-001"


@dataclass(frozen=True)
class RectangularDomain2D:
    """A named, axis-aligned finite two-dimensional domain."""

    bounds: tuple[float, float, float, float]
    name: str = "domain"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(float(item) for item in self.bounds)
        if len(values) != 4 or any(not isfinite(item) for item in values):
            raise ValueError("bounds must contain four finite values.")
        xmin, xmax, ymin, ymax = values
        if not xmin < xmax or not ymin < ymax:
            raise ValueError("bounds must satisfy xmin < xmax and ymin < ymax.")
        name = str(self.name).strip()
        if not name:
            raise ValueError("RectangularDomain2D.name must not be empty.")
        object.__setattr__(self, "bounds", values)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def area(self) -> float:
        xmin, xmax, ymin, ymax = self.bounds
        return (xmax - xmin) * (ymax - ymin)

    def boundary_points(self, boundary: str) -> tuple[tuple[float, float], ...]:
        """Return three points used for conservative rigid-mode auditing."""

        selected = _boundary_name(boundary)
        xmin, xmax, ymin, ymax = self.bounds
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        points = {
            "left": ((xmin, ymin), (xmin, ymid), (xmin, ymax)),
            "right": ((xmax, ymin), (xmax, ymid), (xmax, ymax)),
            "bottom": ((xmin, ymin), (xmid, ymin), (xmax, ymin)),
            "top": ((xmin, ymax), (xmid, ymax), (xmax, ymax)),
        }
        return points[selected]

    def strictly_contains(self, point, *, tolerance: float = 1.0e-10) -> bool:
        x, y = (float(item) for item in point)
        xmin, xmax, ymin, ymax = self.bounds
        margin = float(tolerance)
        return xmin + margin < x < xmax - margin and ymin + margin < y < ymax - margin

    def summary(self) -> dict[str, object]:
        return {
            "kind": "rectangle_2d",
            "name": self.name,
            "bounds": self.bounds,
            "area": self.area,
            "boundaries": tuple(sorted(_BOUNDARIES)),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorBoundaryCondition2D:
    """One constant vector displacement or traction on a named boundary.

    ``None`` leaves a displacement component unconstrained.  Tractions must
    provide both components, including explicit zeros.  General spatial or
    tabulated values will enter later through one common field/amplitude
    contract rather than by adding boundary-condition subclasses.
    """

    name: str
    kind: str
    boundary: str
    value: tuple[float | None, float | None]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("VectorBoundaryCondition2D.name must not be empty.")
        kind = str(self.kind).strip().lower()
        if kind not in {"displacement", "traction"}:
            raise ValueError("kind must be 'displacement' or 'traction'.")
        boundary = _boundary_name(self.boundary)
        if len(self.value) != 2:
            raise ValueError("value must contain the x and y components.")
        values = tuple(None if item is None else float(item) for item in self.value)
        if all(item is None for item in values):
            raise ValueError("At least one boundary-condition component is required.")
        if any(item is not None and not isfinite(item) for item in values):
            raise ValueError("Boundary-condition components must be finite.")
        if kind == "traction" and any(item is None for item in values):
            raise ValueError("Traction conditions require both vector components.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "value", values)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def constrained_components(self) -> tuple[int, ...]:
        if self.kind != "displacement":
            return ()
        return tuple(index for index, value in enumerate(self.value) if value is not None)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "boundary": self.boundary,
            "value": self.value,
            "components": ("x", "y"),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StaticXDEMProblem2D:
    """Provider-independent finite-domain, predefined-crack LEFM problem."""

    domain: RectangularDomain2D
    material: object
    cracks: fracture.CrackSet2D
    conditions: tuple[VectorBoundaryCondition2D, ...]
    name: str = "static_xdem_d"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.domain, RectangularDomain2D):
            raise TypeError("domain must be a RectangularDomain2D.")
        if not isinstance(self.cracks, fracture.CrackSet2D):
            raise TypeError("cracks must be an AgentFEM CrackSet2D.")
        if not hasattr(self.material, "summary"):
            raise TypeError("material must expose AgentFEM fracture-material semantics.")
        conditions = tuple(self.conditions)
        if not conditions or any(
            not isinstance(item, VectorBoundaryCondition2D) for item in conditions
        ):
            raise TypeError("conditions must contain VectorBoundaryCondition2D records.")
        names = [item.name for item in conditions]
        if len(set(names)) != len(names):
            raise UnsupportedFiniteDomainError(
                f"{UnsupportedFiniteDomainError.code}: condition names must be unique."
            )
        for crack in self.cracks.cracks:
            for label, point in (("start", crack.start), ("end", crack.end)):
                if not self.domain.strictly_contains(point, tolerance=self.cracks.tolerance):
                    raise UnsupportedFiniteDomainError(
                        f"{UnsupportedFiniteDomainError.code}: crack "
                        f"{crack.crack_id!r} {label} point must lie strictly inside "
                        "the finite domain; boundary-terminating and external cracks "
                        "are not silently approximated."
                    )
        for tip in self.cracks.tips:
            self.cracks.admissible_tip_radius(tip.tip_id, bounds=self.domain.bounds)
        _validate_condition_overlap(conditions)
        rank = _rigid_body_constraint_rank(self.domain, conditions)
        if rank != 3:
            raise UnsupportedFiniteDomainError(
                f"{UnsupportedFiniteDomainError.code}: displacement conditions "
                f"constrain only {rank}/3 planar rigid-body modes."
            )
        name = str(self.name).strip()
        if not name:
            raise ValueError("StaticXDEMProblem2D.name must not be empty.")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tip_ids(self) -> tuple[str, ...]:
        return tuple(item.tip_id for item in self.cracks.tips)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "static_xdem_d_2d",
            "schema_version": "0.1.0",
            "name": self.name,
            "domain": self.domain.summary(),
            "material": self.material.summary(),
            "cracks": self.cracks.summary(),
            "tip_ids": self.tip_ids,
            "conditions": [item.summary() for item in self.conditions],
            "rigid_body_constraint_rank": _rigid_body_constraint_rank(
                self.domain, self.conditions
            ),
            "metadata": dict(self.metadata),
        }


def rectangular_domain(
    bounds,
    *,
    name: str = "domain",
    metadata: Mapping[str, object] | None = None,
) -> RectangularDomain2D:
    """Construct a finite rectangular domain."""

    return RectangularDomain2D(tuple(bounds), name=name, metadata=metadata or {})


def displacement_bc(
    name: str,
    boundary: str,
    value,
    *,
    metadata: Mapping[str, object] | None = None,
) -> VectorBoundaryCondition2D:
    """Construct a component-aware constant displacement condition."""

    return VectorBoundaryCondition2D(
        name, "displacement", boundary, tuple(value), metadata or {}
    )


def traction_bc(
    name: str,
    boundary: str,
    value,
    *,
    metadata: Mapping[str, object] | None = None,
) -> VectorBoundaryCondition2D:
    """Construct a constant vector traction condition."""

    return VectorBoundaryCondition2D(name, "traction", boundary, tuple(value), metadata or {})


def static_crack_problem(
    *,
    domain: RectangularDomain2D,
    material,
    cracks: fracture.CrackSet2D,
    conditions,
    name: str = "static_xdem_d",
    metadata: Mapping[str, object] | None = None,
) -> StaticXDEMProblem2D:
    """Construct a fail-closed stationary predefined-crack problem."""

    return StaticXDEMProblem2D(
        domain=domain,
        material=material,
        cracks=cracks,
        conditions=tuple(conditions),
        name=name,
        metadata=metadata or {},
    )


def finite_domain_spec(
    problem: StaticXDEMProblem2D,
    *,
    domain_samples: int = 4096,
    boundary_samples: int = 256,
) -> learning.NeuralFieldSpec:
    """Lower a finite-domain problem into the common neural-field language.

    The companion lowers this contract through its experimental finite-domain
    provider.  Executability and external benchmark maturity remain separate:
    a result can run while its scientific evidence is still marked experimental.
    """

    if not isinstance(problem, StaticXDEMProblem2D):
        raise TypeError("problem must be a StaticXDEMProblem2D.")
    domain_count = int(domain_samples)
    boundary_count = int(boundary_samples)
    if domain_count < 64 or boundary_count < 16:
        raise ValueError("finite-domain sampling requires at least 64/16 points.")
    displacement = learning.FieldEncoding(
        name="displacement",
        role="output",
        unit="m",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
        components=("U1", "U2"),
    )
    conditions = tuple(
        learning.ConditionSpec(
            name=item.name,
            kind="boundary",
            target=displacement.name,
            on=item.boundary,
            value=item.value,
            enforcement="hard" if item.kind == "displacement" else "data",
            implementation=("agentfem_learning.neural_fields.xdem:finite_domain_boundary"),
            metadata={"physical_kind": item.kind, **dict(item.metadata)},
        )
        for item in problem.conditions
    )
    integration = learning.IntegrationPlan(
        training=learning.IntegrationRule(
            name="finite_domain_energy_points",
            role="training",
            strategy="xdem:domain_quadrature",
            count=domain_count,
            seed=2026,
            implementation=("agentfem_learning.neural_fields.xdem:finite_domain_quadrature"),
        ),
        validation=learning.IntegrationRule(
            name="finite_domain_validation_points",
            role="validation",
            strategy="xdem:independent_domain_quadrature",
            count=max(domain_count, 4096),
            seed=3026,
            independent_of=("finite_domain_energy_points",),
            implementation=("agentfem_learning.neural_fields.xdem:finite_domain_quadrature"),
        ),
        refinements=(
            learning.IntegrationRule(
                name="finite_domain_refined_points",
                role="refinement",
                strategy="xdem:refined_domain_quadrature",
                count=max(2 * domain_count, 8192),
                seed=4026,
                independent_of=(
                    "finite_domain_energy_points",
                    "finite_domain_validation_points",
                ),
                implementation=(
                    "agentfem_learning.neural_fields.xdem:finite_domain_quadrature"
                ),
            ),
        ),
        metadata={"purpose": "finite_domain_energy_consistency"},
    )
    return learning.NeuralFieldSpec(
        fields=(displacement,),
        objectives=(
            learning.ObjectiveTerm(
                name="internal_strain_energy",
                kind="energy",
                expression="integral(0.5 * epsilon:C:epsilon, domain)",
                dependent_fields=(displacement.name,),
                form="variational",
                measure="domain",
                unit="J/m",
                implementation=(
                    "agentfem_learning.neural_fields.xdem:finite_domain_internal_energy"
                ),
            ),
            learning.ObjectiveTerm(
                name="external_traction_work",
                kind="energy",
                expression="integral(traction dot displacement, traction_boundary)",
                dependent_fields=(displacement.name,),
                form="variational",
                measure="boundary",
                unit="J/m",
                coefficient=-1.0,
                implementation=(
                    "agentfem_learning.neural_fields.xdem:finite_domain_external_work"
                ),
            ),
        ),
        conditions=conditions,
        representations=(
            learning.NeuralRepresentation(
                name="multi_crack_xdem_d_displacement",
                fields=(displacement.name,),
                architecture="xdem:finite_domain_vector_field",
                features=("coordinates", "crack_jump_functions"),
                enrichments=tuple(f"xdem:williams_tip:{tip_id}" for tip_id in problem.tip_ids),
                implementation=(
                    "agentfem_learning.neural_fields.xdem:FiniteDomainVectorNetwork"
                ),
            ),
        ),
        sampling=(
            learning.SamplingPlan(
                name="finite_domain_energy_points",
                on="domain",
                strategy="xdem:uniform_or_gauss",
                count=domain_count,
                seed=2026,
            ),
            learning.SamplingPlan(
                name="finite_domain_boundary_points",
                on="boundary",
                strategy="uniform",
                count=boundary_count,
                seed=2027,
            ),
        ),
        integration=integration,
        purpose="forward",
        required_checks=(
            "condition_error",
            "energy_consistency",
            "crack_jump_evidence",
            "per_tip_stress_intensity",
            "stress_intensity_path_variation",
            "optimization_repeatability",
        ),
        metadata={
            "provider": "agentfem-learning.xdem",
            "problem": "finite_domain_static_xdem_d",
            "scientific_problem": problem.summary(),
            "scientific_fingerprint": problem.fingerprint,
            "maturity": "experimental_solver",
            "executable": True,
            "external_benchmark_verified": False,
        },
    )


def _boundary_name(value: str) -> str:
    selected = str(value).strip().lower()
    if selected not in _BOUNDARIES:
        raise UnsupportedFiniteDomainError(
            f"{UnsupportedFiniteDomainError.code}: boundary must be one of "
            f"{tuple(sorted(_BOUNDARIES))!r}; received {selected!r}."
        )
    return selected


def _validate_condition_overlap(conditions) -> None:
    occupied: dict[tuple[str, int], str] = {}
    for condition in conditions:
        components = (
            condition.constrained_components
            if condition.kind == "displacement"
            else tuple(index for index, value in enumerate(condition.value) if value != 0.0)
        )
        for component in components:
            key = (condition.boundary, component)
            if key in occupied:
                raise UnsupportedFiniteDomainError(
                    f"{UnsupportedFiniteDomainError.code}: conditions "
                    f"{occupied[key]!r} and {condition.name!r} both prescribe "
                    f"boundary {key[0]!r}, component {key[1]}."
                )
            occupied[key] = condition.name


def _rigid_body_constraint_rank(domain, conditions) -> int:
    rows: list[tuple[float, float, float]] = []
    for condition in conditions:
        for component in condition.constrained_components:
            for x, y in domain.boundary_points(condition.boundary):
                rows.append((1.0, 0.0, -y) if component == 0 else (0.0, 1.0, x))
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.asarray(rows, dtype=float), tol=1.0e-12))


__all__ = [
    "RectangularDomain2D",
    "StaticXDEMProblem2D",
    "UnsupportedFiniteDomainError",
    "VectorBoundaryCondition2D",
    "displacement_bc",
    "finite_domain_spec",
    "rectangular_domain",
    "static_crack_problem",
    "traction_bc",
]
