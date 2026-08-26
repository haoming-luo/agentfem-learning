"""Provider-neutral, per-tip interaction-integral reports for neural fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite, pi

import numpy as np
from agentfem import fracture


@dataclass(frozen=True)
class TipIntegrationPlan2D:
    """Validated interaction domains for every tip in a crack set."""

    crack_fingerprint: str
    bounds: tuple[float, float, float, float]
    radii_by_tip: Mapping[str, tuple[float, ...]]
    inner_radius_fraction: float = 0.3
    radial_count: int = 24
    angular_count: int = 96
    relative_path_tolerance: float = 0.03
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fingerprint = str(self.crack_fingerprint).strip()
        if not fingerprint:
            raise ValueError("crack_fingerprint must not be empty.")
        bounds = tuple(float(item) for item in self.bounds)
        if len(bounds) != 4 or any(not isfinite(item) for item in bounds):
            raise ValueError("bounds must contain four finite values.")
        if not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
            raise ValueError("bounds must satisfy xmin < xmax and ymin < ymax.")
        fraction = float(self.inner_radius_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("inner_radius_fraction must satisfy 0 < value < 1.")
        radial = int(self.radial_count)
        angular = int(self.angular_count)
        if radial < 4 or angular < 16:
            raise ValueError(
                "interaction domains require radial_count>=4 and angular_count>=16."
            )
        tolerance = float(self.relative_path_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("relative_path_tolerance must be finite and positive.")
        radii: dict[str, tuple[float, ...]] = {}
        for tip_id, values in self.radii_by_tip.items():
            selected = tuple(float(item) for item in values)
            if len(selected) < 2:
                raise ValueError("Every crack tip requires at least two integration radii.")
            if any(not isfinite(item) or item <= 0.0 for item in selected):
                raise ValueError("Integration radii must be finite and positive.")
            if any(right <= left for left, right in pairwise(selected)):
                raise ValueError("Integration radii must be strictly increasing.")
            radii[str(tip_id)] = selected
        if not radii:
            raise ValueError("TipIntegrationPlan2D requires at least one crack tip.")
        object.__setattr__(self, "crack_fingerprint", fingerprint)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "radii_by_tip", radii)
        object.__setattr__(self, "inner_radius_fraction", fraction)
        object.__setattr__(self, "radial_count", radial)
        object.__setattr__(self, "angular_count", angular)
        object.__setattr__(self, "relative_path_tolerance", tolerance)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tip_ids(self) -> tuple[str, ...]:
        return tuple(self.radii_by_tip)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "tip_interaction_domains_2d",
            "crack_fingerprint": self.crack_fingerprint,
            "bounds": self.bounds,
            "radii_by_tip": dict(self.radii_by_tip),
            "inner_radius_fraction": self.inner_radius_fraction,
            "radial_count": self.radial_count,
            "angular_count": self.angular_count,
            "relative_path_tolerance": self.relative_path_tolerance,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MultiTipStressIntensityReport2D:
    """One stable report collection for all selected crack tips."""

    crack_fingerprint: str
    reports: tuple[fracture.StressIntensityReport, ...]
    integration_plan: TipIntegrationPlan2D
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reports = tuple(self.reports)
        if not reports:
            raise ValueError("MultiTipStressIntensityReport2D requires reports.")
        identifiers = tuple(item.tip_id for item in reports)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Per-tip reports must have unique tip IDs.")
        expected = set(self.integration_plan.tip_ids)
        if set(identifiers) != expected:
            raise ValueError("Per-tip reports and integration plan must cover the same tips.")
        if self.crack_fingerprint != self.integration_plan.crack_fingerprint:
            raise ValueError("Crack and integration-plan fingerprints do not match.")
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def status(self) -> str:
        return (
            "accepted"
            if all(item.status == "accepted" for item in self.reports)
            else "uncertain"
        )

    def report(self, tip_id: str):
        selected = str(tip_id)
        for report in self.reports:
            if report.tip_id == selected:
                return report
        raise KeyError(f"Unknown crack-tip report {selected!r}.")

    def summary(self) -> dict[str, object]:
        return {
            "kind": "multi_tip_stress_intensity_report_2d",
            "schema_version": "0.1.0",
            "status": self.status,
            "crack_fingerprint": self.crack_fingerprint,
            "tip_ids": tuple(item.tip_id for item in self.reports),
            "reports": [item.summary() for item in self.reports],
            "integration_plan": self.integration_plan.summary(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CrackOpeningSIFReport2D:
    """Independent near-face SIF estimate from crack-opening displacement."""

    tip_id: str
    radii: tuple[float, ...]
    k_i_by_radius: tuple[float, ...]
    k_ii_by_radius: tuple[float, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def k_i(self) -> float:
        return _zero_radius_intercept(self.radii, self.k_i_by_radius)

    @property
    def k_ii(self) -> float:
        return _zero_radius_intercept(self.radii, self.k_ii_by_radius)

    @property
    def path_variation(self) -> float:
        values = np.column_stack((self.k_i_by_radius, self.k_ii_by_radius))
        coordinate = np.sqrt(np.asarray(self.radii, dtype=float))
        fitted = np.column_stack(
            (
                np.polyval(np.polyfit(coordinate, values[:, 0], 1), coordinate),
                np.polyval(np.polyfit(coordinate, values[:, 1], 1), coordinate),
            )
        )
        center = np.asarray((self.k_i, self.k_ii))
        spread = np.max(np.linalg.norm(values - fitted, axis=1))
        return float(spread / max(np.linalg.norm(center), 1.0e-30))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "crack_opening_sif_report_2d",
            "tip_id": self.tip_id,
            "K_I": self.k_i,
            "K_II": self.k_ii,
            "radii": self.radii,
            "K_I_by_radius": self.k_i_by_radius,
            "K_II_by_radius": self.k_ii_by_radius,
            "path_variation": self.path_variation,
            "extraction_method": "crack_opening_displacement_extrapolation",
            "metadata": dict(self.metadata),
        }


def tip_integration_plan(
    cracks: fracture.CrackSet2D,
    *,
    bounds,
    tip_ids=None,
    radius_fractions=(0.25, 0.4, 0.6),
    safety_factor: float = 0.45,
    inner_radius_fraction: float = 0.3,
    radial_count: int = 24,
    angular_count: int = 96,
    relative_path_tolerance: float = 0.03,
    metadata: Mapping[str, object] | None = None,
) -> TipIntegrationPlan2D:
    """Create non-overlapping, boundary-clear domains for selected crack tips."""

    if not isinstance(cracks, fracture.CrackSet2D):
        raise TypeError("cracks must be an AgentFEM CrackSet2D.")
    selected_ids = (
        tuple(item.tip_id for item in cracks.tips)
        if tip_ids is None
        else tuple(str(item) for item in tip_ids)
    )
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("tip_ids must contain unique crack-tip identities.")
    fractions = tuple(float(item) for item in radius_fractions)
    if len(fractions) < 2 or any(not 0.0 < item <= 1.0 for item in fractions):
        raise ValueError("radius_fractions require at least two values in (0, 1].")
    if any(right <= left for left, right in pairwise(fractions)):
        raise ValueError("radius_fractions must be strictly increasing.")
    selected_bounds = tuple(float(item) for item in bounds)
    radii = {}
    for tip_id in selected_ids:
        cracks.tip(tip_id)
        admissible = cracks.admissible_tip_radius(
            tip_id,
            bounds=selected_bounds,
            safety_factor=safety_factor,
        )
        radii[tip_id] = tuple(fraction * admissible for fraction in fractions)
    return TipIntegrationPlan2D(
        crack_fingerprint=cracks.fingerprint,
        bounds=selected_bounds,
        radii_by_tip=radii,
        inner_radius_fraction=inner_radius_fraction,
        radial_count=radial_count,
        angular_count=angular_count,
        relative_path_tolerance=relative_path_tolerance,
        metadata=metadata or {},
    )


def stress_intensity_reports(
    field: fracture.FractureField2D,
    *,
    cracks: fracture.CrackSet2D,
    material,
    plan: TipIntegrationPlan2D,
    metadata: Mapping[str, object] | None = None,
) -> MultiTipStressIntensityReport2D:
    """Extract ring-resolved ``K_I``, ``K_II``, and ``J`` for every tip.

    The neural or numerical provider supplies only global stress and
    displacement-gradient evaluations.  Geometry, local coordinate systems,
    admissible domains, interaction-integral normalization, and evidence are
    shared with AgentFEM's ordinary finite-element fracture postprocessor.
    """

    if not isinstance(cracks, fracture.CrackSet2D):
        raise TypeError("cracks must be an AgentFEM CrackSet2D.")
    if not isinstance(plan, TipIntegrationPlan2D):
        raise TypeError("plan must be a TipIntegrationPlan2D.")
    if plan.crack_fingerprint != cracks.fingerprint:
        raise ValueError("The integration plan belongs to a different crack set.")
    if not isinstance(field, fracture.FractureField2D):
        raise TypeError("field must implement displacement, displacement_gradient, and stress.")
    reports = tuple(
        _tip_report(
            field,
            cracks=cracks,
            material=material,
            plan=plan,
            tip_id=tip_id,
            metadata=metadata,
        )
        for tip_id in plan.tip_ids
    )
    return MultiTipStressIntensityReport2D(
        crack_fingerprint=cracks.fingerprint,
        reports=reports,
        integration_plan=plan,
        metadata={"provider": "agentfem-learning.xdem", **dict(metadata or {})},
    )


def crack_opening_sif_reports(
    field: fracture.FractureField2D,
    *,
    cracks: fracture.CrackSet2D,
    material,
    plan: TipIntegrationPlan2D,
    face_offset_fraction: float = 1.0e-6,
) -> tuple[CrackOpeningSIFReport2D, ...]:
    """Estimate every selected tip from two-sided near-face displacements.

    This is an independent diagnostic, not a replacement for the domain
    interaction integral. Agreement between the two methods is stronger
    evidence than either scalar estimate alone.
    """

    offset = float(face_offset_fraction)
    if not 0.0 < offset < 1.0e-2:
        raise ValueError("face_offset_fraction must lie in (0, 1e-2).")
    selected = material.summary()
    ratio = float(selected["poisson_ratio"])
    shear = float(selected["young_modulus"]) / (2.0 * (1.0 + ratio))
    kappa = 3.0 - 4.0 * ratio
    if selected["assumption"] == "plane_stress":
        kappa = (3.0 - ratio) / (1.0 + ratio)
    reports = []
    for tip_id in plan.tip_ids:
        tip = cracks.tip(tip_id)
        radii = plan.radii_by_tip[tip_id]
        extension = np.asarray(tip.extension_direction, dtype=float)
        normal = np.asarray(tip.normal, dtype=float)
        mode_i = []
        mode_ii = []
        for radius in radii:
            center = np.asarray(tip.point, dtype=float) - radius * extension
            epsilon = offset * radius
            points = np.stack((center + epsilon * normal, center - epsilon * normal))
            displacement = np.asarray(field.displacement(points), dtype=float)
            jump = displacement[0] - displacement[1]
            local_jump = np.asarray((jump @ extension, jump @ normal), dtype=float)
            factor = shear * np.sqrt(2.0 * pi / radius) / (kappa + 1.0)
            mode_ii.append(float(factor * local_jump[0]))
            mode_i.append(float(factor * local_jump[1]))
        reports.append(
            CrackOpeningSIFReport2D(
                tip_id=tip_id,
                radii=tuple(radii),
                k_i_by_radius=tuple(mode_i),
                k_ii_by_radius=tuple(mode_ii),
                metadata={
                    "face_offset_fraction": offset,
                    "coordinate_system": "tip_local_extension_normal",
                },
            )
        )
    return tuple(reports)


def _tip_report(field, *, cracks, material, plan, tip_id, metadata):
    tip = cracks.tip(tip_id)
    rotation = np.column_stack((tip.extension_direction, tip.normal))
    auxiliary_i = fracture.WilliamsField2D(tip, material, k_i=1.0)
    auxiliary_ii = fracture.WilliamsField2D(tip, material, k_ii=1.0)
    mode_i: list[float] = []
    mode_ii: list[float] = []
    for outer in plan.radii_by_tip[tip_id]:
        inner = plan.inner_radius_fraction * outer
        local, weights, q_gradient = _annulus_quadrature(
            inner,
            outer,
            radial_count=plan.radial_count,
            angular_count=plan.angular_count,
        )
        points = local @ rotation.T + np.asarray(tip.point, dtype=float)
        actual_stress = _tensor_to_local(field.stress(points), rotation)
        actual_gradient = _tensor_to_local(field.displacement_gradient(points), rotation)
        for auxiliary, collection in (
            (auxiliary_i, mode_i),
            (auxiliary_ii, mode_ii),
        ):
            samples = fracture.InteractionIntegralSamples2D(
                actual_stress=actual_stress,
                actual_displacement_gradient=actual_gradient,
                auxiliary_stress=_tensor_to_local(auxiliary.stress(points), rotation),
                auxiliary_displacement_gradient=_tensor_to_local(
                    auxiliary.displacement_gradient(points), rotation
                ),
                q_gradient=q_gradient,
                weights=weights,
                metadata={"tip_id": tip_id, "outer_radius": outer},
            )
            collection.append(fracture.interaction_integral(samples))
    return fracture.interaction_integral_report(
        crack=cracks,
        tip_id=tip_id,
        integration_radii=plan.radii_by_tip[tip_id],
        mode_i_integrals=mode_i,
        mode_ii_integrals=mode_ii,
        material=material,
        relative_path_tolerance=plan.relative_path_tolerance,
        metadata={
            "provider": "agentfem-learning.xdem",
            "field_protocol": "FractureField2D",
            "inner_radius_fraction": plan.inner_radius_fraction,
            "radial_count": plan.radial_count,
            "angular_count": plan.angular_count,
            **dict(metadata or {}),
        },
    )


def _annulus_quadrature(inner, outer, *, radial_count, angular_count):
    radial_fraction = (np.arange(radial_count, dtype=float) + 0.5) / radial_count
    radial = np.sqrt(inner * inner + radial_fraction * (outer * outer - inner * inner))
    angle = -pi + (np.arange(angular_count, dtype=float) + 0.5) * (2.0 * pi / angular_count)
    rr, tt = np.meshgrid(radial, angle, indexing="ij")
    local = np.stack((rr * np.cos(tt), rr * np.sin(tt)), axis=-1).reshape(-1, 2)
    distance = np.linalg.norm(local, axis=1)
    q_gradient = -local / distance[:, None] / (outer - inner)
    weights = np.full(len(local), pi * (outer * outer - inner * inner) / len(local))
    return local, weights, q_gradient


def _tensor_to_local(values, rotation):
    selected = np.asarray(values, dtype=float)
    return np.einsum("ai,nab,bj->nij", rotation, selected, rotation)


def _zero_radius_intercept(radii, values) -> float:
    coordinate = np.sqrt(np.asarray(radii, dtype=float))
    return float(np.polyfit(coordinate, np.asarray(values, dtype=float), 1)[1])


__all__ = [
    "CrackOpeningSIFReport2D",
    "MultiTipStressIntensityReport2D",
    "TipIntegrationPlan2D",
    "crack_opening_sif_reports",
    "stress_intensity_reports",
    "tip_integration_plan",
]
