"""Published stress-intensity references and fail-closed comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, pi, sqrt

from agentfem import fracture

from .finite_domain import (
    point_displacement,
    rectangular_domain,
    spatial_displacement_bc,
    static_crack_problem,
    traction_bc,
    williams_displacement_field,
)


@dataclass(frozen=True)
class PublishedSIFReference2D:
    """Immutable published values for one stationary LEFM benchmark.

    Values are dimensionless unless ``normalization`` declares otherwise.
    The reference is deliberately separate from a solver result: registering a
    paper does not make a provider verified until ``compare`` accepts every
    declared crack tip.
    """

    key: str
    citation: str
    url: str
    expected: Mapping[str, tuple[float, float]]
    normalization: str
    geometry: Mapping[str, object]
    relative_tolerance: float = 0.05
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {
            str(tip_id): (float(pair[0]), float(pair[1]))
            for tip_id, pair in self.expected.items()
        }
        if not values or any(
            not isfinite(component)
            for pair in values.values()
            for component in pair
        ):
            raise ValueError("PublishedSIFReference2D requires finite tip values.")
        tolerance = float(self.relative_tolerance)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("relative_tolerance must lie in (0, 1).")
        object.__setattr__(self, "expected", values)
        object.__setattr__(self, "geometry", dict(self.geometry))
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def compare(self, reports, *, scale: float = 1.0) -> dict[str, object]:
        """Compare every named tip without averaging away a failed tip."""

        selected_scale = float(scale)
        if not isfinite(selected_scale) or selected_scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        rows = []
        for tip_id, expected in self.expected.items():
            report = reports.report(tip_id)
            actual = (float(report.k_i) / selected_scale, float(report.k_ii) / selected_scale)
            errors = tuple(
                abs(value - target) / max(abs(target), 1.0)
                for value, target in zip(actual, expected, strict=True)
            )
            rows.append(
                {
                    "tip_id": tip_id,
                    "expected_k_i": expected[0],
                    "expected_k_ii": expected[1],
                    "actual_k_i": actual[0],
                    "actual_k_ii": actual[1],
                    "relative_errors": errors,
                    "path_variation": float(report.path_variation),
                    "accepted": max(errors) <= self.relative_tolerance
                    and report.status == "accepted",
                }
            )
        return {
            "kind": "published_sif_comparison_2d",
            "reference": self.summary(),
            "status": "accepted" if all(row["accepted"] for row in rows) else "failed",
            "tips": rows,
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "published_sif_reference_2d",
            "key": self.key,
            "citation": self.citation,
            "url": self.url,
            "expected": dict(self.expected),
            "normalization": self.normalization,
            "geometry": dict(self.geometry),
            "relative_tolerance": self.relative_tolerance,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_summary(cls, summary: Mapping[str, object]):
        """Restore a reference embedded in a serializable scientific problem."""

        payload = dict(summary)
        expected = {
            name: tuple(values)
            for name, values in dict(payload["expected"]).items()
        }
        return cls(
            key=payload["key"],
            citation=payload["citation"],
            url=payload["url"],
            expected=expected,
            normalization=payload["normalization"],
            geometry=payload["geometry"],
            relative_tolerance=payload["relative_tolerance"],
            metadata=payload.get("metadata", {}),
        )


def xvem_mixed_mode_reference() -> PublishedSIFReference2D:
    """Published finite square mixed-mode patch with ``K_I=K_II=1``.

    Benvenuti et al. impose the exact near-tip field on the square boundary.
    The current internal-crack finite-domain provider uses this as the public
    single-tip promotion target; boundary-terminating geometry remains a
    separate capability gate rather than being silently approximated.
    """

    return PublishedSIFReference2D(
        key="benvenuti_chiozzi_manzini_sukumar_2022_mixed_mode",
        citation=(
            "E. Benvenuti, A. Chiozzi, G. Manzini and N. Sukumar, "
            "Extended virtual element method for two-dimensional linear "
            "elastic fracture, 2022, arXiv:2111.04150."
        ),
        url="https://arxiv.org/abs/2111.04150",
        expected={"main:end": (1.0, 1.0)},
        normalization="reported K_I and K_II",
        geometry={
            "domain": (-1.0, 1.0, -1.0, 1.0),
            "crack": ((-1.0, 0.0), (0.0, 0.0)),
            "assumption": "plane_strain",
            "young_modulus": 1.0e5,
            "poisson_ratio": 0.3,
        },
        relative_tolerance=0.05,
        metadata={"paper_section": "5.3", "promotion_role": "single_crack"},
    )


def xvem_mixed_mode_domain_problem(material=None):
    """Create the public boundary-terminating mixed-mode X-VEM problem.

    The left crack mouth is a stable, inactive boundary endpoint.  Only the
    interior endpoint is enriched and reported.  Exact mixed-mode displacement
    data are imposed on the four outer edges through the spatial hard-boundary
    contract; their interior extension is constructed by the provider and is
    not taken from the benchmark solution.
    """

    selected_material = material or fracture.linear_elastic_fracture_material(
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        assumption="plane_strain",
    )
    reference = xvem_mixed_mode_reference()
    boundary_field = williams_displacement_field(
        "mixed_mode_exact_boundary",
        tip=(0.0, 0.0),
        crack_angle=0.0,
        k_i=1.0,
        k_ii=1.0,
        metadata={
            "role": "published_boundary_data",
            "reference": reference.key,
            "interior_extension": "declared_field",
            "evidence_role": "extended_patch_test",
        },
    )
    return static_crack_problem(
        domain=rectangular_domain((-1.0, 1.0, -1.0, 1.0), name="square"),
        material=selected_material,
        cracks=fracture.crack_set(
            fracture.segment(
                "main",
                start=(-1.0, 0.0),
                end=(0.0, 0.0),
                metadata={
                    "active_ends": ("end",),
                    "start_role": "boundary_crack_mouth",
                },
            )
        ),
        conditions=(
            spatial_displacement_bc(
                "exact_outer_displacement",
                ("left", "right", "bottom", "top"),
                boundary_field,
            ),
        ),
        name="xvem_mixed_mode_finite_domain",
        metadata={
            "reference": reference.summary(),
            "reference_scale": 1.0,
            "benchmark_role": "public_single_active_tip_gate",
            "validation_class": "extended_patch_test",
        },
    )


def griffith_center_crack_reference() -> PublishedSIFReference2D:
    """Classical infinite-body Mode-I limit for a centered straight crack."""

    return PublishedSIFReference2D(
        key="griffith_center_crack_remote_tension",
        citation=(
            "A. A. Griffith, The phenomena of rupture and flow in solids, "
            "Philosophical Transactions of the Royal Society A, 1921."
        ),
        url="https://doi.org/10.1098/rsta.1921.0006",
        expected={"main:start": (1.0, 0.0), "main:end": (1.0, 0.0)},
        normalization="K_I / (sigma * sqrt(pi * a)); K_II identically zero",
        geometry={"kind": "center_crack_in_infinite_elastic_solid"},
        relative_tolerance=0.05,
        metadata={"promotion_role": "single_crack_domain_limit"},
    )


def center_crack_domain_problem(
    material,
    *,
    half_crack_length: float = 1.0,
    half_width: float = 12.0,
    half_height: float = 12.0,
    remote_stress: float = 1.0,
):
    """Create a traction-driven finite truncation of Griffith's crack.

    Two point gauges remove only rigid translation and rotation. Domain-size
    convergence remains mandatory before comparison to the infinite-body
    reference.
    """

    a = float(half_crack_length)
    width = float(half_width)
    height = float(half_height)
    stress = float(remote_stress)
    if min(a, width, height, stress) <= 0.0 or a >= width:
        raise ValueError(
            "Require positive geometry/stress and half_crack_length < half_width."
        )
    return static_crack_problem(
        domain=rectangular_domain((-width, width, -height, height), name="plate"),
        material=material,
        cracks=fracture.crack_set(
            fracture.segment("main", start=(-a, 0.0), end=(a, 0.0))
        ),
        conditions=(
            traction_bc("remote_top", "top", (0.0, stress)),
            traction_bc("remote_bottom", "bottom", (0.0, -stress)),
            point_displacement("origin_gauge", (-width, -height), (0.0, 0.0)),
            point_displacement("rotation_gauge", (width, -height), (None, 0.0)),
        ),
        name="griffith_center_crack_domain_limit",
        metadata={
            "reference": griffith_center_crack_reference().summary(),
            "reference_scale": stress * sqrt(pi * a),
        },
    )


def two_collinear_cracks_reference(a_over_l: float = 0.5) -> PublishedSIFReference2D:
    """Exact equal collinear-crack values tabulated by Liew et al.

    The table reports an infinite elastic solid under remote tension with two
    equal cracks of half-length ``a`` and half center spacing ``L``.  At
    ``a/L=0.5`` the exact normalized outer- and inner-tip factors are 1.04796
    and 1.02795, respectively.
    """

    ratio = float(a_over_l)
    if abs(ratio - 0.5) > 1.0e-12:
        raise ValueError("The bundled published table currently fixes a/L=0.5.")
    return PublishedSIFReference2D(
        key="liew_sun_kitipornchai_2007_two_collinear_a_over_l_0_5",
        citation=(
            "K. M. Liew, Y. Sun and S. Kitipornchai, Boundary element-free "
            "method for fracture analysis of 2-D anisotropic piezoelectric "
            "solids, Int. J. Numer. Meth. Engng 69 (2007) 729-749."
        ),
        url="https://doi.org/10.1002/nme.1786",
        expected={
            "left:start": (1.04796, 0.0),
            "left:end": (1.02795, 0.0),
            "right:start": (1.02795, 0.0),
            "right:end": (1.04796, 0.0),
        },
        normalization="K_I / (sigma * sqrt(pi * a)); K_II identically zero",
        geometry={
            "kind": "two_equal_collinear_cracks_in_infinite_elastic_solid",
            "a_over_L": ratio,
            "loading": "uniform_remote_mode_i_tension",
        },
        relative_tolerance=0.05,
        metadata={"published_table": "Table III", "promotion_role": "two_crack"},
    )


def two_collinear_cracks_domain_problem(
    material,
    *,
    half_crack_length: float = 1.0,
    half_center_spacing: float = 2.0,
    half_width: float = 20.0,
    half_height: float = 20.0,
    remote_stress: float = 1.0,
):
    """Create a finite truncation for four-tip interaction evidence."""

    a = float(half_crack_length)
    spacing = float(half_center_spacing)
    width = float(half_width)
    height = float(half_height)
    stress = float(remote_stress)
    if min(a, spacing, width, height, stress) <= 0.0:
        raise ValueError("Two-crack benchmark controls must be positive.")
    if spacing <= a or spacing + a >= width:
        raise ValueError("Cracks must be separated and strictly inside the domain.")
    reference = two_collinear_cracks_reference(a / spacing)
    return static_crack_problem(
        domain=rectangular_domain((-width, width, -height, height), name="plate"),
        material=material,
        cracks=fracture.crack_set(
            fracture.segment(
                "left", start=(-spacing - a, 0.0), end=(-spacing + a, 0.0)
            ),
            fracture.segment(
                "right", start=(spacing - a, 0.0), end=(spacing + a, 0.0)
            ),
        ),
        conditions=(
            traction_bc("remote_top", "top", (0.0, stress)),
            traction_bc("remote_bottom", "bottom", (0.0, -stress)),
            point_displacement("origin_gauge", (-width, -height), (0.0, 0.0)),
            point_displacement("rotation_gauge", (width, -height), (None, 0.0)),
        ),
        name="two_collinear_cracks_domain_limit",
        metadata={
            "reference": reference.summary(),
            "reference_scale": stress * sqrt(pi * a),
        },
    )


__all__ = [
    "PublishedSIFReference2D",
    "center_crack_domain_problem",
    "griffith_center_crack_reference",
    "two_collinear_cracks_domain_problem",
    "two_collinear_cracks_reference",
    "xvem_mixed_mode_domain_problem",
    "xvem_mixed_mode_reference",
]
