"""Topology-aware quadrature for rectangular domains with straight cracks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

import numpy as np


class UnsupportedCutCellError(ValueError):
    """Raised when the declared sampling resolution cannot separate cracks."""

    code = "AFL-XDEM-CUT-CELL-001"


@dataclass(frozen=True)
class CutDomainQuadrature2D:
    """Area quadrature with explicit one-sided crack topology."""

    coordinates: np.ndarray
    weights: np.ndarray
    side_codes: np.ndarray
    cell_ids: np.ndarray
    cell_kinds: tuple[str, ...]
    crack_ids: tuple[str, ...]
    domain_area: float
    grid_shape: tuple[int, int]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        side_codes = np.asarray(self.side_codes, dtype=np.int8)
        cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n, 2).")
        if weights.shape != (len(coordinates),) or np.any(weights <= 0.0):
            raise ValueError("weights must be positive with shape (n,).")
        if side_codes.shape != (len(coordinates), len(self.crack_ids)):
            raise ValueError("side_codes must have one column per crack.")
        if np.any(~np.isin(side_codes, (-1, 1))):
            raise ValueError("Every quadrature point requires a one-sided crack code.")
        if cell_ids.shape != (len(coordinates),):
            raise ValueError("cell_ids must have shape (n,).")
        if len(self.cell_kinds) != len(coordinates):
            raise ValueError("cell_kinds must identify every quadrature point.")
        area = float(self.domain_area)
        if not np.isclose(weights.sum(), area, rtol=1.0e-12, atol=1.0e-14 * area):
            raise ValueError("Cut-domain quadrature weights must conserve domain area.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "side_codes", side_codes)
        object.__setattr__(self, "cell_ids", cell_ids)

    @property
    def fingerprint(self) -> str:
        digest = sha256()
        digest.update(np.ascontiguousarray(self.coordinates).tobytes())
        digest.update(np.ascontiguousarray(self.weights).tobytes())
        digest.update(np.ascontiguousarray(self.side_codes).tobytes())
        digest.update(
            json.dumps(
                {"crack_ids": self.crack_ids, "grid_shape": self.grid_shape},
                sort_keys=True,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        kinds = {kind: self.cell_kinds.count(kind) for kind in set(self.cell_kinds)}
        return {
            "kind": "straight_crack_cut_domain_quadrature_2d",
            "schema_version": "0.1.0",
            "point_count": len(self.coordinates),
            "grid_shape": self.grid_shape,
            "crack_ids": self.crack_ids,
            "domain_area": self.domain_area,
            "weight_sum": float(self.weights.sum()),
            "point_kinds": kinds,
            "fingerprint": self.fingerprint,
        }


def straight_crack_cut_quadrature(
    problem,
    count: int,
    *,
    variant: int = 0,
) -> CutDomainQuadrature2D:
    """Build a deterministic midpoint/cut-cell rule for straight cracks.

    Cells crossed away from a tip are clipped into the two crack half-planes.
    A cell containing a crack tip receives a symmetric 2x2 rule because the
    physical domain remains connected around that endpoint.  This distinction
    prevents a finite crack from being silently extended across its tip cell.
    """

    target = int(count)
    if target < 64:
        raise ValueError("Cut-domain quadrature requires at least 64 target points.")
    xmin, xmax, ymin, ymax = (float(item) for item in problem.domain.bounds)
    width = xmax - xmin
    height = ymax - ymin
    base_nx = max(8, round(sqrt(target * width / height)))
    nx = max(8, base_nx + ((int(variant) % 3) - 1 if variant else 0))
    ny = max(8, round(target / nx))
    dx = width / nx
    dy = height / ny
    cracks = tuple(problem.cracks.cracks)
    points: list[np.ndarray] = []
    weights: list[float] = []
    cell_ids: list[int] = []
    cell_kinds: list[str] = []
    tolerance = 1.0e-12 * max(width, height)

    for j in range(ny):
        cymin = ymin + j * dy
        cymax = cymin + dy
        for i in range(nx):
            cxmin = xmin + i * dx
            cxmax = cxmin + dx
            cell_id = j * nx + i
            bounds = (cxmin, cxmax, cymin, cymax)
            intersecting = [
                crack for crack in cracks if _segment_intersects_cell(crack, bounds, tolerance)
            ]
            if len(intersecting) > 1:
                raise UnsupportedCutCellError(
                    f"{UnsupportedCutCellError.code}: cell {cell_id} intersects "
                    "more than one crack; increase integration density."
                )
            if not intersecting:
                points.append(np.asarray(((cxmin + cxmax) / 2, (cymin + cymax) / 2)))
                weights.append(dx * dy)
                cell_ids.append(cell_id)
                cell_kinds.append("regular")
                continue
            crack = intersecting[0]
            if _cell_contains_tip(crack, bounds, tolerance):
                offset = 0.5 / sqrt(3.0)
                for xi in (-offset, offset):
                    for eta in (-offset, offset):
                        points.append(
                            np.asarray(
                                (
                                    (cxmin + cxmax) / 2 + xi * dx,
                                    (cymin + cymax) / 2 + eta * dy,
                                )
                            )
                        )
                        weights.append(dx * dy / 4.0)
                        cell_ids.append(cell_id)
                        cell_kinds.append("tip")
                continue
            polygon = np.asarray(
                ((cxmin, cymin), (cxmax, cymin), (cxmax, cymax), (cxmin, cymax)),
                dtype=float,
            )
            center = 0.5 * (np.asarray(crack.start) + np.asarray(crack.end))
            normal = np.asarray(crack.normal, dtype=float)
            pieces = (
                _clip_half_plane(polygon, center, normal, 1.0, tolerance),
                _clip_half_plane(polygon, center, normal, -1.0, tolerance),
            )
            nonempty = [(piece, _polygon_area_centroid(piece)) for piece in pieces if len(piece) >= 3]
            if len(nonempty) != 2 or any(item[1][0] <= tolerance**2 for item in nonempty):
                # The segment lies on a cell edge.  Adjacent cell midpoints
                # already sample the two physical sides without a cut-cell rule.
                points.append(np.asarray(((cxmin + cxmax) / 2, (cymin + cymax) / 2)))
                weights.append(dx * dy)
                cell_ids.append(cell_id)
                cell_kinds.append("aligned")
                continue
            for _, (area, centroid) in nonempty:
                points.append(centroid)
                weights.append(area)
                cell_ids.append(cell_id)
                cell_kinds.append("cut")

    expanded = _make_points_one_sided(
        points,
        weights,
        cell_ids,
        cell_kinds,
        cracks,
        tolerance=max(tolerance, 1.0e-10 * min(dx, dy)),
    )
    return CutDomainQuadrature2D(
        coordinates=expanded[0],
        weights=expanded[1],
        side_codes=expanded[2],
        cell_ids=expanded[3],
        cell_kinds=expanded[4],
        crack_ids=tuple(item.crack_id for item in cracks),
        domain_area=problem.domain.area,
        grid_shape=(nx, ny),
    )


def _segment_intersects_cell(crack, bounds, tolerance: float) -> bool:
    xmin, xmax, ymin, ymax = bounds
    start = np.asarray(crack.start, dtype=float)
    delta = np.asarray(crack.end, dtype=float) - start
    lower = 0.0
    upper = 1.0
    for origin, direction, minimum, maximum in (
        (start[0], delta[0], xmin, xmax),
        (start[1], delta[1], ymin, ymax),
    ):
        if abs(direction) <= tolerance:
            if origin < minimum - tolerance or origin > maximum + tolerance:
                return False
            continue
        first = (minimum - origin) / direction
        second = (maximum - origin) / direction
        entry, exit_ = min(first, second), max(first, second)
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if lower > upper + tolerance:
            return False
    return (
        upper >= -tolerance
        and lower <= 1.0 + tolerance
        and upper - lower > 1.0e-12
    )


def _cell_contains_tip(crack, bounds, tolerance: float) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return any(
        xmin + tolerance < point[0] < xmax - tolerance
        and ymin + tolerance < point[1] < ymax - tolerance
        for point in (crack.start, crack.end)
    )


def _clip_half_plane(polygon, origin, normal, side, tolerance):
    selected = []
    for current, following in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        current_value = side * float((current - origin) @ normal)
        following_value = side * float((following - origin) @ normal)
        current_inside = current_value >= -tolerance
        following_inside = following_value >= -tolerance
        if current_inside:
            selected.append(current)
        if current_inside != following_inside:
            fraction = current_value / (current_value - following_value)
            selected.append(current + fraction * (following - current))
    return np.asarray(selected, dtype=float)


def _polygon_area_centroid(polygon):
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    signed_area = 0.5 * cross.sum()
    area = abs(float(signed_area))
    if area == 0.0:
        return 0.0, polygon.mean(axis=0)
    centroid = np.asarray(
        (
            np.sum((polygon[:, 0] + following[:, 0]) * cross),
            np.sum((polygon[:, 1] + following[:, 1]) * cross),
        )
    ) / (6.0 * signed_area)
    return area, centroid


def _make_points_one_sided(points, weights, cell_ids, cell_kinds, cracks, *, tolerance):
    selected_points = []
    selected_weights = []
    selected_sides = []
    selected_cells = []
    selected_kinds = []
    for point, weight, cell_id, kind in zip(
        points, weights, cell_ids, cell_kinds, strict=True
    ):
        sides = []
        on_crack = []
        for index, crack in enumerate(cracks):
            center = 0.5 * (np.asarray(crack.start) + np.asarray(crack.end))
            relative = point - center
            along = float(relative @ np.asarray(crack.tangent))
            normal_distance = float(relative @ np.asarray(crack.normal))
            if abs(normal_distance) <= tolerance and abs(along) <= 0.5 * crack.length:
                on_crack.append(index)
            sides.append(1 if normal_distance >= 0.0 else -1)
        if len(on_crack) > 1:
            raise UnsupportedCutCellError(
                f"{UnsupportedCutCellError.code}: one integration point lies on "
                "multiple cracks; increase integration density."
            )
        if not on_crack:
            selected_points.append(point)
            selected_weights.append(weight)
            selected_sides.append(sides)
            selected_cells.append(cell_id)
            selected_kinds.append(kind)
            continue
        crack_index = on_crack[0]
        normal = np.asarray(cracks[crack_index].normal)
        for side in (-1, 1):
            shifted = point + side * tolerance * normal
            shifted_sides = list(sides)
            shifted_sides[crack_index] = side
            selected_points.append(shifted)
            selected_weights.append(0.5 * weight)
            selected_sides.append(shifted_sides)
            selected_cells.append(cell_id)
            selected_kinds.append("one_sided")
    return (
        np.asarray(selected_points),
        np.asarray(selected_weights),
        np.asarray(selected_sides, dtype=np.int8),
        np.asarray(selected_cells, dtype=np.int64),
        tuple(selected_kinds),
    )


__all__ = [
    "CutDomainQuadrature2D",
    "UnsupportedCutCellError",
    "straight_crack_cut_quadrature",
]
