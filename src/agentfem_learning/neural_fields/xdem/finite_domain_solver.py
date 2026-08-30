"""Experimental finite-domain, multi-crack XDEM-D energy solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from agentfem import fracture

from .cut_domain import straight_crack_cut_quadrature
from .finite_domain import (
    PointDisplacementCondition2D,
    SpatialDisplacementCondition2D,
    SpatialVectorField2D,
    StaticXDEMProblem2D,
    displacement_bc,
    point_displacement,
    rectangular_domain,
    spatial_displacement_bc,
    static_crack_problem,
    traction_bc,
)
from .reference import ReferenceTrainingOptions, _resolve_device
from .tip_reports import (
    crack_opening_sif_reports,
    stress_intensity_reports,
    tip_integration_plan,
)
from .vector_reference import (
    TorchVectorFractureField,
    _elastic_state,
    williams_vector_field,
)


def _westergaard_center_crack_displacement(
    coordinates: torch.Tensor,
    *,
    young_modulus: float,
    poisson_ratio: float,
    assumption: str,
    center,
    half_crack_length: float,
    remote_stress: float,
) -> torch.Tensor:
    """Exact infinite-plate Mode-I displacement in Cartesian coordinates.

    This is the closed form written in terms of the Westergaard complex
    potential.  Signed square roots select the physical two sheets: horizontal
    displacement is odd in ``x`` and crack opening is odd in ``y``.
    """

    modulus = float(young_modulus)
    poisson = float(poisson_ratio)
    crack_length = float(half_crack_length)
    stress = float(remote_stress)
    if modulus <= 0.0 or crack_length <= 0.0 or stress <= 0.0:
        raise ValueError("Westergaard material, crack length, and stress must be positive.")
    selected_assumption = str(assumption).strip().lower()
    if selected_assumption == "plane_stress":
        kappa = (3.0 - poisson) / (1.0 + poisson)
    elif selected_assumption == "plane_strain":
        kappa = 3.0 - 4.0 * poisson
    else:
        raise ValueError("Westergaard displacement requires plane_stress or plane_strain.")

    origin = torch.as_tensor(center, dtype=coordinates.dtype, device=coordinates.device)
    relative = (coordinates - origin) / crack_length
    x = relative[:, 0]
    y = relative[:, 1]
    z = torch.complex(x, y)
    # sqrt(z-1)*sqrt(z+1) has the finite segment [-1, 1] as its branch cut and
    # approaches z, rather than |z|, at infinity.  It avoids the artificial
    # axis cusps produced by reconstructing the branch from sign functions.
    root = torch.sqrt(z - 1.0) * torch.sqrt(z + 1.0)
    signed_c = np.sqrt(2.0) * root.real
    signed_d = np.sqrt(2.0) * root.imag
    b_value = root.real.square() + root.imag.square()
    epsilon = torch.as_tensor(
        1.0e-24, dtype=coordinates.dtype, device=coordinates.device
    )
    safe_b = torch.clamp(b_value, min=epsilon)
    alpha = 0.5 * (kappa - 1.0)
    beta = 0.5 * (kappa + 1.0)
    prefactor = stress * crack_length * (1.0 + poisson) / (
        np.sqrt(2.0) * modulus
    )
    displacement_x = prefactor * (
        alpha * signed_c
        - (y.square() * signed_c - x * y * signed_d) / safe_b
    )
    displacement_y = prefactor * (
        beta * signed_d
        - (y.square() * signed_d + x * y * signed_c) / safe_b
    )
    return torch.stack((displacement_x, displacement_y), dim=1)


def _complete_point_gauge(problem, *, dtype):
    """Return an exact three-mode rigid-motion gauge when one is declared.

    Point gauges used by remote-traction benchmarks are kinematic references,
    not physical clamps.  Enforcing them with a penalty adds artificial local
    strain energy.  A planar rigid motion has three coefficients, so three
    independent scalar point conditions can instead be imposed exactly by a
    differentiable zero-strain projection.
    """

    rows = []
    points = []
    components = []
    values = []
    center = np.asarray(
        (
            0.5 * (problem.domain.bounds[0] + problem.domain.bounds[1]),
            0.5 * (problem.domain.bounds[2] + problem.domain.bounds[3]),
        ),
        dtype=float,
    )
    for condition in problem.conditions:
        if not isinstance(condition, PointDisplacementCondition2D):
            continue
        point = np.asarray(condition.point, dtype=float)
        relative = point - center
        for component in condition.constrained_components:
            row = (
                (1.0, 0.0, -relative[1])
                if component == 0
                else (0.0, 1.0, relative[0])
            )
            rows.append(row)
            points.append(condition.point)
            components.append(component)
            values.append(condition.value[component])
    if len(rows) != 3 or np.linalg.matrix_rank(np.asarray(rows, dtype=float)) != 3:
        return None
    return (
        torch.tensor(points, dtype=dtype),
        torch.tensor(components, dtype=torch.long),
        torch.tensor(values, dtype=dtype),
        torch.tensor(rows, dtype=dtype),
    )


def _initial_tip_amplitudes(problem, *, stress_scale: float, dtype):
    """Build a load-derived LEFM preconditioner for every crack tip.

    Constant boundary tractions determine a least-squares nominal symmetric
    stress tensor.  Its normal and shear tractions on each crack plane provide
    dimensional ``sigma*sqrt(pi*a)`` starting values.  They are only optimizer
    initial conditions; no published SIF or benchmark target enters this path.
    """

    spatial_fields = tuple(
        condition.field
        for condition in problem.conditions
        if isinstance(condition, SpatialDisplacementCondition2D)
    )
    if spatial_fields:
        westergaard_fields = tuple(
            item
            for item in spatial_fields
            if item.family == "westergaard_center_crack_displacement"
        )
        if len(westergaard_fields) == 1:
            field = westergaard_fields[0]
            if field.metadata.get("interior_extension") == "declared_field":
                return torch.zeros(
                    (len(problem.active_tips), 2), dtype=dtype
                )
            parameters = field.parameters
            center = np.asarray(parameters["center"], dtype=float)
            half_length = float(parameters["half_crack_length"])
            amplitudes = []
            for tip in problem.active_tips:
                crack = problem.cracks.crack(tip.crack_id)
                crack_center = 0.5 * (
                    np.asarray(crack.start, dtype=float)
                    + np.asarray(crack.end, dtype=float)
                )
                scale = float(stress_scale) * np.sqrt(crack.length)
                if not np.allclose(crack_center, center, rtol=0.0, atol=problem.cracks.tolerance) or not np.isclose(
                    0.5 * crack.length,
                    half_length,
                    rtol=1.0e-12,
                    atol=problem.cracks.tolerance,
                ):
                    amplitudes.append((0.0, 0.0))
                    continue
                k_i = float(parameters["remote_stress"]) * np.sqrt(
                    np.pi * half_length
                )
                amplitudes.append((k_i / scale, 0.0))
            return torch.tensor(amplitudes, dtype=dtype)
        amplitudes = []
        for tip in problem.active_tips:
            crack = problem.cracks.crack(tip.crack_id)
            scale = float(stress_scale) * np.sqrt(crack.length)
            matching = tuple(
                field
                for field in spatial_fields
                if field.family == "williams_displacement"
                and np.linalg.norm(
                    np.asarray(field.parameters["tip"], dtype=float)
                    - np.asarray(tip.point, dtype=float)
                )
                <= problem.cracks.tolerance
            )
            if len(matching) != 1:
                amplitudes.append((0.0, 0.0))
                continue
            field = matching[0]
            if field.metadata.get("interior_extension") == "declared_field":
                amplitudes.append((0.0, 0.0))
                continue
            amplitudes.append(
                (
                    float(field.parameters["k_i"]) / scale,
                    float(field.parameters["k_ii"]) / scale,
                )
            )
        return torch.tensor(amplitudes, dtype=dtype)

    boundary_normals = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "bottom": (0.0, -1.0),
        "top": (0.0, 1.0),
    }
    matrix = []
    values = []
    for condition in problem.conditions:
        if condition.kind != "traction":
            continue
        nx, ny = boundary_normals[condition.boundary]
        matrix.extend(((nx, 0.0, ny), (0.0, ny, nx)))
        values.extend(condition.value)
    if not matrix:
        return torch.zeros((len(problem.active_tips), 2), dtype=dtype)
    nominal_values, *_ = np.linalg.lstsq(
        np.asarray(matrix, dtype=float), np.asarray(values, dtype=float), rcond=None
    )
    nominal = np.asarray(
        (
            (nominal_values[0], nominal_values[2]),
            (nominal_values[2], nominal_values[1]),
        )
    )
    amplitudes = []
    for tip in problem.active_tips:
        crack = problem.cracks.crack(tip.crack_id)
        tangent = np.asarray(tip.extension_direction, dtype=float)
        normal = np.asarray(tip.normal, dtype=float)
        factor = np.sqrt(np.pi * 0.5 * crack.length)
        k_i = max(float(normal @ nominal @ normal), 0.0) * factor
        k_ii = float(tangent @ nominal @ normal) * factor
        scale = float(stress_scale) * np.sqrt(crack.length)
        amplitudes.append((k_i / scale, k_ii / scale))
    return torch.tensor(amplitudes, dtype=dtype)


_ADDITIVE_JUMP_REPRESENTATION = "additive_jump"
_PUBLISHED_CRACK_COORDINATE_REPRESENTATION = "published_crack_coordinate"
_BOUNDED_SHEET_COORDINATE_REPRESENTATION = "bounded_sheet_coordinate"
_RIEMANN_SHEET_COORDINATE_REPRESENTATION = "riemann_sheet_coordinate"
_SUPPORTED_REPRESENTATIONS = {
    _ADDITIVE_JUMP_REPRESENTATION,
    _PUBLISHED_CRACK_COORDINATE_REPRESENTATION,
    _BOUNDED_SHEET_COORDINATE_REPRESENTATION,
    _RIEMANN_SHEET_COORDINATE_REPRESENTATION,
}
_CRACK_COORDINATE_REPRESENTATIONS = {
    _PUBLISHED_CRACK_COORDINATE_REPRESENTATION,
    _BOUNDED_SHEET_COORDINATE_REPRESENTATION,
    _RIEMANN_SHEET_COORDINATE_REPRESENTATION,
}


def _vector_mlp(hidden_layers, *, input_width: int = 2, dtype):
    widths = (int(input_width), *(int(item) for item in hidden_layers), 2)
    layers: list[torch.nn.Module] = []
    for layer_input_width, output_width in pairwise(widths):
        layer = torch.nn.Linear(layer_input_width, output_width, dtype=dtype)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.3)
        torch.nn.init.zeros_(layer.bias)
        layers.append(layer)
        if output_width != 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


class FiniteDomainVectorNetwork(torch.nn.Module):
    """Vector XDEM field with crack coordinates and explicit tip bases.

    The default regression representation keeps continuous and jump channels
    additive.  A benchmark can instead request the published XDEM coordinate
    form, in which one bounded crack coordinate per segment is appended to the
    neural-network input.  Both forms confine their discontinuity to the
    declared segment, and both retain explicit Williams tip functions.
    """

    def __init__(
        self,
        problem: StaticXDEMProblem2D,
        *,
        displacement_scale: float,
        hidden_layers: tuple[int, ...] = (48, 48, 48),
        crack_decay: float = 8.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.displacement_scale = float(displacement_scale)
        self.crack_decay = float(crack_decay)
        self.representation_family = str(
            problem.metadata.get(
                "neural_representation", _ADDITIVE_JUMP_REPRESENTATION
            )
        )
        if self.representation_family not in _SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                "Unsupported finite-domain XDEM representation "
                f"{self.representation_family!r}; expected one of "
                f"{tuple(sorted(_SUPPORTED_REPRESENTATIONS))!r}."
            )
        if self.displacement_scale <= 0.0:
            raise ValueError("displacement_scale must be positive.")
        if self.crack_decay <= 0.0:
            raise ValueError("crack_decay must be positive.")
        xmin, xmax, ymin, ymax = problem.domain.bounds
        self.register_buffer(
            "domain_center",
            torch.tensor((0.5 * (xmin + xmax), 0.5 * (ymin + ymax)), dtype=dtype),
        )
        self.register_buffer(
            "domain_half_span",
            torch.tensor((0.5 * (xmax - xmin), 0.5 * (ymax - ymin)), dtype=dtype),
        )
        self.register_buffer(
            "crack_centers",
            torch.tensor(
                [
                    (0.5 * (item.start[0] + item.end[0]), 0.5 * (item.start[1] + item.end[1]))
                    for item in problem.cracks.cracks
                ],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "crack_tangents",
            torch.tensor([item.tangent for item in problem.cracks.cracks], dtype=dtype),
        )
        self.register_buffer(
            "crack_normals",
            torch.tensor([item.normal for item in problem.cracks.cracks], dtype=dtype),
        )
        self.register_buffer(
            "crack_lengths",
            torch.tensor([item.length for item in problem.cracks.cracks], dtype=dtype),
        )
        self.register_buffer(
            "crack_has_two_active_tips",
            torch.tensor(
                [
                    set(item.metadata.get("active_ends", ("start", "end")))
                    == {"start", "end"}
                    for item in problem.cracks.cracks
                ],
                dtype=torch.bool,
            ),
        )
        self.active_tip_ids = problem.active_tip_ids
        self.register_buffer(
            "tip_points",
            torch.tensor([item.point for item in problem.active_tips], dtype=dtype),
        )
        self.register_buffer(
            "tip_extensions",
            torch.tensor(
                [item.extension_direction for item in problem.active_tips], dtype=dtype
            ),
        )
        self.register_buffer(
            "tip_normals",
            torch.tensor([item.normal for item in problem.active_tips], dtype=dtype),
        )
        self.register_buffer(
            "tip_crack_lengths",
            torch.tensor(
                [problem.cracks.crack(item.crack_id).length for item in problem.active_tips],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "tip_angles",
            torch.tensor(
                [
                    np.arctan2(item.extension_direction[1], item.extension_direction[0])
                    for item in problem.active_tips
                ],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "tip_support_radii",
            torch.tensor(
                [
                    problem.cracks.admissible_tip_radius(
                        item.tip_id, bounds=problem.domain.bounds
                    )
                    for item in problem.active_tips
                ],
                dtype=dtype,
            ),
        )
        characteristic_length = max(xmax - xmin, ymax - ymin)
        material_summary = problem.material.summary()
        stress_scale = (
            float(material_summary["young_modulus"])
            * self.displacement_scale
            / characteristic_length
        )
        self.register_buffer(
            "tip_sif_scales",
            stress_scale * torch.sqrt(self.tip_crack_lengths.clone()),
        )
        self.tip_amplitudes = torch.nn.Parameter(
            _initial_tip_amplitudes(
                problem,
                stress_scale=stress_scale,
                dtype=dtype,
            )
        )
        if self.representation_family == _RIEMANN_SHEET_COORDINATE_REPRESENTATION:
            input_width = 2 + 2 * len(problem.cracks.cracks)
        elif self.representation_family in _CRACK_COORDINATE_REPRESENTATIONS:
            input_width = 2 + len(problem.cracks.cracks)
        else:
            input_width = 2
        self.regular_network = _vector_mlp(
            hidden_layers, input_width=input_width, dtype=dtype
        )
        self.jump_networks = torch.nn.ModuleList(
            _vector_mlp(hidden_layers, dtype=dtype)
            for _ in (
                problem.cracks.cracks
                if self.representation_family == _ADDITIVE_JUMP_REPRESENTATION
                else ()
            )
        )
        gauge = _complete_point_gauge(problem, dtype=dtype)
        if gauge is None:
            self.register_buffer("gauge_points", torch.empty((0, 2), dtype=dtype))
            self.register_buffer("gauge_components", torch.empty((0,), dtype=torch.long))
            self.register_buffer("gauge_values", torch.empty((0,), dtype=dtype))
            self.register_buffer("gauge_matrix", torch.empty((0, 3), dtype=dtype))
        else:
            points, components, values, matrix = gauge
            self.register_buffer("gauge_points", points)
            self.register_buffer("gauge_components", components)
            self.register_buffer("gauge_values", values)
            self.register_buffer("gauge_matrix", matrix)
        spatial = tuple(
            item
            for item in problem.conditions
            if isinstance(item, SpatialDisplacementCondition2D)
        )
        if spatial and (
            len(spatial) != 1 or set(spatial[0].boundaries) != {"left", "right", "bottom", "top"}
        ):
            raise NotImplementedError(
                "The first hard spatial boundary adapter requires one field on "
                "all four rectangular boundaries."
            )
        self.spatial_boundary = spatial[0] if spatial else None

    @property
    def has_hard_point_gauge(self) -> bool:
        """Whether all planar rigid modes are removed by exact projection."""

        return bool(len(self.gauge_values) == 3)

    @property
    def has_hard_spatial_boundary(self) -> bool:
        return self.spatial_boundary is not None

    def additive_jump_features(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return the legacy LEFM-windowed additive jump channels."""

        relative = coordinates[:, None, :] - self.crack_centers[None, :, :]
        along = torch.einsum("nci,ci->nc", relative, self.crack_tangents)
        normal = torch.einsum("nci,ci->nc", relative, self.crack_normals)
        half_length = 0.5 * self.crack_lengths[None, :]
        closure = 1.0 - (along / half_length).square()
        lefm_window = torch.where(
            closure > 0.0,
            torch.sqrt(torch.clamp(closure, min=1.0e-12)),
            torch.zeros_like(closure),
        )
        smooth_window = torch.relu(closure).square()
        window = torch.where(
            self.crack_has_two_active_tips[None, :],
            lefm_window,
            smooth_window,
        )
        sign = torch.where(normal >= 0.0, 1.0, -1.0)
        # A |normal|-exponential has a cusp at the crack face.  Its nonzero
        # one-sided derivative lets the network manufacture a vanishing face
        # traction inside an arbitrarily thin layer without learning the
        # physical displacement jump.  Gaussian localization preserves the
        # compact neural feature while giving it zero normal derivative on the
        # declared crack surface.
        decay = torch.exp(
            -self.crack_decay * (normal / self.crack_lengths[None, :]).square()
        )
        return sign * window * decay

    def published_crack_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return bounded XDEM crack coordinates for every straight segment.

        This is an independent implementation of the coordinate described by
        Wang et al. (2026): a one-sided sign, a squared endpoint window, and a
        distance decay.  Distances are normalized by crack length so a change
        of engineering units does not change the representation.  Squaring
        the endpoint window makes both the coordinate and its tangential
        derivative vanish at the two tips and prevents a spurious jump on the
        infinite extension of the crack line.
        """

        relative = coordinates[:, None, :] - self.crack_centers[None, :, :]
        along = torch.einsum("nci,ci->nc", relative, self.crack_tangents)
        normal = torch.einsum("nci,ci->nc", relative, self.crack_normals)
        half_length = 0.5 * self.crack_lengths[None, :]
        endpoint_window = torch.relu(
            1.0 - (along / half_length).square()
        ).square()
        side = torch.where(normal >= 0.0, 1.0, -1.0)
        decay = torch.exp(
            -self.crack_decay
            * torch.abs(normal)
            / self.crack_lengths[None, :]
        )
        return side * endpoint_window * decay

    def bounded_sheet_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return two-sheet coordinates without an artificial normal layer.

        This keeps the sign and squared endpoint window but removes normal
        localization.  The coordinate therefore has zero normal derivative on
        either smooth sheet and does not reconnect the two displacement
        branches through a prescribed, strain-carrying transition layer.  It
        is an AgentFEM-Learning formulation candidate, not the published
        equation.
        """

        relative = coordinates[:, None, :] - self.crack_centers[None, :, :]
        along = torch.einsum("nci,ci->nc", relative, self.crack_tangents)
        normal = torch.einsum("nci,ci->nc", relative, self.crack_normals)
        half_length = 0.5 * self.crack_lengths[None, :]
        endpoint_window = torch.relu(
            1.0 - (along / half_length).square()
        ).square()
        side = torch.where(normal >= 0.0, 1.0, -1.0)
        return side * endpoint_window

    def riemann_sheet_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return analytic two-sheet coordinates for finite straight cracks.

        For local ``z=s+i*n`` and half crack length ``a``, this evaluates the
        real and imaginary parts of the branch ``sqrt(z**2-a**2)`` that behaves
        like ``z`` at infinity.  The two traces differ only on ``|s|<a, n=0``
        and merge continuously around both endpoints.  Unlike a localized
        Heaviside feature, this coordinate introduces no prescribed transition
        layer or discontinuity on the infinite crack extension.
        """

        relative = coordinates[:, None, :] - self.crack_centers[None, :, :]
        along = torch.einsum("nci,ci->nc", relative, self.crack_tangents)
        normal = torch.einsum("nci,ci->nc", relative, self.crack_normals)
        half_length = 0.5 * self.crack_lengths[None, :]
        scaled_along = along / half_length
        scaled_normal = normal / half_length
        real_part = scaled_along.square() - scaled_normal.square() - 1.0
        imaginary_part = 2.0 * scaled_along * scaled_normal
        epsilon = torch.as_tensor(
            1.0e-24, dtype=coordinates.dtype, device=coordinates.device
        )
        magnitude = torch.sqrt(
            real_part.square() + imaginary_part.square() + epsilon.square()
        )
        floor = torch.sqrt(epsilon)

        def safe_zero_sqrt(value):
            return torch.sqrt(torch.clamp(value, min=epsilon)) - floor

        mapped_real = torch.where(along >= 0.0, 1.0, -1.0) * safe_zero_sqrt(
            0.5 * (magnitude + real_part)
        )
        mapped_imaginary = torch.where(
            normal >= 0.0, 1.0, -1.0
        ) * safe_zero_sqrt(0.5 * (magnitude - real_part))
        normalized = torch.stack(
            (mapped_real, mapped_imaginary), dim=2
        )
        return normalized.reshape(len(coordinates), -1)

    def crack_features(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return the crack coordinates used by the selected representation."""

        if self.representation_family == _PUBLISHED_CRACK_COORDINATE_REPRESENTATION:
            return self.published_crack_coordinates(coordinates)
        if (
            self.representation_family == _BOUNDED_SHEET_COORDINATE_REPRESENTATION
        ):
            return self.bounded_sheet_coordinates(coordinates)
        if self.representation_family == _RIEMANN_SHEET_COORDINATE_REPRESENTATION:
            return self.riemann_sheet_coordinates(coordinates)
        return self.additive_jump_features(coordinates)

    def tip_features(self, coordinates: torch.Tensor) -> torch.Tensor:
        relative = coordinates[:, None, :] - self.tip_points[None, :, :]
        local_x = torch.einsum("nti,ti->nt", relative, self.tip_extensions)
        local_y = torch.einsum("nti,ti->nt", relative, self.tip_normals)
        radial = torch.sqrt(torch.clamp(local_x.square() + local_y.square(), min=1.0e-24))
        angle = torch.atan2(local_y, local_x)
        half = 0.5 * angle
        backward = torch.minimum(local_x, torch.zeros_like(local_x))
        branch_support = torch.clamp(
            1.0 + backward / self.tip_crack_lengths[None, :], min=0.0, max=1.0
        )
        radial_scale = torch.sqrt(radial / self.tip_crack_lengths[None, :]) * torch.exp(
            -radial / self.tip_crack_lengths[None, :]
        )
        scale = radial_scale * branch_support
        features = torch.stack(
            (
                scale * torch.sin(half),
                scale * torch.cos(half),
                scale * torch.sin(half) * torch.sin(angle),
                scale * torch.cos(half) * torch.sin(angle),
            ),
            dim=2,
        )
        return features.reshape(len(coordinates), -1)

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        normalized = (coordinates - self.domain_center) / self.domain_half_span
        return torch.cat((normalized, self.crack_features(coordinates)), dim=1)

    @staticmethod
    def _smooth_step(value):
        selected = torch.clamp(value, min=0.0, max=1.0)
        return selected.square() * (3.0 - 2.0 * selected)

    def tip_enrichment(self, coordinates: torch.Tensor) -> torch.Tensor:
        enriched = torch.zeros(
            (len(coordinates), 2), dtype=coordinates.dtype, device=coordinates.device
        )
        for index, tip in enumerate(self.problem.active_tips):
            relative = coordinates - self.tip_points[index]
            local_x = relative @ self.tip_extensions[index]
            local_y = relative @ self.tip_normals[index]
            radial = torch.sqrt(
                torch.clamp(local_x.square() + local_y.square(), min=1.0e-24)
            )
            support_radius = self.tip_support_radii[index]
            radial_cut = 1.0 - self._smooth_step(
                (radial - support_radius) / support_radius
            )
            length = self.tip_crack_lengths[index]
            backward_distance = torch.relu(-local_x)
            branch_cut = 1.0 - self._smooth_step(
                (backward_distance - 0.75 * length) / (0.25 * length)
            )
            amplitudes = self.tip_amplitudes[index] * self.tip_sif_scales[index]
            selected = williams_vector_field(
                coordinates,
                young_modulus=self.problem.material.summary()["young_modulus"],
                poisson_ratio=self.problem.material.summary()["poisson_ratio"],
                assumption=self.problem.material.summary()["assumption"],
                k_i=amplitudes[0],
                k_ii=amplitudes[1],
                tip=tip.point,
                crack_angle=self.tip_angles[index],
            )
            enriched = enriched + (radial_cut * branch_cut)[:, None] * selected
        return enriched

    def _raw_displacement(self, coordinates: torch.Tensor) -> torch.Tensor:
        components = self.raw_displacement_components(coordinates)
        return sum(components.values())

    def raw_displacement_components(
        self, coordinates: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return strain-carrying approximation channels before rigid gauges.

        The exact point-gauge correction is a rigid motion and therefore does
        not change stress.  Keeping these channels queryable makes a failed
        crack-face check diagnosable instead of merely reporting one aggregate
        number.
        """

        normalized = (coordinates - self.domain_center) / self.domain_half_span
        crack_features = self.crack_features(coordinates)
        if self.representation_family in _CRACK_COORDINATE_REPRESENTATIONS:
            neural = self.regular_network(
                torch.cat((normalized, crack_features), dim=1)
            )
            return {
                "crack_coordinate_network": self.displacement_scale * neural,
                "williams_tip": self.tip_enrichment(coordinates),
            }
        else:
            regular = self.regular_network(normalized)
            jump = torch.zeros_like(regular)
            for index, network in enumerate(self.jump_networks):
                jump = jump + crack_features[:, index : index + 1] * network(
                    normalized
                )
            return {
                "regular_network": self.displacement_scale * regular,
                "jump_network": self.displacement_scale * jump,
                "williams_tip": self.tip_enrichment(coordinates),
            }

    def _rigid_basis(self, coordinates: torch.Tensor) -> torch.Tensor:
        relative = coordinates - self.domain_center
        basis = torch.zeros(
            (len(coordinates), 2, 3),
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        basis[:, 0, 0] = 1.0
        basis[:, 1, 1] = 1.0
        basis[:, 0, 2] = -relative[:, 1]
        basis[:, 1, 2] = relative[:, 0]
        return basis

    def _spatial_field(self, coordinates: torch.Tensor) -> torch.Tensor:
        condition = self.spatial_boundary
        if condition is None:
            raise RuntimeError("No hard spatial displacement field is configured.")
        selected = condition.field
        parameters = selected.parameters
        material = self.problem.material.summary()
        if selected.family == "williams_displacement":
            return williams_vector_field(
                coordinates,
                young_modulus=material["young_modulus"],
                poisson_ratio=material["poisson_ratio"],
                assumption=material["assumption"],
                k_i=parameters["k_i"],
                k_ii=parameters["k_ii"],
                tip=parameters["tip"],
                crack_angle=parameters["crack_angle"],
            )
        if selected.family == "westergaard_center_crack_displacement":
            return _westergaard_center_crack_displacement(
                coordinates,
                young_modulus=material["young_modulus"],
                poisson_ratio=material["poisson_ratio"],
                assumption=material["assumption"],
                center=parameters["center"],
                half_crack_length=parameters["half_crack_length"],
                remote_stress=parameters["remote_stress"],
            )
        raise NotImplementedError(
            f"Unsupported spatial boundary family {selected.family!r}."
        )

    def _transfinite_boundary_lifting(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Coons lifting assembled only from the four prescribed traces."""

        if (
            self.spatial_boundary.field.metadata.get("interior_extension")
            == "declared_field"
        ):
            return self._spatial_field(coordinates)

        xmin, xmax, ymin, ymax = self.problem.domain.bounds
        xi = (coordinates[:, 0] - xmin) / (xmax - xmin)
        eta = (coordinates[:, 1] - ymin) / (ymax - ymin)
        left_points = torch.stack((torch.full_like(eta, xmin), coordinates[:, 1]), dim=1)
        right_points = torch.stack((torch.full_like(eta, xmax), coordinates[:, 1]), dim=1)
        bottom_points = torch.stack((coordinates[:, 0], torch.full_like(xi, ymin)), dim=1)
        top_points = torch.stack((coordinates[:, 0], torch.full_like(xi, ymax)), dim=1)
        left = self._spatial_field(left_points)
        right = self._spatial_field(right_points)
        bottom = self._spatial_field(bottom_points)
        top = self._spatial_field(top_points)
        corners = self._spatial_field(
            torch.tensor(
                ((xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)),
                dtype=coordinates.dtype,
                device=coordinates.device,
            )
        )
        edge_blend = (
            (1.0 - xi)[:, None] * left
            + xi[:, None] * right
            + (1.0 - eta)[:, None] * bottom
            + eta[:, None] * top
        )
        corner_blend = (
            ((1.0 - xi) * (1.0 - eta))[:, None] * corners[0]
            + (xi * (1.0 - eta))[:, None] * corners[1]
            + ((1.0 - xi) * eta)[:, None] * corners[2]
            + (xi * eta)[:, None] * corners[3]
        )
        return edge_blend - corner_blend

    def _boundary_distance_factor(self, coordinates: torch.Tensor) -> torch.Tensor:
        xmin, xmax, ymin, ymax = self.problem.domain.bounds
        xi = (coordinates[:, 0] - xmin) / (xmax - xmin)
        eta = (coordinates[:, 1] - ymin) / (ymax - ymin)
        return 16.0 * xi * (1.0 - xi) * eta * (1.0 - eta)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        raw = self._raw_displacement(coordinates)
        if self.has_hard_spatial_boundary:
            lifting = self._transfinite_boundary_lifting(coordinates)
            if (
                self.spatial_boundary.field.metadata.get("interior_extension")
                == "declared_field"
            ):
                # Extended patch tests verify a declared exact field and the
                # downstream crack diagnostics. A zero-valued graph connection
                # keeps the common optimizer lifecycle executable without
                # allowing quadrature error to perturb the reference solution.
                return lifting + 0.0 * raw
            return lifting + self._boundary_distance_factor(coordinates)[
                :, None
            ] * raw
        if not self.has_hard_point_gauge:
            return raw
        gauge_raw = self._raw_displacement(self.gauge_points)
        rows = torch.arange(3, device=coordinates.device)
        residual = self.gauge_values - gauge_raw[rows, self.gauge_components]
        coefficients = torch.linalg.solve(self.gauge_matrix, residual)
        return raw + torch.einsum(
            "nic,c->ni", self._rigid_basis(coordinates), coefficients
        )


@dataclass(frozen=True)
class FiniteDomainTrainingOutcome:
    model: FiniteDomainVectorNetwork
    epochs: np.ndarray
    losses: np.ndarray
    coordinates: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    crack_trace_coordinates: np.ndarray
    crack_trace_visual_coordinates: np.ndarray
    crack_trace_side: np.ndarray
    crack_trace_id: np.ndarray
    crack_trace_displacement: np.ndarray
    crack_trace_stress: np.ndarray
    metrics: dict[str, float]
    stress_intensity: object
    crack_opening_stress_intensity: tuple[object, ...]
    quadrature_topology: dict[str, object]
    device: str
    dtype: str

    def write_field(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            coordinates=self.coordinates,
            displacement=self.displacement,
            stress=self.stress,
            crack_trace_coordinates=self.crack_trace_coordinates,
            crack_trace_visual_coordinates=self.crack_trace_visual_coordinates,
            crack_trace_side=self.crack_trace_side,
            crack_trace_id=self.crack_trace_id,
            crack_trace_displacement=self.crack_trace_displacement,
            crack_trace_stress=self.crack_trace_stress,
        )
        return output

    def write_paraview(self, path: str | Path) -> Path:
        """Write one ParaView VTU with duplicated one-sided crack traces.

        The bulk sample grid and both coincident crack faces are cells in one
        unstructured dataset.  Crack-face values are evaluated at a tiny
        one-sided offset but stored on duplicated centerline coordinates, so
        ParaView never averages the two physical traces into one value.
        """

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_discontinuous_vtu(output, self)
        return output


def train_finite_domain(
    spec,
    options: ReferenceTrainingOptions,
    *,
    tip_plan_options: dict[str, object] | None = None,
) -> FiniteDomainTrainingOutcome:
    """Train one finite-domain predefined-crack problem."""

    problem = problem_from_spec(spec)
    parameters = _material_parameters(problem)
    dtype = torch.float64 if options.dtype == "float64" else torch.float32
    device = _resolve_device(options.device, dtype=options.dtype)
    torch.manual_seed(int(options.seed))
    displacement_scale = _displacement_scale(problem)
    characteristic_length = max(
        problem.domain.bounds[1] - problem.domain.bounds[0],
        problem.domain.bounds[3] - problem.domain.bounds[2],
    )
    parameters["stress_scale"] = (
        parameters["young_modulus"] * displacement_scale / characteristic_length
    )
    parameters["length_scale"] = characteristic_length
    model = FiniteDomainVectorNetwork(
        problem,
        displacement_scale=displacement_scale,
        hidden_layers=options.hidden_layers,
        dtype=dtype,
    ).to(device)
    rules = spec.integration
    training_quadrature, training_topology = _cut_domain_quadrature(
        problem,
        rules.training.count,
        seed=rules.training.seed,
        dtype=dtype,
        device=device,
    )
    validation_quadrature, validation_topology = _cut_domain_quadrature(
        problem,
        rules.validation.count,
        seed=rules.validation.seed,
        dtype=dtype,
        device=device,
    )
    refinement = rules.refinements[0]
    refined_quadrature, refined_topology = _cut_domain_quadrature(
        problem,
        refinement.count,
        seed=refinement.seed,
        dtype=dtype,
        device=device,
    )
    boundary_count = next(
        (
            int(item.count)
            for item in spec.sampling
            if item.on == "boundary" and item.count is not None
        ),
        256,
    )
    boundary_data = _boundary_data(
        problem, count=boundary_count, dtype=dtype, device=device
    )
    crack_face_data = _crack_face_data(
        problem, count=boundary_count, dtype=dtype, device=device
    )
    energy_scale = parameters["young_modulus"] * displacement_scale**2

    def objective(quadrature, *, create_graph, consistency_weight=0.0):
        points, weights = quadrature
        displacement = model(points)
        _, _, density = _elastic_state(
            displacement, points, parameters, create_graph=create_graph
        )
        internal = torch.sum(weights * density)
        external, condition_error = _boundary_terms(model, boundary_data)
        if consistency_weight > 0.0:
            crack_face_error = _crack_face_traction_error(
                model, crack_face_data, parameters, create_graph=create_graph
            )
            equilibrium_points = points[: min(256, max(32, len(points) // 8))]
            equilibrium_error = _equilibrium_error(
                model, equilibrium_points, parameters, create_graph=create_graph
            )
        else:
            crack_face_error = torch.zeros_like(internal)
            equilibrium_error = torch.zeros_like(internal)
        total = internal - external
        selected = total / energy_scale + options.boundary_penalty * condition_error / (
            displacement_scale * displacement_scale
        )
        if consistency_weight > 0.0:
            stress_scale_squared = parameters["stress_scale"] ** 2
            selected = selected + consistency_weight * (
                crack_face_error / stress_scale_squared
                + equilibrium_error
                * parameters["length_scale"] ** 2
                / stress_scale_squared
            )
        return (
            selected,
            internal,
            external,
            condition_error,
            crack_face_error,
            equilibrium_error,
        )

    epochs: list[float] = []
    losses: list[float] = []
    regular_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "tip_amplitudes"
    ]
    optimizer = torch.optim.Adam(
        (
            {"params": regular_parameters, "lr": options.learning_rate},
            {"params": (model.tip_amplitudes,), "lr": 10.0 * options.learning_rate},
        )
    )
    for epoch in range(1, options.adam_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, _, _ = objective(
            training_quadrature,
            create_graph=True,
        )
        loss.backward()
        optimizer.step()
        epochs.append(float(epoch))
        losses.append(float(loss.detach().cpu()))
        if options.progress and (epoch == 1 or epoch % 100 == 0):
            physical_tip_k = (
                model.tip_amplitudes.detach() * model.tip_sif_scales[:, None]
            )
            print(
                f"[agentfem-learning.xdem] finite-domain Adam {epoch}/"
                f"{options.adam_epochs}: loss={losses[-1]:.6e} | "
                f"max|K_tip|={float(physical_tip_k.abs().max().cpu()):.6e}",
                flush=True,
            )

    if options.lbfgs_steps:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=12,
            history_size=30,
            line_search_fn="strong_wolfe",
        )
        for step in range(1, options.lbfgs_steps + 1):

            def closure():
                optimizer_lbfgs.zero_grad(set_to_none=True)
                selected, _, _, _, _, _ = objective(
                    training_quadrature,
                    create_graph=True,
                )
                selected.backward()
                return selected

            optimizer_lbfgs.step(closure)
            selected, _, _, _, _, _ = objective(
                training_quadrature,
                create_graph=True,
            )
            epochs.append(float(options.adam_epochs + step))
            losses.append(float(selected.detach().cpu()))

    evaluation = _evaluation_grid(problem, rules.validation.count, dtype=dtype, device=device)
    prediction = model(evaluation)
    _, stress, _ = _elastic_state(prediction, evaluation, parameters, create_graph=False)
    training_values = objective(
        training_quadrature, create_graph=False, consistency_weight=1.0
    )
    validation_values = objective(
        validation_quadrature, create_graph=False, consistency_weight=1.0
    )
    refined_values = objective(
        refined_quadrature, create_graph=False, consistency_weight=1.0
    )
    field = TorchVectorFractureField(model, parameters, dtype=dtype, device=device)
    plan = tip_integration_plan(
        problem.cracks,
        bounds=problem.domain.bounds,
        tip_ids=problem.active_tip_ids,
        relative_path_tolerance=0.08,
        metadata={"problem": problem.name},
        **dict(tip_plan_options or {}),
    )
    reports = stress_intensity_reports(
        field,
        cracks=problem.cracks,
        material=problem.material,
        plan=plan,
        metadata={"problem": problem.name, "solver": "finite_domain_xdem_d"},
    )
    opening_reports = crack_opening_sif_reports(
        field,
        cracks=problem.cracks,
        material=problem.material,
        plan=plan,
    )
    (
        trace_coordinates,
        trace_visual_coordinates,
        trace_side,
        trace_id,
    ) = _crack_trace(problem)
    trace_tensor = torch.as_tensor(trace_coordinates, dtype=dtype, device=device)
    trace_prediction = model(trace_tensor)
    trace_stress = field.stress(trace_coordinates)
    energy_gap = abs(
        float(validation_values[1].detach().cpu()) - float(refined_values[1].detach().cpu())
    ) / max(abs(float(refined_values[1].detach().cpu())), energy_scale * 1.0e-12)
    component_face_errors = _crack_face_component_errors(
        model, crack_face_data, parameters
    )
    metrics = {
        "training_internal_energy": float(training_values[1].detach().cpu()),
        "validation_internal_energy": float(validation_values[1].detach().cpu()),
        "refined_internal_energy": float(refined_values[1].detach().cpu()),
        "validation_external_work": float(validation_values[2].detach().cpu()),
        "relative_boundary_error": float(
            torch.sqrt(validation_values[3]).detach().cpu() / displacement_scale
        ),
        "relative_crack_face_traction_error": float(
            torch.sqrt(validation_values[4]).detach().cpu()
            / parameters["stress_scale"]
        ),
        "relative_equilibrium_residual": float(
            torch.sqrt(validation_values[5]).detach().cpu()
            * parameters["length_scale"]
            / parameters["stress_scale"]
        ),
        "validation_refinement_energy_gap": float(energy_gap),
        "maximum_stress_intensity_path_variation": float(
            max(item.path_variation for item in reports.reports)
        ),
        "training_quadrature_weight_error": float(
            abs(float(training_quadrature[1].sum().detach().cpu()) - problem.domain.area)
            / problem.domain.area
        ),
        "training_cut_or_tip_point_fraction": float(
            sum(
                kind in {"cut", "tip", "one_sided"}
                for kind in training_topology.cell_kinds
            )
            / len(training_topology.cell_kinds)
        ),
        "training_crack_adaptive_point_fraction": float(
            sum(
                kind
                in {"cut", "tip", "one_sided", "near_crack", "aligned"}
                for kind in training_topology.cell_kinds
            )
            / len(training_topology.cell_kinds)
        ),
        "maximum_sif_extractor_disagreement": float(
            max(
                np.linalg.norm(
                    np.asarray((domain_report.k_i, domain_report.k_ii))
                    - np.asarray((opening_report.k_i, opening_report.k_ii))
                )
                / max(
                    np.linalg.norm(
                        np.asarray((domain_report.k_i, domain_report.k_ii))
                    ),
                    1.0e-30,
                )
                for domain_report, opening_report in zip(
                    reports.reports, opening_reports, strict=True
                )
            )
        ),
    }
    metrics.update(
        {
            f"relative_crack_face_traction_{name}": float(
                torch.sqrt(value).detach().cpu() / parameters["stress_scale"]
            )
            for name, value in component_face_errors.items()
        }
    )
    return FiniteDomainTrainingOutcome(
        model=model,
        epochs=np.asarray(epochs, dtype=float),
        losses=np.asarray(losses, dtype=float),
        coordinates=evaluation.detach().cpu().numpy(),
        displacement=prediction.detach().cpu().numpy(),
        stress=stress.detach().cpu().numpy(),
        crack_trace_coordinates=trace_coordinates,
        crack_trace_visual_coordinates=trace_visual_coordinates,
        crack_trace_side=trace_side,
        crack_trace_id=trace_id,
        crack_trace_displacement=trace_prediction.detach().cpu().numpy(),
        crack_trace_stress=trace_stress,
        metrics=metrics,
        stress_intensity=reports,
        crack_opening_stress_intensity=opening_reports,
        quadrature_topology={
            "training": training_topology.summary(),
            "validation": validation_topology.summary(),
            "refinement": refined_topology.summary(),
        },
        device=str(device),
        dtype=options.dtype,
    )


def problem_from_spec(spec) -> StaticXDEMProblem2D:
    metadata = dict(spec.metadata)
    if metadata.get("provider") != "agentfem-learning.xdem":
        raise ValueError("Finite-domain trainer requires the AgentFEM-Learning provider.")
    if metadata.get("problem") != "finite_domain_static_xdem_d":
        raise NotImplementedError("Finite-domain trainer received another problem type.")
    summary = dict(metadata["scientific_problem"])
    domain_summary = dict(summary["domain"])
    domain = rectangular_domain(
        domain_summary["bounds"],
        name=domain_summary["name"],
        metadata=domain_summary.get("metadata", {}),
    )
    material_summary = dict(summary["material"])
    material = fracture.linear_elastic_fracture_material(
        young_modulus=material_summary["young_modulus"],
        poisson_ratio=material_summary["poisson_ratio"],
        assumption=material_summary["assumption"],
    )
    crack_summary = dict(summary["cracks"])
    cracks = fracture.crack_set(
        *(
            fracture.segment(
                item["crack_id"],
                start=item["start"],
                end=item["end"],
                metadata=item.get("metadata", {}),
            )
            for item in crack_summary["cracks"]
        ),
        name=crack_summary["name"],
        tolerance=crack_summary["tolerance"],
        metadata=crack_summary.get("metadata", {}),
    )
    conditions = []
    for item in summary["conditions"]:
        if item.get("location") == "point":
            conditions.append(
                point_displacement(
                    item["name"],
                    item["point"],
                    item["value"],
                    metadata=item.get("metadata", {}),
                )
            )
        elif item.get("location") == "boundaries":
            field_summary = dict(item["field"])
            conditions.append(
                spatial_displacement_bc(
                    item["name"],
                    item["boundaries"],
                    SpatialVectorField2D(
                        name=field_summary["name"],
                        family=field_summary["family"],
                        parameters=field_summary["parameters"],
                        metadata=field_summary.get("metadata", {}),
                    ),
                    metadata=item.get("metadata", {}),
                )
            )
        else:
            constructor = displacement_bc if item["kind"] == "displacement" else traction_bc
            conditions.append(
                constructor(
                    item["name"],
                    item["boundary"],
                    item["value"],
                    metadata=item.get("metadata", {}),
                )
            )
    return static_crack_problem(
        domain=domain,
        material=material,
        cracks=cracks,
        conditions=conditions,
        name=summary["name"],
        metadata=summary.get("metadata", {}),
    )


def _material_parameters(problem):
    selected = problem.material.summary()
    return {
        "young_modulus": float(selected["young_modulus"]),
        "poisson_ratio": float(selected["poisson_ratio"]),
        "assumption": str(selected["assumption"]),
    }


def _displacement_scale(problem):
    prescribed = max(
        (
            abs(value)
            for condition in problem.conditions
            if condition.kind == "displacement"
            and not isinstance(condition, SpatialDisplacementCondition2D)
            for value in condition.value
            if value is not None
        ),
        default=0.0,
    )
    characteristic_length = max(
        problem.domain.bounds[1] - problem.domain.bounds[0],
        problem.domain.bounds[3] - problem.domain.bounds[2],
    )
    modulus = float(problem.material.summary()["young_modulus"])
    spatial = max(
        (
            _spatial_displacement_scale(
                condition.field,
                characteristic_length=characteristic_length,
                modulus=modulus,
            )
            for condition in problem.conditions
            if isinstance(condition, SpatialDisplacementCondition2D)
        ),
        default=0.0,
    )
    traction = max(
        (
            float(np.linalg.norm(condition.value))
            for condition in problem.conditions
            if condition.kind == "traction"
        ),
        default=0.0,
    )
    xmin, xmax, ymin, ymax = problem.domain.bounds
    length = max(xmax - xmin, ymax - ymin)
    scale = max(prescribed, spatial, traction * length / modulus)
    if scale <= 0.0:
        raise ValueError("The finite-domain problem has no nonzero mechanical loading.")
    return scale


def _spatial_displacement_scale(field, *, characteristic_length, modulus):
    parameters = field.parameters
    if field.family == "williams_displacement":
        return (
            np.hypot(float(parameters["k_i"]), float(parameters["k_ii"]))
            * np.sqrt(characteristic_length)
            / modulus
        )
    if field.family == "westergaard_center_crack_displacement":
        return (
            float(parameters["remote_stress"])
            * characteristic_length
            / modulus
        )
    raise NotImplementedError(
        f"Unsupported spatial boundary family {field.family!r}."
    )


def _stratified_domain_quadrature(problem, count, *, seed, dtype, device):
    """Area-exact Monte Carlo quadrature with resolved crack-tip disks.

    Uniform points can almost entirely miss the small regions that determine
    the energy of a square-root crack-tip field.  This partition integrates
    the same physical domain (weights still sum exactly to its area) while
    assigning a controlled number of independent samples to every admissible
    tip neighborhood.  It is variance reduction, not an artificial tip-energy
    multiplier.
    """

    selected_count = int(count)
    generator = np.random.default_rng(int(seed))
    xmin, xmax, ymin, ymax = problem.domain.bounds
    tip_points = np.asarray([item.point for item in problem.cracks.tips], dtype=float)
    tip_radii = np.asarray(
        [
            0.75
            * problem.cracks.admissible_tip_radius(
                item.tip_id, bounds=problem.domain.bounds
            )
            for item in problem.cracks.tips
        ],
        dtype=float,
    )
    disk_areas = np.pi * tip_radii**2
    background_area = problem.domain.area - float(np.sum(disk_areas))
    if background_area <= 0.0:
        raise ValueError("Crack-tip quadrature disks exceed the finite-domain area.")
    background_count = max(16, selected_count // 2)
    tip_total = selected_count - background_count
    base_tip_count, remainder = divmod(tip_total, len(tip_points))
    tip_counts = tuple(
        base_tip_count + (1 if index < remainder else 0)
        for index in range(len(tip_points))
    )
    if background_count < 16:
        raise ValueError("Finite-domain quadrature needs more background samples.")

    background_chunks = []
    accepted = 0
    while accepted < background_count:
        candidate_count = max(2 * (background_count - accepted), 128)
        candidate = np.column_stack(
            (
                generator.uniform(xmin, xmax, candidate_count),
                generator.uniform(ymin, ymax, candidate_count),
            )
        )
        distance = np.linalg.norm(
            candidate[:, None, :] - tip_points[None, :, :], axis=2
        )
        keep = candidate[np.all(distance >= tip_radii[None, :], axis=1)]
        selected = keep[: background_count - accepted]
        background_chunks.append(selected)
        accepted += len(selected)
    points = [np.vstack(background_chunks)]
    weights = [np.full(background_count, background_area / background_count)]
    for point, radius, area, tip_count in zip(
        tip_points, tip_radii, disk_areas, tip_counts, strict=True
    ):
        radial = radius * np.sqrt(generator.random(tip_count))
        angle = 2.0 * np.pi * generator.random(tip_count)
        points.append(
            point[None, :]
            + np.column_stack((radial * np.cos(angle), radial * np.sin(angle)))
        )
        weights.append(np.full(tip_count, area / tip_count))
    coordinates = torch.as_tensor(np.vstack(points), dtype=dtype, device=device)
    selected_weights = torch.as_tensor(
        np.concatenate(weights), dtype=dtype, device=device
    )
    return coordinates.detach().requires_grad_(True), selected_weights


def _tensor_domain_quadrature(problem, count, *, seed, dtype, device):
    """Deterministic midpoint rule that covers every part of the domain.

    The seed is accepted to preserve the common integration-provider contract;
    independence is supplied by the distinct training, validation, and
    refinement resolutions rather than random point clouds that a neural field
    can overfit between.
    """

    del seed
    xmin, xmax, ymin, ymax = problem.domain.bounds
    width = xmax - xmin
    height = ymax - ymin
    target = int(count)
    nx = max(8, round(np.sqrt(target * width / height)))
    ny = max(8, round(target / nx))
    x = xmin + (np.arange(nx, dtype=float) + 0.5) * width / nx
    y = ymin + (np.arange(ny, dtype=float) + 0.5) * height / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    values = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    coordinates = torch.as_tensor(values, dtype=dtype, device=device)
    weights = torch.full(
        (len(values),),
        problem.domain.area / len(values),
        dtype=dtype,
        device=device,
    )
    return coordinates.detach().requires_grad_(True), weights


def _cut_domain_quadrature(problem, count, *, seed, dtype, device):
    """Convert provider-neutral cut-cell evidence to differentiable tensors."""

    topology = straight_crack_cut_quadrature(problem, count, variant=seed)
    coordinates = torch.as_tensor(
        topology.coordinates, dtype=dtype, device=device
    ).detach().requires_grad_(True)
    weights = torch.as_tensor(topology.weights, dtype=dtype, device=device)
    return (coordinates, weights), topology


def _evaluation_grid(problem, count, *, dtype, device):
    side = max(16, int(np.sqrt(int(count))))
    xmin, xmax, ymin, ymax = problem.domain.bounds
    x = torch.linspace(xmin, xmax, side, dtype=dtype, device=device)
    y = torch.linspace(ymin, ymax, side, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).detach().requires_grad_(True)


def _boundary_data(problem, *, count, dtype, device):
    selected = []
    count = int(count)
    xmin, xmax, ymin, ymax = problem.domain.bounds
    for condition in problem.conditions:
        if isinstance(condition, PointDisplacementCondition2D):
            coordinates = torch.as_tensor(
                (condition.point,), dtype=dtype, device=device
            )
            selected.append((condition, coordinates, 0.0))
            continue
        boundaries = (
            condition.boundaries
            if isinstance(condition, SpatialDisplacementCondition2D)
            else (condition.boundary,)
        )
        for boundary in boundaries:
            fraction = (torch.arange(count, dtype=dtype, device=device) + 0.5) / count
            if boundary in {"left", "right"}:
                x = torch.full_like(fraction, xmin if boundary == "left" else xmax)
                y = ymin + (ymax - ymin) * fraction
                length = ymax - ymin
            else:
                x = xmin + (xmax - xmin) * fraction
                y = torch.full_like(fraction, ymin if boundary == "bottom" else ymax)
                length = xmax - xmin
            selected.append((condition, torch.stack((x, y), dim=1), length))
    return tuple(selected)


def _crack_face_data(problem, *, count, dtype, device):
    """Return independent one-sided samples for traction-free crack faces."""

    selected = []
    per_crack = max(16, int(count) // len(problem.cracks.cracks))
    fraction = (torch.arange(per_crack, dtype=dtype, device=device) + 0.5) / per_crack
    for crack in problem.cracks.cracks:
        start = torch.as_tensor(crack.start, dtype=dtype, device=device)
        end = torch.as_tensor(crack.end, dtype=dtype, device=device)
        centerline = start[None, :] * (1.0 - fraction[:, None]) + end[None, :] * fraction[:, None]
        normal = torch.as_tensor(crack.normal, dtype=dtype, device=device)
        epsilon = max(1.0e-8, 1.0e-6 * crack.length)
        for side in (-1.0, 1.0):
            points = (centerline + side * epsilon * normal[None, :]).detach()
            selected.append(
                (
                    points.requires_grad_(True),
                    side * normal,
                )
            )
    return tuple(selected)


def _crack_face_traction_error(model, crack_face_data, parameters, *, create_graph):
    errors = []
    for coordinates, outward_normal in crack_face_data:
        displacement = model(coordinates)
        _, stress, _ = _elastic_state(
            displacement, coordinates, parameters, create_graph=create_graph
        )
        traction = torch.einsum("nij,j->ni", stress, outward_normal)
        errors.append(traction.square().sum(dim=1).mean())
    return torch.stack(errors).mean()


def _crack_face_component_errors(model, crack_face_data, parameters):
    """Resolve crack-face traction error by approximation channel."""

    if model.has_hard_spatial_boundary:
        return {}
    selected: dict[str, list[torch.Tensor]] = {}
    for coordinates, outward_normal in crack_face_data:
        for name, displacement in model.raw_displacement_components(
            coordinates
        ).items():
            _, stress, _ = _elastic_state(
                displacement, coordinates, parameters, create_graph=False
            )
            traction = torch.einsum("nij,j->ni", stress, outward_normal)
            selected.setdefault(name, []).append(
                traction.square().sum(dim=1).mean()
            )
    return {
        name: torch.stack(values).mean()
        for name, values in selected.items()
    }


def _equilibrium_error(model, coordinates, parameters, *, create_graph):
    """Mean squared strong-form residual away from enriched tip disks."""

    displacement = model(coordinates)
    _, stress, _ = _elastic_state(
        displacement,
        coordinates,
        parameters,
        # A stress divergence always requires the first derivative graph.
        create_graph=True,
    )
    components = []
    for row in range(2):
        divergence = torch.zeros(
            len(coordinates), dtype=coordinates.dtype, device=coordinates.device
        )
        for column in range(2):
            derivative = torch.autograd.grad(
                stress[:, row, column].sum(),
                coordinates,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            divergence = divergence + derivative[:, column]
        components.append(divergence)
    return torch.stack(components, dim=1).square().sum(dim=1).mean()


def _boundary_terms(model, boundary_data):
    external = None
    errors = []
    for condition, coordinates, length in boundary_data:
        if (
            isinstance(condition, SpatialDisplacementCondition2D)
            and model.has_hard_spatial_boundary
        ):
            continue
        if (
            isinstance(condition, PointDisplacementCondition2D)
            and model.has_hard_point_gauge
        ):
            continue
        prediction = model(coordinates)
        if condition.kind == "traction":
            traction = torch.as_tensor(
                condition.value, dtype=prediction.dtype, device=prediction.device
            )
            contribution = length * torch.mean(prediction @ traction)
            external = contribution if external is None else external + contribution
        else:
            for component in condition.constrained_components:
                errors.append(
                    (prediction[:, component] - condition.value[component]).square().mean()
                )
    if external is None:
        sample = next(iter(boundary_data))[1]
        external = torch.zeros((), dtype=sample.dtype, device=sample.device)
    condition_error = torch.stack(errors).mean() if errors else torch.zeros_like(external)
    return external, condition_error


def _crack_trace(problem, *, count_per_crack: int = 64):
    coordinates = []
    visual_coordinates = []
    sides = []
    identifiers = []
    for crack in problem.cracks.cracks:
        fraction = (np.arange(count_per_crack, dtype=float) + 0.5) / count_per_crack
        centerline = (
            np.asarray(crack.start)[None, :] * (1.0 - fraction[:, None])
            + np.asarray(crack.end)[None, :] * fraction[:, None]
        )
        epsilon = 1.0e-7 * crack.length
        normal = np.asarray(crack.normal)
        for side in (1, -1):
            coordinates.append(centerline + side * epsilon * normal)
            visual_coordinates.append(centerline)
            sides.extend([side] * count_per_crack)
            identifiers.extend([crack.crack_id] * count_per_crack)
    return (
        np.vstack(coordinates),
        np.vstack(visual_coordinates),
        np.asarray(sides, dtype=int),
        np.asarray(identifiers, dtype="U"),
    )


def _write_discontinuous_vtu(path: Path, outcome: FiniteDomainTrainingOutcome) -> None:
    bulk_count = len(outcome.coordinates)
    side = round(np.sqrt(bulk_count))
    if side * side != bulk_count:
        raise ValueError("ParaView export requires the provider's square evaluation grid.")
    points_2d = np.vstack(
        (outcome.coordinates, outcome.crack_trace_visual_coordinates)
    )
    points = np.column_stack((points_2d, np.zeros(len(points_2d))))
    displacement_2d = np.vstack(
        (outcome.displacement, outcome.crack_trace_displacement)
    )
    displacement = np.column_stack(
        (displacement_2d, np.zeros(len(displacement_2d)))
    )
    stress_2d = np.concatenate(
        (outcome.stress, outcome.crack_trace_stress), axis=0
    )
    stress = np.zeros((len(stress_2d), 9), dtype=float)
    stress[:, 0] = stress_2d[:, 0, 0]
    stress[:, 1] = stress_2d[:, 0, 1]
    stress[:, 3] = stress_2d[:, 1, 0]
    stress[:, 4] = stress_2d[:, 1, 1]
    crack_side = np.concatenate(
        (np.zeros(bulk_count, dtype=int), outcome.crack_trace_side)
    )
    names = tuple(dict.fromkeys(str(item) for item in outcome.crack_trace_id))
    identifiers = {name: index + 1 for index, name in enumerate(names)}
    crack_id = np.concatenate(
        (
            np.zeros(bulk_count, dtype=int),
            np.asarray([identifiers[str(item)] for item in outcome.crack_trace_id]),
        )
    )

    cells: list[tuple[int, ...]] = []
    types: list[int] = []
    for j in range(side - 1):
        for i in range(side - 1):
            lower = j * side + i
            cells.append((lower, lower + 1, lower + side + 1, lower + side))
            types.append(9)  # VTK_QUAD
    trace_offset = bulk_count
    for name in names:
        for selected_side in (1, -1):
            local = np.flatnonzero(
                (outcome.crack_trace_id == name)
                & (outcome.crack_trace_side == selected_side)
            )
            for left, right in pairwise(local):
                cells.append((trace_offset + int(left), trace_offset + int(right)))
                types.append(3)  # VTK_LINE
    offsets = np.cumsum([len(cell) for cell in cells], dtype=int)
    connectivity = np.asarray([item for cell in cells for item in cell], dtype=int)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n')
        stream.write(
            '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        )
        stream.write("  <UnstructuredGrid>\n")
        stream.write(
            f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">\n'
        )
        stream.write('      <PointData Vectors="Displacement" Tensors="Stress">\n')
        _write_vtk_array(stream, "Displacement", displacement, 3, "Float64")
        _write_vtk_array(stream, "Stress", stress, 9, "Float64")
        _write_vtk_array(stream, "CrackSide", crack_side, 1, "Int32")
        _write_vtk_array(stream, "CrackId", crack_id, 1, "Int32")
        stream.write("      </PointData>\n")
        stream.write("      <Points>\n")
        _write_vtk_array(stream, None, points, 3, "Float64")
        stream.write("      </Points>\n")
        stream.write("      <Cells>\n")
        _write_vtk_array(stream, "connectivity", connectivity, 1, "Int64")
        _write_vtk_array(stream, "offsets", offsets, 1, "Int64")
        _write_vtk_array(stream, "types", np.asarray(types), 1, "UInt8")
        stream.write("      </Cells>\n")
        stream.write("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")
    temporary.replace(path)


def _write_vtk_array(stream, name, values, components: int, vtk_type: str) -> None:
    label = "" if name is None else f' Name="{name}"'
    component_text = "" if components == 1 else f' NumberOfComponents="{components}"'
    stream.write(
        f'        <DataArray type="{vtk_type}"{label}{component_text} format="ascii">\n'
    )
    flat = np.asarray(values).reshape(-1)
    for start in range(0, len(flat), 1024):
        chunk = flat[start : start + 1024]
        if np.issubdtype(chunk.dtype, np.integer):
            stream.write("          " + " ".join(str(int(item)) for item in chunk) + "\n")
        else:
            stream.write(
                "          " + " ".join(f"{float(item):.17g}" for item in chunk) + "\n"
            )
    stream.write("        </DataArray>\n")


__all__ = [
    "FiniteDomainTrainingOutcome",
    "FiniteDomainVectorNetwork",
    "problem_from_spec",
    "train_finite_domain",
]
