"""Train a GINO across AgentFEM heat-transfer domains of different width."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ufl
from agentfem import constitutive, datasets, extensions, fields, learning, mesh, models, studies
from mpi4py import MPI


def _solve(width: float, amplitude: float, *, shape: tuple[int, int]):
    height = 1.0
    domain = mesh.rectangle(
        (0.0, 0.0),
        (float(width), height),
        (16, 16),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=domain,
        name="variable_width_plate",
    )
    temperature = model.field(fields.temperature(domain, value=300.0))
    model.material(
        constitutive.thermoelastic(
            name="thermal solid",
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    exterior = mesh.boundary(
        domain,
        lambda x: (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], width)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], height)
        ),
        name="isothermal_boundary",
        tag=1,
    )
    model.prescribed_temperature(temperature, 300.0, on=exterior)
    coordinate = ufl.SpatialCoordinate(domain)
    normalized_x = coordinate[0] / float(width)
    source = float(amplitude) * ufl.sin(ufl.pi * normalized_x) * ufl.sin(ufl.pi * coordinate[1])
    model.heat_source(source)
    simulation = model.step(target=temperature, name="steady_heat").solve_result()
    observation = learning.regular_grid(
        bounds=((0.025 * width, 0.975 * width), (0.025, 0.975)),
        shape=shape,
        coordinate_unit="m",
    )
    sampled = datasets.fem_observation_sample(
        temperature,
        observation,
        unit="K",
        outside="raise",
    )
    points = observation.points()
    source_values = (
        float(amplitude)
        * np.sin(np.pi * points[:, 0] / float(width))
        * np.sin(np.pi * points[:, 1])
    )
    return source_values, sampled.values.reshape(-1) - 300.0, points, simulation


def build_dataset(widths, amplitudes, *, shape, prefix):
    sources, responses, coordinates, case_ids, metadata = [], [], [], [], []
    for width in widths:
        for amplitude in amplitudes:
            source, response, points, simulation = _solve(width, amplitude, shape=shape)
            sources.append(source[None, :])
            responses.append(response[None, :])
            coordinates.append(points)
            case_ids.append(f"{prefix}-w-{width:.3f}-q-{amplitude:.0f}")
            metadata.append(
                {
                    "width": float(width),
                    "source_amplitude": float(amplitude),
                    "solver_status": simulation.status,
                }
            )
    point_count = int(np.prod(shape))
    source_encoding = learning.FieldEncoding(
        name="heat_source",
        role="input",
        unit="W/m^3",
        representation="point_samples",
        shape=(1, point_count),
        mesh_policy="registered_mesh_family",
    )
    response_encoding = learning.FieldEncoding(
        name="temperature_rise",
        role="output",
        unit="K",
        representation="point_samples",
        shape=(1, point_count),
        mesh_policy="registered_mesh_family",
    )
    return (
        datasets.ScientificFieldDataset(
            case_ids=tuple(case_ids),
            encodings=(source_encoding, response_encoding),
            fields={
                "heat_source": np.asarray(sources),
                "temperature_rise": np.asarray(responses),
            },
            coordinates={"observation_points": np.asarray(coordinates)},
            case_metadata=tuple(metadata),
            name=f"gino_{prefix}_registered_plate_family",
            metadata={
                "generator": "AgentFEM steady heat transfer",
                "topology": "registered rectangular observation grid",
                "coordinate_system": "cartesian",
                "coordinate_unit": "m",
            },
        ),
        source_encoding,
        response_encoding,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/gino_geometry_operator"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    shape = (6, 6) if arguments.smoke else (16, 16)
    amplitudes = (7.5e4, 1.5e5) if arguments.smoke else tuple(np.linspace(5.0e4, 2.0e5, 5))
    training, source, response = build_dataset(
        (0.8, 1.0, 1.2), amplitudes, shape=shape, prefix="train"
    )
    validation, _, _ = build_dataset((0.9, 1.1), (1.0e5,), shape=shape, prefix="validation")
    geometry_test, _, _ = build_dataset(
        (0.85, 1.15), (1.25e5,), shape=shape, prefix="geometry-test"
    )
    specification = learning.NeuralOperatorSpec(
        architecture="gino",
        inputs=(source,),
        outputs=(response,),
        boundary_encoding="explicit point coordinates and isothermal exterior",
        required_checks=("held_out_field_error", "geometry_transfer"),
    )
    extensions.load_extension("agentfem-learning.neuraloperator")
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="variable_geometry_heat_operator",
    )
    result = model.step(
        target=specification,
        dataset=training,
        validation_dataset=validation,
        test_dataset=geometry_test,
        input_geometry="observation_points",
        coordinate_system="cartesian",
        coordinate_unit="m",
        latent_shape=(6, 6) if arguments.smoke else (16, 16),
        n_modes=(3, 3) if arguments.smoke else (8, 8),
        hidden_channels=8 if arguments.smoke else 32,
        n_layers=2 if arguments.smoke else 4,
        input_radius=0.45 if arguments.smoke else 0.2,
        output_radius=0.45 if arguments.smoke else 0.2,
        epochs=20 if arguments.smoke else 250,
        batch_size=2,
        patience=10 if arguments.smoke else 40,
        relative_l2_tolerance=0.25,
        progress=True,
        output=output / "training",
    ).solve_result()
    training.write(output / "training_dataset")
    validation.write(output / "validation_dataset")
    geometry_test.write(output / "geometry_test_dataset")
    print(result.format())
    return result


if __name__ == "__main__":
    main()
