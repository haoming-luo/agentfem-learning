"""Train a GINO across AgentFEM heat-transfer domains of different width."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ufl
from agentfem import (
    constitutive,
    datasets,
    extensions,
    fields,
    learning,
    mesh,
    models,
    studies,
    verification,
)
from mpi4py import MPI

from agentfem_learning.neural_operators.neuraloperator import OperatorCheck


def _solve(
    width: float,
    amplitude: float,
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
):
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
    input_grid = learning.regular_grid(
        bounds=((0.025 * width, 0.975 * width), (0.025, 0.975)),
        shape=input_shape,
        coordinate_unit="m",
    )
    output_grid = learning.regular_grid(
        bounds=((0.025 * width, 0.975 * width), (0.025, 0.975)),
        shape=output_shape,
        coordinate_unit="m",
    )
    sampled = datasets.fem_observation_sample(
        temperature,
        output_grid,
        unit="K",
        outside="raise",
    )
    input_points = input_grid.points()
    source_values = (
        float(amplitude)
        * np.sin(np.pi * input_points[:, 0] / float(width))
        * np.sin(np.pi * input_points[:, 1])
    )
    return (
        source_values,
        sampled.values.reshape(-1) - 300.0,
        input_points,
        output_grid.points(),
        simulation,
    )


def build_dataset(widths, amplitudes, *, input_shape, output_shape, prefix):
    sources, responses, input_points, output_points = [], [], [], []
    case_ids, metadata, plate_widths = [], [], []
    for width in widths:
        for amplitude in amplitudes:
            source, response, source_points, query_points, simulation = _solve(
                width,
                amplitude,
                input_shape=input_shape,
                output_shape=output_shape,
            )
            sources.append(source[None, :])
            responses.append(response[None, :])
            input_points.append(source_points)
            output_points.append(query_points)
            plate_widths.append(float(width))
            case_ids.append(f"{prefix}-w-{width:.3f}-q-{amplitude:.0f}")
            metadata.append(
                {
                    "width": float(width),
                    "source_amplitude": float(amplitude),
                    "solver_status": simulation.status,
                }
            )
    input_count = int(np.prod(input_shape))
    output_count = int(np.prod(output_shape))
    source_encoding = learning.FieldEncoding(
        name="heat_source",
        role="input",
        unit="W/m^3",
        representation="point_samples",
        shape=(1, input_count),
        mesh_policy="registered_mesh_family",
    )
    response_encoding = learning.FieldEncoding(
        name="temperature_rise",
        role="output",
        unit="K",
        representation="point_samples",
        shape=(1, output_count),
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
            parameters={"plate_width": np.asarray(plate_widths)},
            coordinates={
                "source_points": np.asarray(input_points),
                "response_points": np.asarray(output_points),
            },
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


def _analytical_temperature_check(context):
    conductivity = 10.0
    predicted = context.predictions["temperature_rise"][:, 0, :]
    exact = []
    coordinates = context.validation_dataset.coordinates["response_points"]
    for points, metadata in zip(
        coordinates, context.validation_dataset.case_metadata, strict=True
    ):
        width = float(metadata["width"])
        amplitude = float(metadata["source_amplitude"])
        factor = amplitude / (conductivity * np.pi**2 * (1.0 / width**2 + 1.0))
        exact.append(
            factor * np.sin(np.pi * points[:, 0] / width) * np.sin(np.pi * points[:, 1])
        )
    exact = np.asarray(exact)
    relative_error = float(np.linalg.norm(predicted - exact) / np.linalg.norm(exact))
    return verification.VerificationClaim.compare(
        name="analytical_solution_error",
        observable="relative_l2_temperature_error",
        actual=relative_error,
        expected=0.0,
        reference="separable sine-source Dirichlet heat equation",
        absolute_tolerance=0.25,
        validity_domain="constant conductivity rectangular plates",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/gino_geometry_operator"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_shape = (6, 6)
    output_shape = input_shape
    test_output_shape = (8, 8)
    widths = (0.8, 1.0, 1.2) if arguments.smoke else (0.75, 0.85, 0.95, 1.05, 1.15, 1.25)
    amplitudes = (7.5e4, 1.5e5) if arguments.smoke else (5.0e4, 1.0e5, 1.5e5)
    training, source, response = build_dataset(
        widths,
        amplitudes,
        input_shape=input_shape,
        output_shape=output_shape,
        prefix="train",
    )
    validation, _, _ = build_dataset(
        (0.9, 1.1),
        (1.0e5,) if arguments.smoke else (7.5e4, 1.25e5),
        input_shape=input_shape,
        output_shape=output_shape,
        prefix="validation",
    )
    query_test, _, _ = build_dataset(
        (0.95,),
        (1.25e5,),
        input_shape=input_shape,
        output_shape=test_output_shape,
        prefix="query-test",
    )
    specification = learning.NeuralOperatorSpec(
        architecture="gino",
        inputs=(source,),
        outputs=(response,),
        boundary_encoding="explicit point coordinates and isothermal exterior",
        parameter_inputs=("plate_width",),
        required_checks=(
            "held_out_field_error",
            "geometry_transfer",
            "output_query_transfer",
            "analytical_solution_error",
        ),
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
        test_dataset=query_test,
        input_geometry="source_points",
        output_queries="response_points",
        coordinate_system="cartesian",
        coordinate_unit="m",
        latent_shape=(6, 6),
        n_modes=(3, 3),
        hidden_channels=8 if arguments.smoke else 16,
        n_layers=2,
        input_radius=0.45,
        output_radius=0.45,
        epochs=20 if arguments.smoke else 120,
        batch_size=2 if arguments.smoke else 3,
        learning_rate=5.0e-3 if arguments.smoke else 3.0e-3,
        patience=10 if arguments.smoke else 30,
        relative_l2_tolerance=0.25,
        check_evaluators=(
            OperatorCheck(
                name="analytical_solution_error",
                evaluator=_analytical_temperature_check,
                version="rectangular-sine-heat-v1",
            ),
        ),
        progress=True,
        output=output / "training",
    ).solve_result()
    training.write(output / "training_dataset")
    validation.write(output / "validation_dataset")
    query_test.write(output / "query_test_dataset")
    print(result.format())
    return result


if __name__ == "__main__":
    main()
