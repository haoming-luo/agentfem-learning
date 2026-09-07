"""Generate AgentFEM heat fields and train a verified FNO mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ufl
from agentfem import constitutive, datasets, extensions, fields, learning, mesh, models, studies
from mpi4py import MPI


def _solve_heat_case(amplitude: float, center_x: float, *, grid):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (16, 16),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=domain,
        name="heated_plate",
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
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        ),
        name="isothermal_boundary",
        tag=1,
    )
    model.prescribed_temperature(temperature, 300.0, on=exterior)
    coordinate = ufl.SpatialCoordinate(domain)
    width = 0.12
    source = float(amplitude) * ufl.exp(
        -((coordinate[0] - float(center_x)) ** 2 + (coordinate[1] - 0.5) ** 2) / width**2
    )
    model.heat_source(source)
    simulation = model.step(target=temperature, name="steady_heat").solve_result()
    observation = datasets.fem_observation_sample(
        temperature,
        grid,
        unit="K",
        outside="raise",
    )
    xx, yy = np.meshgrid(*grid.axes, indexing="ij")
    source_values = float(amplitude) * np.exp(
        -((xx - float(center_x)) ** 2 + (yy - 0.5) ** 2) / width**2
    )
    return source_values, observation.values, simulation


def _build_field_dataset(
    *,
    shape,
    amplitudes,
    centers,
    name: str,
    case_prefix: str,
):
    grid = learning.regular_grid(
        bounds=((0.025, 0.975), (0.025, 0.975)),
        shape=shape,
        coordinate_unit="m",
    )
    sources = []
    temperatures = []
    case_ids = []
    parameters = []
    case_metadata = []
    for amplitude in amplitudes:
        for center_x in centers:
            source, temperature, simulation = _solve_heat_case(
                amplitude,
                center_x,
                grid=grid,
            )
            case_id = f"{case_prefix}-q-{amplitude:.0f}-x-{center_x:.3f}"
            sources.append(source[None, ...])
            temperatures.append(temperature[None, ...])
            case_ids.append(case_id)
            parameters.append((amplitude, center_x))
            case_metadata.append(
                {
                    "solver_status": simulation.status,
                    "study": "steady_heat_transfer",
                }
            )
    source_encoding = learning.FieldEncoding(
        name="heat_source",
        role="input",
        unit="W/m^3",
        representation="structured_grid",
        shape=(1, *shape),
        mesh_policy="mesh_independent_coordinates",
    )
    temperature_encoding = learning.FieldEncoding(
        name="temperature_rise",
        role="output",
        unit="K",
        representation="structured_grid",
        shape=(1, *shape),
        mesh_policy="mesh_independent_coordinates",
    )
    values = np.asarray(parameters)
    return datasets.ScientificFieldDataset(
        case_ids=tuple(case_ids),
        encodings=(source_encoding, temperature_encoding),
        fields={
            "heat_source": np.asarray(sources),
            "temperature_rise": np.asarray(temperatures) - 300.0,
        },
        parameters={
            "source_amplitude": values[:, 0],
            "source_center_x": values[:, 1],
        },
        case_metadata=tuple(case_metadata),
        name=name,
        metadata={
            "generator": "AgentFEM steady heat transfer",
            "observation_grid": grid.summary(),
        },
    ), source_encoding, temperature_encoding


def build_dataset(*, smoke: bool = False):
    shape = (8, 8) if smoke else (20, 20)
    dataset, source_encoding, temperature_encoding = _build_field_dataset(
        shape=shape,
        amplitudes=np.linspace(5.0e4, 2.0e5, 3 if smoke else 5),
        centers=np.linspace(0.25, 0.75, 4 if smoke else 6),
        name="agentfem_heat_source_operator",
        case_prefix="train",
    )
    specification = learning.NeuralOperatorSpec(
        architecture="fno",
        inputs=(source_encoding,),
        outputs=(temperature_encoding,),
        boundary_encoding="fixed_isothermal_exterior",
        parameter_inputs=("source_amplitude", "source_center_x"),
        required_checks=("held_out_field_error", "resolution_transfer"),
    )
    return dataset, specification


def build_resolution_dataset(*, smoke: bool = False):
    dataset, _, _ = _build_field_dataset(
        shape=(12, 12) if smoke else (28, 28),
        amplitudes=(7.5e4, 1.75e5),
        centers=(0.35, 0.65),
        name="agentfem_heat_source_operator_resolution_transfer",
        case_prefix="resolution",
    )
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/fno_heat_operator"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    extensions.load_extension("agentfem-learning.neuraloperator")
    dataset, specification = build_dataset(smoke=arguments.smoke)
    resolution_dataset = build_resolution_dataset(smoke=arguments.smoke)
    dataset.write(output / "field_dataset")
    resolution_dataset.write(output / "resolution_dataset")
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        name="heat_source_to_temperature_operator",
    )
    result = model.step(
        target=specification,
        dataset=dataset,
        test_dataset=resolution_dataset,
        n_modes=(3, 3) if arguments.smoke else (8, 8),
        hidden_channels=16 if arguments.smoke else 32,
        n_layers=3 if arguments.smoke else 4,
        epochs=100 if arguments.smoke else 250,
        batch_size=4,
        learning_rate=5.0e-3,
        patience=30 if arguments.smoke else 40,
        relative_l2_tolerance=0.10,
        progress=True,
        output=output / "training",
    ).solve_result()
    print(result.format())
    print(f"Field dataset: {output / 'field_dataset' / 'manifest.json'}")
    print(
        "Resolution dataset: "
        f"{output / 'resolution_dataset' / 'manifest.json'}"
    )
    print(f"Training result: {output / 'training' / 'result.json'}")
    return result


if __name__ == "__main__":
    main()
