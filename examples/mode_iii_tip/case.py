"""Run the first AgentFEM -> XDEM -> SimulationResult reference workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentfem import extensions, models, studies

from agentfem_learning.neural_fields.xdem import mode_iii_tip_spec


def run(*, output: Path, epochs: int, device: str, dtype: str, progress: bool):
    extensions.load_extension("agentfem-learning.xdem")

    study = studies.static_solid(
        dimension=2,
        assumption="plane_strain",
        name="mode_iii_neural_field",
    )
    model = models.create(study=study, name="williams_mode_iii_tip")
    neural_field = mode_iii_tip_spec()

    step = model.step(
        target=neural_field,
        epochs=epochs,
        device=device,
        dtype=dtype,
        progress=progress,
        output=output,
    )
    return step.solve_result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/mode_iii_tip"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--progress", action="store_true")
    options = parser.parse_args()
    result = run(
        output=options.output,
        epochs=options.epochs,
        device=options.device,
        dtype=options.dtype,
        progress=options.progress,
    )
    print(result.format())


if __name__ == "__main__":
    main()
