"""Run the first vector-elastic XDEM reference through AgentFEM."""

from __future__ import annotations

from pathlib import Path

from agentfem import extensions, models, studies

from agentfem_learning.neural_fields.xdem import vector_tip_spec


def main() -> None:
    extensions.load_extension("agentfem-learning.xdem")
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        name="mixed_mode_vector_reference",
    )
    result = model.step(
        target=vector_tip_spec(
            assumption="plane_stress",
            k_i=1.0,
            k_ii=0.5,
        ),
        epochs=500,
        output=Path("outputs/vector_mixed_mode_tip"),
        seed=2026,
    ).solve_result()
    print(result.summary())


if __name__ == "__main__":
    main()
