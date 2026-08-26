"""Run the public single- or two-crack XDEM-D promotion candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentfem import extensions, fracture, models, studies

from agentfem_learning.neural_fields import xdem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("xvem", "center", "two"), default="xvem"
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lbfgs-steps", type=int, default=12)
    parser.add_argument("--domain-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("outputs/xdem_benchmark"))
    args = parser.parse_args()

    extensions.load_extension("agentfem-learning.xdem")
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        assumption="plane_strain",
    )
    if args.case == "xvem":
        problem = xdem.xvem_mixed_mode_domain_problem(material)
    else:
        factory = (
            xdem.center_crack_domain_problem
            if args.case == "center"
            else xdem.two_collinear_cracks_domain_problem
        )
        problem = factory(material, remote_stress=1000.0)
    specification = xdem.finite_domain_spec(
        problem,
        domain_samples=args.domain_samples,
        boundary_samples=256,
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        name=f"xdem_{args.case}_crack_promotion_candidate",
    )
    result = model.step(
        target=specification,
        epochs=args.epochs,
        lbfgs_steps=args.lbfgs_steps,
        seed=args.seed,
        progress=True,
        output=args.output / args.case,
    ).solve_result()
    comparison = result.metadata["published_benchmark"]
    print(f"published benchmark: {comparison['status']}")
    for tip in comparison["tips"]:
        print(
            f"  {tip['tip_id']}: K_I={tip['actual_k_i']:.6g}, "
            f"K_II={tip['actual_k_ii']:.6g}, accepted={tip['accepted']}"
        )


if __name__ == "__main__":
    main()
