from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--objective",
        choices=(
            "final_only",
            "blrpng",
            "blrpng_safe",
            "blrpng_annealed",
            "legacy_path",
            "dual_stage",
            "blrpng_v2",
            "blrpng_plus",
        ),
    )
    parser.add_argument("--initialize-from")
    parser.add_argument("--resume")
    parser.add_argument("--output")
    parser.add_argument("--train-data")
    parser.add_argument("--validation-data", action="append")
    parser.add_argument("--boundary-budget", type=int)
    parser.add_argument("--representation", choices=("lifted", "raw_legacy"))
    args = parser.parse_args()
    config = load_config(args.config)
    with workspace_audit("train") as audit:
        result = train(
            config,
            seed=args.seed,
            objective=args.objective,
            initialization_checkpoint=args.initialize_from,
            resume_checkpoint=args.resume,
            output_checkpoint=args.output,
            train_data_path=args.train_data,
            validation_data_paths=args.validation_data,
            boundary_budget=args.boundary_budget,
            representation_override=args.representation,
        )
        audit["checkpoint"] = result["checkpoint"]
    print(
        json.dumps(
            {
                "checkpoint": result["checkpoint"],
                "best_validation_metric": result["best_validation_metric"],
                "epochs": len(result["history"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
