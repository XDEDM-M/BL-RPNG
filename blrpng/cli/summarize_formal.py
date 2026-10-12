from __future__ import annotations

import argparse
import json

from blrpng.analysis.formal import formal_seed_summary
from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one completed formal seed.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or (
        f"artifacts/reports/{config['experiment']['name']}_seed{args.seed}_summary.json"
    )
    with workspace_audit("summarize_formal"):
        payload = formal_seed_summary(
            experiment_name=config["experiment"]["name"],
            seed=args.seed,
            budgets=[int(value) for value in config["data"]["b_budgets"]],
        )
        write_json(output, payload)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "a_path_pretraining_improvement": payload[
                    "a_path_pretraining_improvement"
                ],
                "effects": payload["effects"],
                "budget4_main_relative_to_budget8_ordinary": payload[
                    "budget4_main_relative_to_budget8_ordinary"
                ],
                "output": output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
