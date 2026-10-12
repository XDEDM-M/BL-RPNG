from __future__ import annotations

import argparse
import json

import torch

from blrpng.analysis.reporting import best_final_stability
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", default="artifacts/reports/training_stability_cost.json")
    args = parser.parse_args()
    rows = []
    with workspace_audit("summarize_training"):
        for value in args.checkpoint:
            path = resolve_in_project(value, must_exist=True)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "checkpoint": str(path),
                    "seed": checkpoint["training_seed"],
                    "objective": checkpoint["objective"],
                    "boundary_budget": checkpoint.get("boundary_budget"),
                    "stability": best_final_stability(checkpoint["history"]),
                    "training_seconds": checkpoint["elapsed_seconds"],
                }
            )
        write_json(args.output, rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
