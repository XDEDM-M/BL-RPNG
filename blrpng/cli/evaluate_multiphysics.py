from __future__ import annotations

import argparse
import json

import torch

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.multiphysics import evaluate_multiphysics_index
from blrpng.paths import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen multiphysics v4 query index.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    equation = str(config["data"]["equation"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with workspace_audit("evaluate_multiphysics") as audit:
        model, _ = load_checkpoint_model(args.checkpoint, device=device)
        metrics = evaluate_multiphysics_index(
            model, args.index, device=device, equation=equation,
            equation_parameters=config["data"], horizon=args.horizon,
        )
        output = write_json(args.output, {
            "schema": "blrpng_multiphysics_evaluation_v1",
            "frozen_index": args.index,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "metrics": metrics,
        })
        audit["output"] = str(output)
    print(json.dumps({"output": str(output), "primary": metrics["primary_rollout_relative_l2"]}, indent=2))


if __name__ == "__main__":
    main()

