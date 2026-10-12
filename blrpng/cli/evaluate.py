from __future__ import annotations

import argparse
import json

import torch

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.schema import load_dataset
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.metrics import evaluate_bundle
from blrpng.paths import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data")
    parser.add_argument("--raw-weights", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["training"]["checkpoint_path"]
    data_path = args.data or config["data"]["test_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with workspace_audit("evaluate") as audit:
        model, checkpoint = load_checkpoint_model(
            checkpoint_path,
            device=device,
            use_ema=not args.raw_weights,
        )
        bundle = load_dataset(data_path, modes2=int(checkpoint["model_config"]["modes2"]))
        if checkpoint["manifest_sha256"] != bundle.meta["manifest_sha256"]:
            raise ValueError("Checkpoint and evaluation data use different manifests.")
        coarse = (
            load_dataset(config["data"]["validation_path"])
            if "validation_path" in config["data"]
            else None
        )
        report = evaluate_bundle(
            model,
            bundle,
            device=device,
            horizon=int(config["evaluation"]["horizon"]),
            near_boundary_width=int(config["evaluation"]["near_boundary_width"]),
            equation={
                "dt": float(config["data"]["dt"]),
                "epsilon": float(config["data"]["eps"]),
                "gamma": float(config["data"]["gamma"]),
            },
            coarse_bundle=coarse,
            coarse_initial_resolution=int(
                config["data"].get(
                    "b_coarse_grid",
                    config["data"].get("coarse_grid", 64),
                )
            ),
        )
        report.update(
            {
                "checkpoint": checkpoint_path,
                "data": data_path,
                "weights": "raw" if args.raw_weights else "ema",
                "training_seed": checkpoint["training_seed"],
                "representation": checkpoint["representation"],
                "objective": checkpoint["objective"],
            }
        )
        write_json(config["evaluation"]["output_path"], report)
        audit["primary_rollout_rel_l2"] = report["primary_rollout_rel_l2"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
