from __future__ import annotations

import argparse
import json

import torch

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.schema import load_dataset
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.metrics import evaluate_bundle
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every frozen fine-grid split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output", default="artifacts/reports/allen_cahn_blind_suite.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    name = config["experiment"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = (
        "b_train",
        "blind_seen",
        "blind_unseen",
        "blind_frequency_ood",
        "blind_amplitude_ood",
    )
    with workspace_audit("evaluate_suite"):
        model, checkpoint = load_checkpoint_model(args.checkpoint, device=device)
        reports = {}
        coarse_train_path = resolve_in_project(
            f"artifacts/data/{name}_b_train_coarse.pt"
        )
        coarse_train = load_dataset(coarse_train_path) if coarse_train_path.exists() else None
        for split in splits:
            path = resolve_in_project(
                f"artifacts/data/{name}_{split}_fine.pt", must_exist=True
            )
            bundle = load_dataset(path, modes2=int(checkpoint["model_config"]["modes2"]))
            if bundle.meta["manifest_sha256"] != checkpoint["manifest_sha256"]:
                raise ValueError(f"Manifest mismatch for {split}")
            reports[split] = evaluate_bundle(
                model,
                bundle,
                device=device,
                horizon=int(data["num_blind_time_steps"]),
                near_boundary_width=4,
                equation={
                    "dt": float(data["dt"]),
                    "epsilon": float(data["eps"]),
                    "gamma": float(data["gamma"]),
                },
                coarse_bundle=coarse_train if split == "b_train" else None,
                coarse_initial_resolution=int(data["b_coarse_grid"]),
            )
        payload = {
            "checkpoint": args.checkpoint,
            "primary_split": "blind_unseen",
            "primary_metric": reports["blind_unseen"]["primary_rollout_rel_l2"],
            "reports": reports,
        }
        write_json(args.output, payload)
    print(
        json.dumps(
            {
                "checkpoint": payload["checkpoint"],
                "primary_split": payload["primary_split"],
                "primary_metric": payload["primary_metric"],
                "split_metrics": {
                    split: report["primary_rollout_rel_l2"]
                    for split, report in reports.items()
                },
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
