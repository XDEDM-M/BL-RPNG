from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.multiphysics import evaluate_multiphysics_index
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json
from blrpng.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train converged LR scratch baselines and immediately evaluate frozen query."
    )
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="Exclusive task endpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        choices=("lr_scratch_converged", "boundary_augmented_lr_scratch"),
        default="lr_scratch_converged",
    )
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 30:
        raise ValueError("Require 0 <= start < end <= 30.")
    require_prerequisite_gate(
        "artifacts/manifests/poisson_negative_control_query_data_gate.json"
    )
    base = load_config(args.config)
    representation = (
        "boundary_augmented"
        if args.method == "boundary_augmented_lr_scratch"
        else "lifted"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completed = []
    with workspace_audit(f"run_poisson_scratch_{args.start}_{args.end}") as audit:
        for task in range(args.start, args.end):
            result_stem = (
                f"seed42_lr_scratch_task{task:02d}"
                if args.method == "lr_scratch_converged"
                else f"seed42_{args.method}_task{task:02d}"
            )
            result_path = ensure_artifact_path(
                f"artifacts/experiments/poisson_negative_control/"
                f"{result_stem}.json"
            )
            if result_path.exists():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("checkpoint_deleted_after_evaluation") is True:
                    completed.append(existing)
                    continue
            local = copy.deepcopy(base)
            local["experiment"]["seed"] = int(args.seed)
            local["training"].update({
                "epochs": 100,
                "lr_scheduler": "cosine",
                "minimum_lr": 1.0e-5,
                "early_stopping_patience": 15,
                "minimum_epochs": 40,
                "recovery_interval": 25,
            })
            local["training"]["adapter"]["strategy"] = "full"
            stem = f"poisson_task{task:02d}_m4_s{args.seed}_{args.method}"
            training = train(
                local,
                seed=args.seed,
                objective="final_only",
                representation_override=representation,
                output_checkpoint=f"artifacts/checkpoints/poisson/{stem}.pt",
                train_data_path=(
                    f"artifacts/data/poisson_negative_control/"
                    f"task{task:02d}_support_64_m4.index.json"
                ),
                validation_data_paths=[
                    f"artifacts/data/poisson_negative_control/"
                    f"task{task:02d}_validation_64_all.index.json"
                ],
            )
            checkpoint_path = resolve_in_project(training["checkpoint"], must_exist=True)
            checkpoint_sha = sha256_file(checkpoint_path)
            model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
            query_metrics = {}
            for resolution in (128, 256):
                query_metrics[str(resolution)] = evaluate_multiphysics_index(
                    model,
                    (
                        f"artifacts/data/poisson_negative_control/"
                        f"task{task:02d}_query_{resolution}_all.index.json"
                    ),
                    device=device,
                    equation="poisson",
                    equation_parameters=base["data"],
                    horizon=1,
                )
            payload = {
                "schema": "blrpng_poisson_scratch_evaluation_v1",
                "task": task,
                "seed": args.seed,
                "method": args.method,
                "support_budget": 4,
                "training_protocol": {
                    "epochs_max": 100,
                    "minimum_epochs": 40,
                    "early_stopping_patience": 15,
                    "scheduler": "cosine",
                    "minimum_lr": 1.0e-5,
                    "query_used_for_checkpoint_selection": False,
                },
                "epochs_completed": int(checkpoint["epoch"]) + 1,
                "stopped_early": bool(checkpoint.get("stopped_early", False)),
                "best_validation_metric": float(checkpoint["best_validation_metric"]),
                "training_seconds": float(checkpoint["elapsed_seconds"]),
                "checkpoint_sha256_before_deletion": checkpoint_sha,
                "query": query_metrics,
                "checkpoint_deleted_after_evaluation": True,
            }
            write_json(result_path, payload)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint_path.unlink()
            recovery = ensure_artifact_path(f"artifacts/checkpoints/{stem}_recovery.pt")
            recovery.unlink(missing_ok=True)
            completed.append(payload)
        audit.update({
            "tasks": list(range(args.start, args.end)), "method": args.method,
            "results": len(completed),
            "full_scratch_checkpoints_retained": False,
        })
    print(json.dumps({
        "tasks": [row["task"] for row in completed],
        "method": args.method,
        "mean_256_relative_l2": sum(
            row["query"]["256"]["primary_rollout_relative_l2"] for row in completed
        ) / len(completed),
        "mean_epochs": sum(row["epochs_completed"] for row in completed) / len(completed),
        "full_checkpoints_retained": False,
    }, indent=2))


if __name__ == "__main__":
    main()
