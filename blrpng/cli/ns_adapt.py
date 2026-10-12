"""Adapt a frozen SfePy-NS source model to one target boundary using LR data."""
from __future__ import annotations

import argparse
import copy
import json
import math

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, subset_index
from blrpng.paths import exclusive_artifact_lock
from blrpng.training.adapters import compact_full_checkpoint_for_inference, compact_task_checkpoint
from blrpng.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navier_stokes_sfepy_seed42.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--adapter", choices=("nlt", "lora", "full"), default="nlt")
    parser.add_argument("--objective", choices=("final_only", "newton_path"), default="final_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if not 0 <= args.task < 24:
        raise ValueError("SfePy-NS task must be in [0, 24).")
    if args.source_checkpoint is None and args.adapter != "full":
        raise ValueError("NLT/LoRA adaptation requires --source-checkpoint.")
    require_prerequisite_gate("artifacts/manifests/navier_stokes_sfepy_solver_gate.json")
    config = copy.deepcopy(load_config(args.config))
    config["experiment"]["seed"] = int(args.seed)
    config["data"]["representation"] = "lifted"
    if args.source_checkpoint is not None:
        source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
        for key in ("in_channels", "out_channels", "condition_channels", "modes1", "modes2", "width", "layers", "probe_mode", "field_scales", "gauge_channels"):
            if key in source["model_config"]:
                config["model"][key] = source["model_config"][key]
    adaptation = config["training"]["adaptation"]
    config["training"].update(
        {
            "batch_size": int(adaptation["batch_size"]),
            "gradient_accumulation_steps": 1,
            "lr": float(adaptation["lr"]),
            "minimum_optimizer_steps": int(adaptation["minimum_optimizer_steps"]),
            "max_optimizer_steps": int(adaptation["max_optimizer_steps"]),
            "plateau_patience": int(adaptation["plateau_patience"]),
            "early_stopping_patience": int(adaptation["early_stopping_patience"]),
            "lr_scheduler": "plateau",
            "recovery_interval": 1,
        }
    )
    config["training"]["adapter"]["strategy"] = args.adapter
    stem = f"ns_sfepy_task{args.task:02d}_m{args.budget}_s{args.seed}_{args.adapter}_{args.objective}"
    parent_support = "artifacts/data/navier_stokes_sfepy/target_support_64.index.json"
    parent_validation = "artifacts/data/navier_stokes_sfepy/target_validation_64.index.json"
    train_index = subset_index(
        parent_support,
        f"artifacts/data/navier_stokes_sfepy/task{args.task:02d}_support_64_m{args.budget}.index.json",
        target_task_id=args.task,
        max_rollouts=args.budget,
    )
    validation_index = subset_index(
        parent_validation,
        f"artifacts/data/navier_stokes_sfepy/task{args.task:02d}_validation_64.index.json",
        target_task_id=args.task,
    )
    transitions = sum(int(row["steps"]) for row in read_index(train_index)["shards"])
    updates_per_epoch = math.ceil(transitions / int(config["training"]["batch_size"]))
    config["training"]["epochs"] = math.ceil(
        int(config["training"]["max_optimizer_steps"]) / max(updates_per_epoch, 1)
    )
    internal_objective = "final_only" if args.objective == "final_only" else "blrpng_safe"
    with exclusive_artifact_lock(stem):
        with workspace_audit("ns_adapt") as audit:
            result = train(
                config,
                seed=args.seed,
                objective=internal_objective,
                initialization_checkpoint=args.source_checkpoint,
                resume_checkpoint=args.resume,
                output_checkpoint=f"artifacts/checkpoints/navier_stokes_sfepy/{stem}.pt",
                train_data_path=train_index,
                validation_data_paths=[validation_index],
            )
            if args.adapter in {"nlt", "lora"}:
                checkpoint = compact_task_checkpoint(
                    result["checkpoint"], base_checkpoint=args.source_checkpoint
                )
            else:
                checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            audit.update(
                {
                    "task": args.task,
                    "budget": args.budget,
                    "adapter": args.adapter,
                    "objective": args.objective,
                    "checkpoint": str(checkpoint),
                    "query_used": False,
                }
            )
    print(
        json.dumps(
            {
                "task": args.task,
                "checkpoint": str(checkpoint),
                "best_validation_metric": result["best_validation_metric"],
                "query_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
