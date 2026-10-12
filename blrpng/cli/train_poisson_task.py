from __future__ import annotations

import argparse
import copy
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock
from blrpng.training.adapters import compact_full_checkpoint_for_inference, compact_task_checkpoint
from blrpng.training.engine import train


METHODS = {
    "ordinary_nlt": (
        "artifacts/checkpoints/poisson/poisson_source_s42_final_only.inference.pt",
        "nlt", "lifted",
    ),
    "path_nlt": (
        "artifacts/checkpoints/poisson/poisson_source_s42_blrpng.inference.pt",
        "nlt", "lifted",
    ),
    "lr_scratch": (None, "full", "lifted"),
    "boundary_augmented_lr_scratch": (None, "full", "boundary_augmented"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Poisson target-boundary adapter.")
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.task < 30:
        raise ValueError("Poisson task must be in [0, 30).")
    require_prerequisite_gate(
        "artifacts/manifests/poisson_negative_control_method_freeze.json"
    )
    config = copy.deepcopy(load_config(args.config))
    initialization, adapter, representation = METHODS[args.method]
    config["experiment"]["seed"] = int(args.seed)
    config["training"]["adapter"]["strategy"] = adapter
    stem = f"poisson_task{args.task:02d}_m{args.budget}_s{args.seed}_{args.method}"
    train_index = (
        f"artifacts/data/poisson_negative_control/"
        f"task{args.task:02d}_support_64_m{args.budget}.index.json"
    )
    validation_index = (
        f"artifacts/data/poisson_negative_control/"
        f"task{args.task:02d}_validation_64_all.index.json"
    )
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_poisson_task") as audit:
            result = train(
                config,
                seed=args.seed,
                objective="final_only",
                initialization_checkpoint=initialization,
                representation_override=representation,
                output_checkpoint=f"artifacts/checkpoints/poisson/{stem}.pt",
                train_data_path=train_index,
                validation_data_paths=[validation_index],
            )
            if adapter == "nlt":
                checkpoint = compact_task_checkpoint(
                    result["checkpoint"], base_checkpoint=initialization
                )
            else:
                checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            audit.update({
                "task": args.task, "budget": args.budget, "method": args.method,
                "seed": args.seed, "checkpoint": str(checkpoint),
                "target_query_used": False,
            })
    print(json.dumps({
        "task": args.task, "method": args.method, "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
        "target_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
