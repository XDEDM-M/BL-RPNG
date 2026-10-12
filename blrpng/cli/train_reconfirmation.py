from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock
from blrpng.training.adapters import (
    compact_full_checkpoint_for_inference,
    compact_task_checkpoint,
)
from blrpng.training.engine import train


METHODS = {
    "nlt": ("source_blrpng_checkpoint", "final_only", "nlt"),
    "ordinary_nlt": ("source_final_checkpoint", "final_only", "nlt"),
    "nlt_path": ("source_blrpng_checkpoint", "blrpng_v2", "nlt"),
    "nlt_commutator": ("source_blrpng_checkpoint", "commutator_only", "nlt"),
    "nlt_commutator_defect": ("source_blrpng_checkpoint", "pino", "nlt"),
    "lr_scratch": (None, "final_only", "full"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt one frozen Allen-Cahn reconfirmation task using LR data only."
    )
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    parser.add_argument("--training-config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--tag",
        help="Optional artifact suffix for explicitly exploratory convergence runs.",
    )
    args = parser.parse_args()
    protocol = load_config(args.config)
    base = load_config(args.training_config)
    require_prerequisite_gate(
        "artifacts/manifests/allen_cahn_reconfirmation_lr_data_gate.json"
    )
    require_prerequisite_gate(protocol["gates"]["required_method_freeze_manifest"])
    task_count = len(protocol["blind_protocol"]["families"]) * int(
        protocol["blind_protocol"]["tasks_per_family"]
    )
    if not 0 <= args.task < task_count:
        raise ValueError(f"task must be in [0, {task_count}).")
    if args.tag is not None and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.tag) is None:
        raise ValueError("tag must contain only lowercase letters, digits, '_' or '-'.")
    initialization_key, objective, adapter = METHODS[args.method]
    initialization = base["data"][initialization_key] if initialization_key else None
    local = copy.deepcopy(base)
    local["experiment"]["name"] = "allen_cahn_blind_reconfirmation"
    local["experiment"]["training_seed"] = args.seed
    local["training"]["adapter"] = {
        **local["training"].get("adapter", {}),
        "strategy": adapter,
    }
    if args.epochs is not None:
        local["training"]["epochs"] = int(args.epochs)
        # Keep the frozen late LR drop meaningful for shorter smoke runs.
        local["training"]["lr_drop_epoch"] = min(
            int(local["training"]["lr_drop_epoch"]), max(int(args.epochs) - 2, 1)
        )
    if args.lr is not None:
        local["training"]["lr"] = float(args.lr)
    train_path = (
        f"artifacts/data/reconfirmation/task{args.task:02d}_support_64_m{args.budget}.index.json"
    )
    validation_path = (
        f"artifacts/data/reconfirmation/task{args.task:02d}_validation_64_all.index.json"
    )
    stem = (
        f"reconfirmation_task{args.task:02d}_m{args.budget}_s{args.seed}_{args.method}"
    )
    if args.tag:
        stem += f"_{args.tag}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_reconfirmation") as audit:
            result = train(
                local,
                seed=args.seed,
                objective=objective,
                initialization_checkpoint=initialization,
                output_checkpoint=f"artifacts/checkpoints/reconfirmation/{stem}.pt",
                train_data_path=train_path,
                validation_data_paths=[validation_path],
            )
            if adapter == "nlt":
                compact = compact_task_checkpoint(
                    result["checkpoint"], base_checkpoint=initialization
                )
            else:
                compact = compact_full_checkpoint_for_inference(result["checkpoint"])
            result["checkpoint"] = str(compact)
            audit.update({
                "task": args.task,
                "budget": args.budget,
                "seed": args.seed,
                "method": args.method,
                "objective": objective,
                "checkpoint": result["checkpoint"],
                "hr_query_used": False,
                "exploratory_tag": args.tag,
            })
    print(json.dumps({
        "checkpoint": result["checkpoint"],
        "best_validation_metric": result["best_validation_metric"],
        "hr_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
