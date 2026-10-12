from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock
from blrpng.training.engine import train
from blrpng.training.adapters import compact_task_checkpoint


METHODS = {
    "raw_lr_scratch": (None, "final_only", "full"),
    "boundary_augmented_scratch": (None, "final_only", "full"),
    "lift_conditioned_scratch": (None, "final_only", "full"),
    "lr_scratch": (None, "final_only", "full"),
    "ordinary_final": ("source_final_checkpoint", "final_only", "full"),
    "pathpre_final": ("source_blrpng_checkpoint", "final_only", "full"),
    "pathpre_fixed": ("source_blrpng_checkpoint", "blrpng", "full"),
    "lora": ("source_blrpng_checkpoint", "final_only", "lora"),
    "nlt": ("source_blrpng_checkpoint", "final_only", "nlt"),
    "nlt_mix": ("source_blrpng_checkpoint", "final_only", "nlt_mix"),
    "ordinary_lora": ("source_final_checkpoint", "final_only", "lora"),
    "ordinary_nlt": ("source_final_checkpoint", "final_only", "nlt"),
    "nlt_path": ("source_blrpng_checkpoint", "blrpng_v2", "nlt"),
    "nlt_mix_path": ("source_blrpng_checkpoint", "blrpng_v2", "nlt_mix"),
    "nlt_commutator": ("source_blrpng_checkpoint", "commutator_only", "nlt"),
    "nlt_commutator_defect": ("source_blrpng_checkpoint", "pino", "nlt"),
    "blrpng_v2": ("source_blrpng_checkpoint", "blrpng_v2", "lora"),
    "blrpng_plus": ("source_blrpng_checkpoint", "blrpng_plus", "lora"),
    "blrpng_plus_d1e3": ("source_blrpng_checkpoint", "blrpng_plus", "lora"),
    "blrpng_plus_d1e2": ("source_blrpng_checkpoint", "blrpng_plus", "lora"),
    "blrpng_v2_nlt": ("source_blrpng_checkpoint", "blrpng_v2", "nlt"),
    "blrpng_plus_nlt": ("source_blrpng_checkpoint", "blrpng_plus", "nlt"),
    "hr_few": (None, "final_only", "full"),
    "hr_oracle": (None, "final_only", "full"),
    "hr_oracle_ordinary": ("source_final_checkpoint", "final_only", "full"),
    "hr_oracle_path": ("source_blrpng_checkpoint", "final_only", "full"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one target-boundary adaptation task.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--commutator-weight", type=float)
    parser.add_argument("--defect-weight", type=float)
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--adapter-alpha", type=float)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume")
    parser.add_argument("--adapter-initialization")
    args = parser.parse_args()
    config = load_config(args.config)
    if not 0 <= args.task < int(config["data"]["target_boundaries"]):
        raise ValueError("Target task is outside the frozen calibration task range.")
    initialization_key, objective, adapter = METHODS[args.method]
    is_hr = args.method.startswith("hr_")
    split = "oracle_train" if is_hr else "support_lr"
    resolution = 256 if is_hr else 64
    train_path = f"artifacts/data/tasks/task{args.task:02d}_{split}_{resolution}_m{args.budget}.index.json"
    validation_split = "oracle_validation" if is_hr else "validation_lr"
    validation_path = f"artifacts/data/tasks/task{args.task:02d}_{validation_split}_{resolution}_all.index.json"
    local = copy.deepcopy(config)
    if args.lr is not None:
        local["training"]["lr"] = float(args.lr)
    if args.epochs is not None:
        local["training"]["epochs"] = int(args.epochs)
    if args.commutator_weight is not None:
        local["training"]["commutator_weight"] = float(args.commutator_weight)
    if args.defect_weight is not None:
        local["training"]["defect_weight"] = float(args.defect_weight)
    local["training"]["adapter"] = {
        **local["training"].get("adapter", {}), "strategy": adapter,
    }
    if args.adapter_rank is not None:
        local["training"]["adapter"]["rank"] = int(args.adapter_rank)
    if args.adapter_alpha is not None:
        local["training"]["adapter"]["alpha"] = float(args.adapter_alpha)
    if args.method == "raw_lr_scratch":
        local["data"]["representation"] = "raw_legacy"
        local["model"]["in_channels"] = int(local["model"]["out_channels"]) + 2
    elif args.method == "boundary_augmented_scratch":
        local["data"]["representation"] = "boundary_augmented"
        local["model"]["in_channels"] = 3 * int(local["model"]["out_channels"]) + 2
    elif args.method == "lift_conditioned_scratch":
        local["data"]["representation"] = "lift_conditioned"
        local["model"]["in_channels"] = 3 * int(local["model"]["out_channels"]) + 2
    if args.method == "blrpng_plus_d1e3":
        local["training"]["defect_weight"] = 1.0e-3
    elif args.method == "blrpng_plus_d1e2":
        local["training"]["defect_weight"] = 1.0e-2
    # HR specialists do not use cross-resolution consistency or unlabeled defect.
    if is_hr:
        # Preserve the already-frozen HR audit's transition-level shuffle.
        # General schema-v4 source/target training remains streaming by default.
        local["training"]["stream_v4"] = False
        local["training"]["commutator_weight"] = 0.0
        local["training"]["batch_size"] = int(
            local.get("resources", {}).get("train_batch_256", 2)
        )
        # HR optimizer snapshots are several GiB; keep one end-of-run recovery
        # point so all convergence-audit initializations have identical I/O.
        local["training"]["recovery_interval"] = int(local["training"]["epochs"])
    initialization = config["data"].get(initialization_key) if initialization_key else None
    tag = f"_{args.run_tag}" if args.run_tag else ""
    stem = f"v2_task{args.task:02d}_m{args.budget}_s{args.seed}_{args.method}{tag}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_task") as audit:
            result = train(
                local, seed=args.seed, objective=objective,
                initialization_checkpoint=initialization,
                adapter_initialization_checkpoint=args.adapter_initialization,
                resume_checkpoint=args.resume,
                output_checkpoint=f"artifacts/checkpoints/v2/{stem}.pt",
                train_data_path=train_path,
                validation_data_paths=[validation_path],
            )
            if adapter in {"lora", "nlt", "nlt_mix"}:
                compact = compact_task_checkpoint(
                    result["checkpoint"], base_checkpoint=initialization,
                )
                result["checkpoint"] = str(compact)
            # A successful final task checkpoint is itself the retained recovery
            # point. Optimizer snapshots are transient and would otherwise exceed
            # the frozen 300 GB experiment budget across the matrix.
            checkpoint_stem = Path(result["checkpoint"]).name.removesuffix(".adapter.pt").removesuffix(".pt")
            recovery = Path(result["checkpoint"]).resolve().parents[1] / f"{checkpoint_stem}_recovery.pt"
            recovery.unlink(missing_ok=True)
            audit.update({"task": args.task, "budget": args.budget, "method": args.method, "checkpoint": result["checkpoint"]})
    print(json.dumps({
        "checkpoint": result["checkpoint"],
        "best_validation_metric": result["best_validation_metric"],
    }, indent=2))


if __name__ == "__main__":
    main()
