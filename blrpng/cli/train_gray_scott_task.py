from __future__ import annotations

import argparse
import copy
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock, resolve_in_project, sha256_file
from blrpng.training.adapters import compact_task_checkpoint
from blrpng.training.engine import train


PATH_SOURCE = "artifacts/checkpoints/gray_scott/gray_scott_source_s42_blrpng.inference.pt"
SAFE_PATH_SOURCE = (
    "artifacts/checkpoints/gray_scott/"
    "gray_scott_source_s42_blrpng_safe_annealed.inference.pt"
)
FINAL_SOURCE = "artifacts/checkpoints/gray_scott/gray_scott_source_s42_final_only.inference.pt"
METHODS = {
    "ordinary_nlt": (FINAL_SOURCE, "final_only"),
    "path_nlt": (PATH_SOURCE, "final_only"),
    "path_nlt_gated": (PATH_SOURCE, "blrpng_v2"),
    "path_nlt_commutator": (PATH_SOURCE, "commutator_only"),
    "path_nlt_plus": (PATH_SOURCE, "blrpng_plus"),
    "safe_path_nlt": (SAFE_PATH_SOURCE, "final_only"),
    "safe_path_nlt_gated": (SAFE_PATH_SOURCE, "blrpng_v2"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt one Gray--Scott target task with NLT.")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--convergence-repair", action="store_true",
        help="Use the frozen 100-epoch LR-only convergence protocol.",
    )
    args = parser.parse_args()
    if not 0 <= args.task < 16:
        raise ValueError("Gray--Scott task must be in [0,16).")
    require_prerequisite_gate("artifacts/manifests/gray_scott_seed42_method_freeze.json")
    if args.method.startswith("safe_path_"):
        source_freeze_path = resolve_in_project(
            "artifacts/manifests/"
            "gray_scott_source_s42_blrpng_safe_annealed_freeze.json",
            must_exist=True,
        )
        source_freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
        if not source_freeze.get("frozen") or source_freeze.get("target_query_used") is not False:
            raise PermissionError("Safe source checkpoint is not query-locked and frozen.")
        if sha256_file(SAFE_PATH_SOURCE) != source_freeze.get("checkpoint_sha256"):
            raise RuntimeError("Safe source checkpoint hash does not match its freeze manifest.")
    config = copy.deepcopy(load_config(args.config))
    initialization, objective = METHODS[args.method]
    config["experiment"]["seed"] = args.seed
    config["training"]["adapter"]["strategy"] = "nlt"
    suffix = ""
    if args.convergence_repair:
        repair_path = resolve_in_project(
            "artifacts/manifests/gray_scott_seed42_convergence_repair.json",
            must_exist=True,
        )
        repair = json.loads(repair_path.read_text(encoding="utf-8"))
        if not repair.get("frozen") or not repair.get("passed"):
            raise PermissionError("Gray--Scott convergence repair is not frozen.")
        frozen_training = repair["training"]
        for key in (
            "epochs", "minimum_epochs", "early_stopping_patience",
            "lr_drop_epoch", "lr_drop_gamma", "batch_size",
        ):
            config["training"][key] = frozen_training[key]
        suffix = "_e100"
    stem = (
        f"gray_scott_task{args.task:02d}_m{args.budget}_s{args.seed}_"
        f"{args.method}{suffix}"
    )
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_gray_scott_task") as audit:
            result = train(
                config, seed=args.seed, objective=objective,
                initialization_checkpoint=initialization,
                output_checkpoint=f"artifacts/checkpoints/gray_scott/{stem}.pt",
                train_data_path=(
                    f"artifacts/data/gray_scott_seed42/"
                    f"task{args.task:02d}_support_64_m{args.budget}.index.json"
                ),
                validation_data_paths=[
                    f"artifacts/data/gray_scott_seed42/"
                    f"task{args.task:02d}_validation_64_all.index.json"
                ],
            )
            checkpoint = compact_task_checkpoint(
                result["checkpoint"], base_checkpoint=initialization
            )
            audit.update({
                "task": args.task, "budget": args.budget, "method": args.method,
                "objective": objective, "checkpoint": str(checkpoint),
                "convergence_repair": args.convergence_repair,
                "target_query_used": False,
            })
    print(json.dumps({
        "task": args.task, "method": args.method, "objective": objective,
        "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
        "convergence_repair": args.convergence_repair,
        "target_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
