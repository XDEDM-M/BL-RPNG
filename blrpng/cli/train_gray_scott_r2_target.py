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


FINAL_SOURCE = (
    "artifacts/checkpoints/gray_scott_r2/"
    "gray_scott_r2_source_s42_final_only.inference.pt"
)
PATH_SOURCE = (
    "artifacts/checkpoints/gray_scott_r2/"
    "gray_scott_r2_source_s42_blrpng_safe_annealed_r005.inference.pt"
)
METHODS = {
    "final_nlt": (FINAL_SOURCE, "final_only"),
    "path_nlt": (PATH_SOURCE, "final_only"),
    "final_nlt_multiscale": (FINAL_SOURCE, "blrpng_v2"),
    "path_nlt_multiscale": (PATH_SOURCE, "blrpng_v2"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one query-locked Gray--Scott R2 adapter.")
    parser.add_argument("--config", default="configs/gray_scott_r2_development.yaml")
    parser.add_argument("--task", type=int, choices=(0, 4, 8, 12), required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    selection = require_prerequisite_gate(
        "artifacts/manifests/gray_scott_r2_source_selection.json"
    )
    if selection.get("target_query_used") is not False:
        raise PermissionError("R2 target development must remain query locked.")
    initialization, objective = METHODS[args.method]
    expected_key = "final_only" if args.method.startswith("final_") else "path_r005"
    expected = selection["source_candidates"][expected_key]
    resolved_initialization = resolve_in_project(initialization, must_exist=True)
    if sha256_file(resolved_initialization) != expected["checkpoint_sha256"]:
        raise RuntimeError("Selected R2 source checkpoint hash mismatch.")

    validation = resolve_in_project(
        f"artifacts/data/gray_scott_r2_seed42/"
        f"task{args.task:02d}_validation_long_64.index.json",
        must_exist=True,
    )
    config = copy.deepcopy(load_config(args.config))
    config["experiment"]["seed"] = args.seed
    config["training"].update({
        "epochs": 1 if args.smoke else 100,
        "minimum_epochs": 1 if args.smoke else 60,
        "early_stopping_patience": 1 if args.smoke else 15,
        "batch_size": 4,
        "lr_drop_epoch": 90,
        "lr_drop_gamma": 0.2,
        "path_gradient_ratio": 0.05,
        "calibration_resolutions": [128, 256],
        "commutator_rollout_steps": 2,
    })
    config["training"]["adapter"]["strategy"] = "nlt"
    suffix = "_smoke" if args.smoke else ""
    stem = f"gray_scott_r2_task{args.task:02d}_m4_s{args.seed}_{args.method}{suffix}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_gray_scott_r2_target") as audit:
            result = train(
                config,
                seed=args.seed,
                objective=objective,
                initialization_checkpoint=str(resolved_initialization),
                output_checkpoint=f"artifacts/checkpoints/gray_scott_r2/{stem}.pt",
                train_data_path=(
                    "artifacts/data/gray_scott_seed42/"
                    f"task{args.task:02d}_support_64_m4.index.json"
                ),
                validation_data_paths=[str(validation)],
            )
            checkpoint = compact_task_checkpoint(
                result["checkpoint"], base_checkpoint=str(resolved_initialization)
            )
            audit.update({
                "task": args.task,
                "method": args.method,
                "objective": objective,
                "checkpoint": str(checkpoint),
                "validation_horizon": 50,
                "target_query_used": False,
                "smoke": args.smoke,
            })
    print(json.dumps({
        "task": args.task,
        "method": args.method,
        "objective": objective,
        "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
        "target_query_used": False,
        "smoke": args.smoke,
    }, indent=2))


if __name__ == "__main__":
    main()
