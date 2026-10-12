from __future__ import annotations

import argparse
import copy
import json

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock, resolve_in_project
from blrpng.training.adapters import compact_full_checkpoint_for_inference
from blrpng.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a converged lifted Gray--Scott scratch baseline.")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--continuation-e300", action="store_true",
        help="Continue the frozen e100 scratch model with the saturation protocol.",
    )
    args = parser.parse_args()
    if not 0 <= args.task < 16:
        raise ValueError("Gray--Scott task must be in [0,16).")
    repair_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_convergence_repair.json",
        must_exist=True,
    )
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    if not repair.get("frozen") or not repair.get("passed"):
        raise PermissionError("Convergence repair must be frozen before scratch training.")
    config = copy.deepcopy(load_config(args.config))
    config["experiment"]["seed"] = args.seed
    config["training"]["adapter"]["strategy"] = "full"
    for key in (
        "epochs", "minimum_epochs", "early_stopping_patience",
        "lr_drop_epoch", "lr_drop_gamma", "batch_size",
    ):
        config["training"][key] = repair["training"][key]
    initialization = None
    suffix = "e100"
    if args.continuation_e300:
        supersession = json.loads(resolve_in_project(
            "artifacts/manifests/gray_scott_scratch_saturation_supersession.json",
            must_exist=True,
        ).read_text(encoding="utf-8"))
        if supersession.get("query_opening_authorized") is not False:
            raise PermissionError("Scratch continuation requires a query-locked supersession.")
        initialization = (
            f"artifacts/checkpoints/gray_scott/gray_scott_task{args.task:02d}_"
            f"m{args.budget}_s{args.seed}_lifted_scratch_e100.inference.pt"
        )
        config["training"].update({
            "epochs": 200,
            "minimum_epochs": 60,
            "early_stopping_patience": 30,
            "lr": 2e-4,
            "lr_drop_epoch": 180,
            "lr_drop_gamma": 0.2,
        })
        suffix = "e300"
    stem = (
        f"gray_scott_task{args.task:02d}_m{args.budget}_s{args.seed}_"
        f"lifted_scratch_{suffix}"
    )
    with exclusive_artifact_lock(stem):
        with workspace_audit(f"train_{stem}") as audit:
            result = train(
                config,
                seed=args.seed,
                objective="final_only",
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
            checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            audit.update({
                "task": args.task, "checkpoint": str(checkpoint),
                "continuation_e300": args.continuation_e300,
                "target_query_used": False,
            })
    print(json.dumps({
        "task": args.task,
        "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
        "continuation_e300": args.continuation_e300,
        "target_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
