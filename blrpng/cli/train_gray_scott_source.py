from __future__ import annotations

import argparse
import copy
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock, sha256_file, write_json
from blrpng.training.adapters import compact_full_checkpoint_for_inference
from blrpng.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Gray--Scott source operator.")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    parser.add_argument(
        "--objective", choices=("final_only", "blrpng", "blrpng_safe_annealed"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require_prerequisite_gate("artifacts/manifests/gray_scott_seed42_lr_data_gate.json")
    config = copy.deepcopy(load_config(args.config))
    config["experiment"]["seed"] = args.seed
    config["training"]["adapter"]["strategy"] = "full"
    stem = f"gray_scott_source_s{args.seed}_{args.objective}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_gray_scott_source") as audit:
            result = train(
                config, seed=args.seed, objective=args.objective,
                output_checkpoint=f"artifacts/checkpoints/gray_scott/{stem}.pt",
                train_data_path="artifacts/data/gray_scott_seed42/source_train_128.index.json",
                validation_data_paths=[
                    "artifacts/data/gray_scott_seed42/source_validation_128.index.json"
                ],
            )
            checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            freeze = write_json(
                f"artifacts/manifests/{stem}_freeze.json",
                {
                    "schema": "blrpng_gray_scott_source_freeze_v1",
                    "frozen": True, "seed": args.seed, "objective": args.objective,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "best_validation_metric": result["best_validation_metric"],
                    "target_data_used": False, "target_query_used": False,
                },
            )
            audit.update({
                "objective": args.objective, "checkpoint": str(checkpoint),
                "freeze": str(freeze), "target_query_used": False,
            })
    print(json.dumps({
        "objective": args.objective, "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
        "target_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
