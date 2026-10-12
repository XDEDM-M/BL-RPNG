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
    parser = argparse.ArgumentParser(description="Train a 256-grid Gray--Scott R2 source model.")
    parser.add_argument("--config", default="configs/gray_scott_r2_development.yaml")
    parser.add_argument("--objective", choices=("final_only", "blrpng_safe_annealed"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path-ratio", type=float)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    gate = require_prerequisite_gate("artifacts/manifests/gray_scott_r2_seed42_data_gate.json")
    if gate.get("query_used") is not False:
        raise PermissionError("R2 source training requires query-untouched data.")
    config = copy.deepcopy(load_config(args.config))
    config["experiment"]["seed"] = args.seed
    config["training"]["adapter"]["strategy"] = "full"
    if args.path_ratio is not None:
        if args.objective != "blrpng_safe_annealed" or not 0.0 < args.path_ratio <= 0.15:
            raise ValueError("A custom path ratio is only valid for safe path in (0,0.15].")
        config["training"]["path_gradient_ratio"] = float(args.path_ratio)
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    stem = f"gray_scott_r2_source_s{args.seed}_{args.objective}{suffix}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_gray_scott_r2_source") as audit:
            result = train(
                config, seed=args.seed, objective=args.objective,
                output_checkpoint=f"artifacts/checkpoints/gray_scott_r2/{stem}.pt",
                train_data_path=(
                    "artifacts/data/gray_scott_r2_seed42/source_train_256.index.json"
                ),
                validation_data_paths=[
                    "artifacts/data/gray_scott_r2_seed42/source_validation_256.index.json"
                ],
            )
            checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            freeze = write_json(f"artifacts/manifests/{stem}_freeze.json", {
                "schema": "blrpng_gray_scott_r2_source_freeze_v1",
                "frozen": True, "seed": args.seed, "objective": args.objective,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "best_validation_metric": result["best_validation_metric"],
                "source_resolution": 256, "input_feature_mode": "gray_scott_reaction",
                "path_gradient_ratio": float(config["training"]["path_gradient_ratio"]),
                "target_data_used": False, "target_query_used": False,
            })
            audit.update({"checkpoint": str(checkpoint), "freeze": str(freeze)})
    print(json.dumps({
        "objective": args.objective, "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
    }, indent=2))


if __name__ == "__main__":
    main()
