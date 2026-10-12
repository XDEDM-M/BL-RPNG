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
    parser = argparse.ArgumentParser(description="Train one frozen Poisson source objective.")
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument("--objective", choices=("final_only", "blrpng"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    require_prerequisite_gate("artifacts/manifests/poisson_negative_control_lr_data_gate.json")
    local = copy.deepcopy(config)
    local["experiment"]["seed"] = int(args.seed)
    local["training"]["adapter"]["strategy"] = "full"
    stem = f"poisson_source_s{args.seed}_{args.objective}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_poisson_source") as audit:
            result = train(
                local,
                seed=args.seed,
                objective=args.objective,
                output_checkpoint=f"artifacts/checkpoints/poisson/{stem}.pt",
                train_data_path=(
                    "artifacts/data/poisson_negative_control/source_train_128.index.json"
                ),
                validation_data_paths=[
                    "artifacts/data/poisson_negative_control/source_validation_128.index.json"
                ],
            )
            compact = compact_full_checkpoint_for_inference(result["checkpoint"])
            freeze = write_json(
                f"artifacts/manifests/{stem}_freeze.json",
                {
                    "schema": "blrpng_poisson_source_checkpoint_freeze_v1",
                    "frozen": True,
                    "seed": args.seed,
                    "objective": args.objective,
                    "checkpoint": str(compact),
                    "checkpoint_sha256": sha256_file(compact),
                    "best_validation_metric": result["best_validation_metric"],
                    "target_lr_used": False,
                    "target_query_used": False,
                },
            )
            audit.update({
                "objective": args.objective, "seed": args.seed,
                "checkpoint": str(compact), "freeze": str(freeze),
                "target_query_used": False,
            })
    print(json.dumps({
        "objective": args.objective, "checkpoint": str(compact),
        "checkpoint_sha256": sha256_file(compact),
        "best_validation_metric": result["best_validation_metric"],
        "target_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
