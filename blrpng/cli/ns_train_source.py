"""Train one matched SfePy-NS source-pretraining objective after solver gating."""
from __future__ import annotations

import argparse
import copy
import json
import math

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import exclusive_artifact_lock, sha256_file, write_json
from blrpng.training.adapters import compact_full_checkpoint_for_inference
from blrpng.training.engine import train


OBJECTIVES = {
    "ordinary": ("final_only", "newton_correction"),
    "endpoint": ("blrpng_safe", "endpoint_increment"),
    "newton": ("blrpng_safe", "newton_correction"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navier_stokes_sfepy_seed42.yaml")
    parser.add_argument("--objective", choices=tuple(OBJECTIVES), required=True)
    parser.add_argument("--architecture", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    require_prerequisite_gate("artifacts/manifests/navier_stokes_sfepy_solver_gate.json")
    local = copy.deepcopy(config)
    local["experiment"]["seed"] = int(args.seed)
    local["data"]["representation"] = "lifted"
    local["model"].update(local["model_search"]["candidates"][args.architecture])
    local["training"]["adapter"]["strategy"] = "full"
    internal_objective, auxiliary = OBJECTIVES[args.objective]
    local["training"]["auxiliary_target"] = auxiliary
    local["training"]["lr_scheduler"] = "plateau"
    local["training"]["minimum_epochs"] = 1
    local["training"]["recovery_interval"] = 1
    split = "development" if args.development else "source"
    train_index = f"artifacts/data/navier_stokes_sfepy/{split}_train_{128 if args.development else 256}.index.json"
    validation_index = f"artifacts/data/navier_stokes_sfepy/{split}_validation_{128 if args.development else 256}.index.json"
    index = read_index(train_index)
    transitions = sum(int(row["steps"]) for row in index["shards"])
    batches = math.ceil(transitions / int(local["training"]["batch_size"]))
    updates_per_epoch = math.ceil(
        batches / int(local["training"]["gradient_accumulation_steps"])
    )
    max_steps = int(local["training"]["max_optimizer_steps"])
    local["training"]["epochs"] = math.ceil(max_steps / max(updates_per_epoch, 1))
    stem = f"ns_sfepy_{split}_{args.architecture}_{args.objective}_s{args.seed}"
    with exclusive_artifact_lock(stem):
        with workspace_audit("ns_train_source") as audit:
            result = train(
                local,
                seed=args.seed,
                objective=internal_objective,
                resume_checkpoint=args.resume,
                output_checkpoint=f"artifacts/checkpoints/navier_stokes_sfepy/{stem}.pt",
                train_data_path=train_index,
                validation_data_paths=[validation_index],
            )
            compact = compact_full_checkpoint_for_inference(result["checkpoint"])
            freeze = write_json(
                f"artifacts/manifests/{stem}_freeze.json",
                {
                    "schema": "blrpng_ns_sfepy_source_checkpoint_freeze_v1",
                    "frozen": True,
                    "objective": args.objective,
                    "auxiliary_target": auxiliary,
                    "architecture": args.architecture,
                    "seed": args.seed,
                    "checkpoint": str(compact),
                    "checkpoint_sha256": sha256_file(compact),
                    "best_validation_metric": result["best_validation_metric"],
                    "optimizer_steps": result.get("optimizer_steps"),
                    "query_used": False,
                },
            )
            audit.update({"checkpoint": str(compact), "freeze": str(freeze), "query_used": False})
    print(json.dumps({"checkpoint": str(compact), "freeze": str(freeze)}, indent=2))


if __name__ == "__main__":
    main()
