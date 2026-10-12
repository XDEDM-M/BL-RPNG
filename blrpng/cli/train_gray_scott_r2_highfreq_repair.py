from __future__ import annotations

import copy
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import exclusive_artifact_lock, resolve_in_project, sha256_file, write_json
from blrpng.training.adapters import compact_task_checkpoint
from blrpng.training.engine import train


def main() -> None:
    repair = require_prerequisite_gate(
        "artifacts/manifests/gray_scott_r2_highfreq_repair.json"
    )
    source = resolve_in_project(repair["source_checkpoint"], must_exist=True)
    if sha256_file(source) != repair["source_checkpoint_sha256"]:
        raise RuntimeError("High-frequency repair source hash mismatch.")
    config = copy.deepcopy(load_config("configs/gray_scott_r2_development.yaml"))
    config["training"].update(repair["training"] | {
        "lr_drop_gamma": 0.2,
    })
    config["training"].pop("validation", None)
    config["training"]["adapter"].update(repair["adapter"])
    stem = "gray_scott_r2_task00_m4_s42_path_nlt_lora_exploratory"
    with exclusive_artifact_lock(stem):
        with workspace_audit("train_gray_scott_r2_highfreq_repair") as audit:
            result = train(
                config,
                seed=42,
                objective="final_only",
                initialization_checkpoint=str(source),
                output_checkpoint=f"artifacts/checkpoints/gray_scott_r2/{stem}.pt",
                train_data_path=(
                    "artifacts/data/gray_scott_seed42/task00_support_64_m4.index.json"
                ),
                validation_data_paths=[
                    "artifacts/data/gray_scott_r2_seed42/"
                    "task00_validation_long_64.index.json"
                ],
            )
            checkpoint = compact_task_checkpoint(
                result["checkpoint"], base_checkpoint=str(source)
            )
            baseline = json.loads(resolve_in_project(
                "artifacts/manifests/gray_scott_r2_target_selection.json",
                must_exist=True,
            ).read_text(encoding="utf-8"))["task_metrics"]["0"]["path_nlt"]
            passed = float(result["best_validation_metric"]) < float(baseline)
            outcome = write_json(
                "artifacts/manifests/gray_scott_r2_highfreq_repair_outcome.json",
                {
                    "schema": "blrpng_gray_scott_r2_highfreq_repair_outcome_v1",
                    "frozen": True,
                    "passed": passed,
                    "exploratory_after_query": True,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "validation_rel_l2": result["best_validation_metric"],
                    "baseline_path_nlt_validation_rel_l2": baseline,
                    "relative_improvement": 1.0 - float(result["best_validation_metric"]) / float(baseline),
                    "current_query_used_for_training_or_selection": False,
                    "new_query_required_for_confirmation": passed,
                },
            )
            audit.update({"checkpoint": str(checkpoint), "outcome": str(outcome), "passed": passed})
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "best_validation_metric": result["best_validation_metric"],
        "passed": passed,
        "outcome": str(outcome),
    }, indent=2))


if __name__ == "__main__":
    main()
