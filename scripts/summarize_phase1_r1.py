from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json


def _write_json_durable(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if path.read_text(encoding="utf-8") != encoded:
        raise OSError(f"R1 LR summary failed read-back verification: {path}")


def _method(trial_id: str) -> str:
    if trial_id.startswith("scratch_"):
        return "lifted_scratch"
    if trial_id.startswith("ordinary_"):
        return "ordinary_transfer"
    if trial_id.startswith("path_"):
        return "target_path_ft"
    return "path_source_final_only_ft"


def summarize(config_path: Path, results_root: Path, expected_tasks: list[int]) -> dict[str, Any]:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_trials = {str(row["id"]): _method(str(row["id"])) for row in spec["trials"]}
    experiment = spec["experiment"]
    seed = int(experiment["screen_model_seed"])
    budget = int(experiment["support_budget"])
    support_seed = int(experiment.get("support_seed", 0))
    records = []
    missing = []
    for trial_id, method in expected_trials.items():
        namespace = f"{trial_id}_m{budget}_s{seed}"
        if support_seed:
            namespace += f"_ss{support_seed}"
        result_path = results_root / namespace / "task_results.json"
        if not result_path.exists():
            missing.append(trial_id)
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        tasks = payload.get("tasks", {})
        run_dir = result_path.parent
        saved_config_path = run_dir / "config.json"
        if not saved_config_path.exists():
            raise ValueError(f"{trial_id}: missing frozen config descriptor")
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
        descriptor = payload.get("descriptor", {})
        descriptor_sha = sha256_json(descriptor)
        if (
            payload.get("config_sha256") != descriptor_sha
            or saved_config.get("config_sha256") != descriptor_sha
            or saved_config.get("descriptor") != descriptor
        ):
            raise ValueError(f"{trial_id}: config descriptor/hash mismatch")
        if set(tasks) - {str(task) for task in expected_tasks}:
            raise ValueError(f"{trial_id}: result contains tasks outside this screen")
        if set(tasks) != {str(task) for task in expected_tasks}:
            missing.append(trial_id)
            continue
        descriptor_trial = descriptor.get("trial", {})
        if descriptor_trial.get("id") != trial_id:
            raise ValueError(f"{trial_id}: result descriptor trial ID mismatch")
        spec_sha = sha256_file(config_path)
        if descriptor.get("trial_spec_sha256") != spec_sha:
            raise ValueError(f"{trial_id}: trial spec changed since this result was produced")
        if descriptor.get("model_seed") != seed or descriptor.get("support_budget") != budget:
            raise ValueError(f"{trial_id}: model seed or support budget mismatch")
        if descriptor.get("support_seed") != support_seed:
            raise ValueError(f"{trial_id}: support seed mismatch")
        if descriptor.get("task_ids") != expected_tasks:
            raise ValueError(f"{trial_id}: configured task set mismatch")
        manifest_path = PROJECT_ROOT / str(descriptor.get("data_manifest", ""))
        if (
            not manifest_path.exists()
            or sha256_file(manifest_path) != descriptor.get("data_manifest_sha256")
        ):
            raise ValueError(f"{trial_id}: task manifest hash mismatch")
        source_path = descriptor.get("source_checkpoint")
        if source_path:
            source_checkpoint = PROJECT_ROOT / str(source_path)
            if (
                not source_checkpoint.exists()
                or sha256_file(source_checkpoint) != descriptor.get("source_checkpoint_sha256")
            ):
                raise ValueError(f"{trial_id}: source checkpoint hash mismatch")
        elif descriptor.get("source_checkpoint_sha256") is not None:
            raise ValueError(f"{trial_id}: unexpected source checkpoint hash")
        if descriptor.get("smoke_only"):
            raise ValueError(f"{trial_id}: smoke result entered the ranking namespace")
        values = {}
        for task in expected_tasks:
            result = tasks[str(task)]
            if (
                int(result.get("task", -1)) != task
                or result.get("trial_id") != trial_id
                or int(result.get("model_seed", -1)) != seed
                or int(result.get("support_budget", -1)) != budget
                or int(result.get("support_seed", -1)) != support_seed
                or result.get("config_sha256") != descriptor_sha
                or result.get("descriptor_sha256") != descriptor_sha
                or result.get("source_checkpoint_sha256") != descriptor.get("source_checkpoint_sha256")
            ):
                raise ValueError(f"{trial_id} task {task}: task provenance mismatch")
            for index_field, hash_field in (
                ("support_index", "support_index_sha256"),
                ("validation_index", "validation_index_sha256"),
            ):
                index_path = PROJECT_ROOT / str(result.get(index_field, ""))
                if not index_path.exists() or sha256_file(index_path) != result.get(hash_field):
                    raise ValueError(f"{trial_id} task {task}: {index_field} hash mismatch")
            checkpoint_path = PROJECT_ROOT / str(result.get("checkpoint", ""))
            if (
                not checkpoint_path.exists()
                or sha256_file(checkpoint_path) != result.get("checkpoint_sha256")
            ):
                raise ValueError(f"{trial_id} task {task}: checkpoint hash mismatch")
            if result.get("query_metric") is not None:
                raise ValueError(f"{trial_id} task {task}: query metric is present")
            if "LR" not in str(result.get("selection_data", "")).upper():
                raise ValueError(f"{trial_id} task {task}: selection source is not LR validation")
            values[str(task)] = float(result["best_validation_metric"])
        records.append({
            "trial_id": trial_id,
            "method": method,
            "config_sha256": payload.get("config_sha256"),
            "task_count": len(values),
            "mean_task_lr_validation_rollout_rel_l2": sum(values.values()) / len(values),
            "worst_task_lr_validation_rollout_rel_l2": max(values.values()),
            "task_validation_metrics": values,
            "source_checkpoint_sha256": descriptor.get("source_checkpoint_sha256"),
            "selection_data": descriptor.get("selection_data"),
        })

    rankings = {}
    for method in sorted(set(expected_trials.values())):
        group = sorted(
            (record for record in records if record["method"] == method),
            key=lambda record: (record["mean_task_lr_validation_rollout_rel_l2"], record["worst_task_lr_validation_rollout_rel_l2"], record["trial_id"]),
        )
        rankings[method] = {
            "expected_configurations": sum(value == method for value in expected_trials.values()),
            "completed_configurations": len(group),
            "winner": group[0] if group else None,
            "ranked_trial_ids": [record["trial_id"] for record in group],
        }
    return {
        "schema": "blrpng_phase1_r1_lr_summary_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "results_root": str(results_root.relative_to(PROJECT_ROOT)),
        "expected_tasks": expected_tasks,
        "selection_metric": "unweighted mean of per-configured-task best 64x64 20-step LR-validation rollout relative-L2",
        "query_read": False,
        "complete": not missing and len(records) == len(expected_trials),
        "missing_or_incomplete_trials": missing,
        "rankings_by_method": rankings,
        "trial_results": sorted(records, key=lambda record: (record["method"], record["mean_task_lr_validation_rollout_rel_l2"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank complete R1 target-adaptation screens using LR validation only.")
    parser.add_argument("--config", default="configs/phase1_r1.yaml")
    parser.add_argument("--results-root", default="results/phase1/R1")
    parser.add_argument("--output", default="artifacts/reports/phase1_r1_lr_summary.json")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    results_root = PROJECT_ROOT / args.results_root
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = [int(task) for task in spec["experiment"]["tasks"]]
    report = summarize(config_path, results_root, tasks)
    output = PROJECT_ROOT / args.output
    _write_json_durable(output, report)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "method", "trial_id", "config_sha256", "task_count",
            "mean_task_lr_validation_rollout_rel_l2",
            "worst_task_lr_validation_rollout_rel_l2", "task_validation_metrics",
        ])
        writer.writeheader()
        for record in report["trial_results"]:
            row = dict(record)
            row["task_validation_metrics"] = json.dumps(row["task_validation_metrics"], sort_keys=True)
            writer.writerow({
                field: row.get(field, "")
                for field in writer.fieldnames or []
            })
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "complete": report["complete"],
        "missing_or_incomplete_trials": report["missing_or_incomplete_trials"],
        "rankings_by_method": {
            method: {
                "completed_configurations": value["completed_configurations"],
                "expected_configurations": value["expected_configurations"],
                "winner": value["winner"]["trial_id"] if value["winner"] else None,
                "winner_mean_lr_validation": value["winner"]["mean_task_lr_validation_rollout_rel_l2"] if value["winner"] else None,
            }
            for method, value in report["rankings_by_method"].items()
        },
        "output": str(output),
        "csv": str(csv_path),
    }, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
