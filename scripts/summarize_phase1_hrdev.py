from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json
from blrpng.data.shards import read_index


TASK_FAMILIES = {
    0: "high_frequency",
    1: "amplitude_ood",
    2: "matern_gp",
    3: "localized_bump",
    4: "mixed_fourier",
    5: "piecewise_smooth",
}


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
        raise OSError(f"HR-development promotion summary failed read-back verification: {path}")
FINALIST_MODEL_SEEDS = (0, 1, 42)
FINALIST_SUPPORT_SEED = 271829
FINALIST_SUPPORT_BUDGET = 4


def summarize(
    trial_ids: list[str],
    seeds: list[int],
    support_seed: int,
    support_budget: int,
    reports_root: Path,
    seed_trial_aliases: dict[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    if len(set(trial_ids)) != len(trial_ids) or not trial_ids:
        raise ValueError("Provide one or more unique candidate trial IDs.")
    if tuple(seeds) != FINALIST_MODEL_SEEDS:
        raise ValueError(
            f"Use the frozen finalist model seeds {list(FINALIST_MODEL_SEEDS)} in this order; "
            "do not select seeds after observing their results."
        )
    if support_seed != FINALIST_SUPPORT_SEED or support_budget != FINALIST_SUPPORT_BUDGET:
        raise ValueError("Finalist promotion requires the frozen m=4, support-seed 271829 development draw.")
    seed_trial_aliases = seed_trial_aliases or {}
    manifest_path = PROJECT_ROOT / "artifacts/manifests/phase1_development/hrdev_20261012/tasks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_json({key: value for key, value in manifest.items() if key != "sha256"})
    if manifest.get("sha256") != manifest_hash:
        raise ValueError("Frozen HR-development task manifest seal is invalid.")
    task_rows = {int(row["target_task_id"]): row for row in manifest["tasks"]}

    candidates = []
    for trial_id in trial_ids:
        cells: dict[tuple[int, int], dict] = {}
        descriptors: dict[int, set[str]] = defaultdict(set)
        for seed in seeds:
            effective_trial_id = seed_trial_aliases.get((trial_id, seed), trial_id)
            for task, family in TASK_FAMILIES.items():
                path = reports_root / f"{effective_trial_id}_task{task:02d}_s{seed}_ss{support_seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing finalist HR-development cell: {path}")
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("model_trial_id") != effective_trial_id:
                    raise ValueError(f"{path}: candidate trial ID mismatch")
                if int(report.get("task", -1)) != task or report.get("family") != family:
                    raise ValueError(f"{path}: task/family does not match frozen mapping")
                if int(report.get("model_seed", -1)) != seed:
                    raise ValueError(f"{path}: model seed mismatch")
                if int(report.get("support_seed", -1)) != support_seed:
                    raise ValueError(f"{path}: support draw mismatch")
                if int(report.get("support_budget", -1)) != support_budget:
                    raise ValueError(f"{path}: support budget mismatch")
                if report.get("smoke_only"):
                    raise ValueError(f"{path}: smoke output cannot promote a candidate")
                if report.get("query_read") is not False:
                    raise ValueError(f"{path}: query access must be explicitly false")
                if report.get("training_or_within_task_checkpoint_selection_used_hr") is not False:
                    raise ValueError(f"{path}: HR data influenced training or within-task selection")
                if report.get("hr_validation_is_development_only") is not True:
                    raise ValueError(f"{path}: report is not marked HR development only")
                if int(report.get("hr_resolution", -1)) != 256 or int(report.get("rollout_steps", -1)) != 20:
                    raise ValueError(f"{path}: HR development protocol mismatch")
                if int(report.get("hr_validation_rollouts", -1)) != 8:
                    raise ValueError(f"{path}: expected eight HR validation rollouts")
                if report.get("task_manifest_sha256") != manifest_hash:
                    raise ValueError(f"{path}: HR development manifest hash mismatch")
                if int(report.get("boundary_id", -1)) != int(task_rows[task]["boundary_id"]):
                    raise ValueError(f"{path}: frozen boundary ID mismatch")
                checkpoint = PROJECT_ROOT / str(report.get("checkpoint", ""))
                if not checkpoint.exists() or sha256_file(checkpoint) != report.get("checkpoint_sha256"):
                    raise ValueError(f"{path}: checkpoint hash mismatch")
                result_path = PROJECT_ROOT / str(report.get("task_result", ""))
                if not result_path.exists() or sha256_file(result_path) != report.get("task_result_sha256"):
                    raise ValueError(f"{path}: LR-selected task-result hash mismatch")
                task_result = json.loads(result_path.read_text(encoding="utf-8"))
                task_config_path = result_path.parent / "config.json"
                if not task_config_path.exists():
                    raise ValueError(f"{path}: missing task-run config descriptor")
                task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
                descriptor = task_config.get("descriptor", {})
                descriptor_sha = sha256_json(descriptor)
                if (
                    int(task_result.get("task", -1)) != task
                    or task_result.get("trial_id") != effective_trial_id
                    or int(task_result.get("model_seed", -1)) != seed
                    or int(task_result.get("support_seed", -1)) != support_seed
                    or int(task_result.get("support_budget", -1)) != support_budget
                    or task_result.get("config_sha256") != report.get("model_config_sha256")
                    or task_result.get("descriptor_sha256") != descriptor_sha
                    or descriptor_sha != report.get("model_config_sha256")
                    or task_config.get("config_sha256") != descriptor_sha
                    or task_result.get("query_metric") is not None
                    or "LR" not in str(task_result.get("selection_data", "")).upper()
                    or task_result.get("checkpoint_sha256") != report.get("checkpoint_sha256")
                    or float(task_result.get("best_validation_metric", float("nan")))
                    != float(report.get("lr_validation_metric", float("inf")))
                ):
                    raise ValueError(f"{path}: HR report does not match its LR-selected task result")
                for index_field, hash_field, expected_split, expected_resolution, expected_fidelity in (
                    ("support_index", "support_index_sha256", "support", 64, 0),
                    ("validation_index", "validation_index_sha256", "validation_lr", 64, 0),
                ):
                    source_index_path = PROJECT_ROOT / str(task_result.get(index_field, ""))
                    if not source_index_path.exists():
                        raise ValueError(f"{path}: missing {index_field}")
                    source_index = read_index(source_index_path, verify_hashes=True)
                    # Task results store the index's canonical content seal, not
                    # the bytewise SHA-256 of its JSON serialization.
                    if source_index.get("sha256") != task_result.get(hash_field):
                        raise ValueError(f"{path}: {index_field} content seal mismatch")
                    source_meta = source_index.get("metadata", {})
                    if (
                        source_meta.get("split") != expected_split
                        or int(source_meta.get("resolution", -1)) != expected_resolution
                        or int(source_meta.get("fidelity", -1)) != expected_fidelity
                        or int(source_meta.get("steps", -1)) != 20
                        or int(source_meta.get("target_task_id", -1)) != task
                        or (
                            expected_split == "support"
                            and int(source_meta.get("support_seed") or 0) != support_seed
                        )
                        or source_meta.get("task_manifest_sha256") != manifest_hash
                    ):
                        raise ValueError(f"{path}: {index_field} provenance mismatch")
                    if expected_split == "support" and len(source_index.get("shards", [])) != support_budget:
                        raise ValueError(f"{path}: support index does not match the requested budget")
                hr_index_path = PROJECT_ROOT / str(report.get("hr_validation_index", ""))
                if not hr_index_path.exists():
                    raise ValueError(f"{path}: missing HR validation index")
                hr_index = read_index(hr_index_path, verify_hashes=True)
                if (
                    hr_index.get("sha256") != report.get("hr_validation_index_sha256")
                    or hr_index.get("metadata", {}).get("split") != "validation_hr"
                    or int(hr_index.get("metadata", {}).get("resolution", -1)) != 256
                    or int(hr_index.get("metadata", {}).get("steps", -1)) != 20
                    or int(hr_index.get("metadata", {}).get("fidelity", -1)) != 1
                    or len([row for row in hr_index.get("shards", []) if int(row["target_task_id"]) == task]) != 8
                ):
                    raise ValueError(f"{path}: HR index role, fidelity, task, or horizon mismatch")
                metric = float(report["metrics"]["rollout_rel_l2"])
                if not (metric >= 0.0 and metric < float("inf")):
                    raise ValueError(f"{path}: invalid HR validation metric")
                descriptors[seed].add(str(report.get("model_config_sha256")))
                cells[(seed, task)] = {
                    "seed": seed,
                    "task": task,
                    "family": family,
                    "effective_trial_id": effective_trial_id,
                    "rollout_rel_l2": metric,
                    "report": str(path.relative_to(PROJECT_ROOT)),
                    "report_sha256": sha256_file(path),
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "lr_validation_rollout_rel_l2": float(report["lr_validation_metric"]),
                }

        for seed in seeds:
            if len(descriptors[seed]) != 1:
                raise ValueError(f"{trial_id}: config hash differs across tasks for seed {seed}")

        by_task = {}
        seed_means = {}
        for task, family in TASK_FAMILIES.items():
            values = [cells[(seed, task)]["rollout_rel_l2"] for seed in seeds]
            by_task[family] = sum(values) / len(values)
        for seed in seeds:
            values = [cells[(seed, task)]["rollout_rel_l2"] for task in TASK_FAMILIES]
            seed_means[str(seed)] = sum(values) / len(values)
        all_values = [cell["rollout_rel_l2"] for cell in cells.values()]
        candidates.append({
            "trial_id": trial_id,
            "model_seeds": seeds,
            "support_seed": support_seed,
            "support_budget": support_budget,
            "cell_count": len(cells),
            "mean_hr_dev_rollout_rel_l2": sum(all_values) / len(all_values),
            "worst_task_mean_hr_dev_rollout_rel_l2": max(by_task.values()),
            "mean_by_boundary_family": by_task,
            "mean_by_model_seed": seed_means,
            "model_config_sha256_by_seed": {
                str(seed): next(iter(descriptors[seed])) for seed in seeds
            },
            "cells": [cells[(seed, task)] for seed in seeds for task in TASK_FAMILIES],
        })
    candidates.sort(key=lambda row: (
        row["mean_hr_dev_rollout_rel_l2"],
        row["worst_task_mean_hr_dev_rollout_rel_l2"],
        row["trial_id"],
    ))
    return {
        "schema": "blrpng_phase1_hrdev_promotion_summary_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection": "unweighted mean of all six boundary-family HR-development validation rollout relative-L2 values across each candidate's fixed model seeds; worst-family mean is a tie-break/diagnostic",
        "model_seeds": seeds,
        "support_seed": support_seed,
        "support_budget": support_budget,
        "reports_root": str(reports_root.relative_to(PROJECT_ROOT)),
        "query_read": False,
        "within_task_checkpoint_selection_used_hr": False,
        "candidate_count": len(candidates),
        "promoted_candidate": candidates[0]["trial_id"],
        "ranked_candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate complete all-family, multi-seed HR development finalist evaluations.")
    parser.add_argument("--trial-id", action="append", required=True, help="Finalist configuration ID; repeat for each candidate.")
    parser.add_argument(
        "--seed-trial-alias", action="append", default=[],
        help="Map one candidate/seed to an immutable retry ID: CANDIDATE:SEED=ACTUAL_ID.",
    )
    parser.add_argument("--seed", action="append", type=int, required=True, help="Model seed; at least three unique seeds are required.")
    parser.add_argument("--support-seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--reports-root", default="artifacts/reports/phase1_hrdev_evaluations")
    parser.add_argument("--output", default="artifacts/reports/phase1_hrdev_promotion.json")
    args = parser.parse_args()
    seed_trial_aliases: dict[tuple[str, int], str] = {}
    for value in args.seed_trial_alias:
        candidate_seed, separator, effective_id = value.partition("=")
        candidate, separator2, seed_value = candidate_seed.rpartition(":")
        if not separator or not separator2 or not candidate or not effective_id:
            raise ValueError(f"Invalid --seed-trial-alias value: {value!r}")
        key = (candidate, int(seed_value))
        if candidate not in args.trial_id or key[1] not in args.seed:
            raise ValueError(f"Seed trial alias is outside the requested matrix: {value!r}")
        if key in seed_trial_aliases:
            raise ValueError(f"Duplicate seed trial alias for {candidate!r}, seed {key[1]}")
        seed_trial_aliases[key] = effective_id
    output = PROJECT_ROOT / args.output
    report = summarize(
        trial_ids=args.trial_id,
        seeds=args.seed,
        support_seed=args.support_seed,
        support_budget=args.budget,
        reports_root=PROJECT_ROOT / args.reports_root,
        seed_trial_aliases=seed_trial_aliases,
    )
    _write_json_durable(output, report)
    print(json.dumps({
        "promoted_candidate": report["promoted_candidate"],
        "ranked": [
            {
                "trial_id": row["trial_id"],
                "mean_hr_dev_rollout_rel_l2": row["mean_hr_dev_rollout_rel_l2"],
                "worst_task_mean_hr_dev_rollout_rel_l2": row["worst_task_mean_hr_dev_rollout_rel_l2"],
            }
            for row in report["ranked_candidates"]
        ],
        "output": str(output),
        "query_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
