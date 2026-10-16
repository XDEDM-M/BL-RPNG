#!/usr/bin/env python3
"""Recompute the R5 LR=3e-5 seed1 development replication and paired context."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blrpng.paths import PROJECT_ROOT, sha256_file


TASKS = range(6)
SUPPORT_SEED = 271829
LOW_SEED0 = "ordinary_full_comm256_lr3e-5-lr3e5screen"
LOW_SEED1_PREFIX = "ordinary_full_comm256_lr3e-5-lr3e5seed1rep-task"


def unique_report(pattern: str) -> Path:
    paths = sorted((PROJECT_ROOT / "artifacts/reports/phase1_hrdev_evaluations").glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one HR-dev report for {pattern}, found {len(paths)}")
    return paths[0]


def hr_metric(pattern: str, task_result: dict | None = None) -> dict:
    path = unique_report(pattern)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("query_read") is not False:
        raise ValueError(f"Query discipline missing: {path}")
    if report.get("training_or_within_task_checkpoint_selection_used_hr") is not False:
        raise ValueError(f"HR-dev was used for selection: {path}")
    if task_result is not None:
        if report.get("checkpoint_sha256") != task_result.get("checkpoint_sha256"):
            raise ValueError(f"Selected checkpoint mismatch: {path}")
        if report.get("task_result_sha256") != sha256_file(
            PROJECT_ROOT / result_path(task_result)
        ):
            raise ValueError(f"Task report hash mismatch: {path}")
        if report.get("lr_validation_metric") != task_result.get(
            "best_validation_metric"
        ):
            raise ValueError(f"LR selection metric mismatch: {path}")
    return {
        "mean": float(report["metrics"]["rollout_rel_l2"]),
        "report": str(path.relative_to(PROJECT_ROOT)),
        "report_sha256": sha256_file(path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "task_result_sha256": report["task_result_sha256"],
    }


def result_path(result: dict) -> str:
    # Task results are stored beneath their immutable run namespace. The report
    # stores a project-relative source path in its provenance record.
    for candidate in (PROJECT_ROOT / "results/phase1/R5").glob("*/task*.json"):
        if candidate.name == f"task{int(result['task']):02d}.json":
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                data.get("trial_id") == result.get("trial_id")
                and data.get("config_sha256") == result.get("config_sha256")
            ):
                return str(candidate.relative_to(PROJECT_ROOT))
    raise FileNotFoundError(f"Cannot locate result for {result.get('trial_id')}")


def load_baseline(task: int, trial: str, seed: int) -> tuple[dict, str]:
    suffix = f"_task{task:02d}_s{seed}_ss{SUPPORT_SEED}.json"
    candidates = sorted(
        (PROJECT_ROOT / "artifacts/reports/phase1_hrdev_evaluations").glob(
            f"{trial}*{suffix}"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {trial} HR-dev report for task={task}, seed={seed}; "
            f"found {[p.name for p in candidates]}"
        )
    report_path = candidates[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("query_read") is not False:
        raise ValueError(f"Query discipline missing: {report_path}")
    return report, str(report_path.relative_to(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/reports/phase1_r5_lr3e5_seed1_replication_summary_20261016.json",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml"
    ledger_path = PROJECT_ROOT / "docs/EXPERIMENTS.csv"
    ledger_rows = list(csv.DictReader(ledger_path.open(newline="", encoding="utf-8")))

    low_seed1 = []
    low_seed1_hr = []
    low_seed0_hr = []
    ordinary_1e4 = {0: [], 1: []}
    pathsource = {0: [], 1: []}
    task_results = []
    train_costs = []
    hr_costs = []
    task_pairs = []

    for task in TASKS:
        namespace = (
            f"ordinary_full_comm256_lr3e-5-lr3e5seed1rep-task{task:02d}_m4_s1_"
            f"ss{SUPPORT_SEED}"
        )
        result_file = PROJECT_ROOT / f"results/phase1/R5/{namespace}/task{task:02d}.json"
        run_config_file = PROJECT_ROOT / f"results/phase1/R5/{namespace}/config.json"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        run_config = json.loads(run_config_file.read_text(encoding="utf-8"))
        if result.get("epochs_run") != 100:
            raise ValueError(f"Unexpected epoch count in {result_file}")
        if result.get("query_metric") is not None:
            raise ValueError(f"Query metric present in {result_file}")
        if result.get("config_sha256") != run_config.get("config_sha256"):
            raise ValueError(f"Config fingerprint mismatch in {result_file}")
        hr1 = hr_metric(
            f"{LOW_SEED1_PREFIX}{task:02d}_task{task:02d}_s1_ss{SUPPORT_SEED}.json",
            result,
        )
        hr0 = hr_metric(
            f"{LOW_SEED0}_task{task:02d}_s0_ss{SUPPORT_SEED}.json"
        )

        ordinary_reports = {}
        path_reports = {}
        for seed in (0, 1):
            ord_report, ord_path = load_baseline(
                task, "pilot_ordinary_full_comm256", seed
            )
            path_report, path_path = load_baseline(
                task, "pilot_pathsource_comm256", seed
            )
            ordinary_reports[seed] = (ord_report, ord_path)
            path_reports[seed] = (path_report, path_path)
            ordinary_1e4[seed].append(
                {
                    "mean": float(ord_report["metrics"]["rollout_rel_l2"]),
                    "report": ord_path,
                }
            )
            pathsource[seed].append(
                {
                    "mean": float(path_report["metrics"]["rollout_rel_l2"]),
                    "report": path_path,
                }
            )

        done = [
            row
            for row in ledger_rows
            if row["status"] == "DONE"
            and row["raw_output_path"] == str(result_file)
            and row["round"] == "R5"
        ]
        if len(done) != 1:
            raise ValueError(f"Expected one matching training ledger row: {result_file}")
        train_cost = float(done[0]["elapsed_gpu_hours"])
        if abs(train_cost - float(result["gpu_hours"])) > 1e-12:
            raise ValueError(f"Training cost mismatch in {result_file}")
        matching_hr = [
            row
            for row in ledger_rows
            if row["round"] == "HRDEV"
            and row["status"] == "DONE"
            and row["raw_output_path"]
            == str(PROJECT_ROOT / hr1["report"])
        ]
        if len(matching_hr) != 1:
            raise ValueError(f"Expected one matching HR-dev ledger row: {hr1['report']}")
        hr_cost = float(matching_hr[0]["elapsed_gpu_hours"])
        train_costs.append(train_cost)
        hr_costs.append(hr_cost)
        low_seed1.append(float(result["best_validation_metric"]))
        low_seed1_hr.append(hr1["mean"])
        low_seed0_hr.append(hr0["mean"])
        task_results.append(
            {
                "task": task,
                "family": result["family"],
                "lr_validation_rel_l2": float(result["best_validation_metric"]),
                "hr_dev_seed1_rel_l2": hr1["mean"],
                "hr_dev_seed0_rel_l2": hr0["mean"],
                "hr_dev_ordinary_seed0_rel_l2": ordinary_reports[0][0]["metrics"][
                    "rollout_rel_l2"
                ],
                "hr_dev_ordinary_seed1_rel_l2": ordinary_reports[1][0]["metrics"][
                    "rollout_rel_l2"
                ],
                "hr_dev_pathsource_seed0_rel_l2": path_reports[0][0]["metrics"][
                    "rollout_rel_l2"
                ],
                "hr_dev_pathsource_seed1_rel_l2": path_reports[1][0]["metrics"][
                    "rollout_rel_l2"
                ],
                "training_gpu_hours": train_cost,
                "hr_dev_gpu_hours": hr_cost,
                "task_result": str(result_file.relative_to(PROJECT_ROOT)),
                "task_result_sha256": sha256_file(result_file),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "checkpoint": result["checkpoint"],
                "config_sha256": result["config_sha256"],
                "source_checkpoint_sha256": result["source_checkpoint_sha256"],
                "support_index_sha256": result["support_index_sha256"],
                "validation_index_sha256": result["validation_index_sha256"],
                "hr_dev_report": hr1["report"],
                "hr_dev_report_sha256": hr1["report_sha256"],
            }
        )
        task_pairs.append(
            {
                "task": task,
                "low_lr_minus_ordinary_seed0": hr0["mean"]
                - float(ordinary_reports[0][0]["metrics"]["rollout_rel_l2"]),
                "low_lr_minus_ordinary_seed1": hr1["mean"]
                - float(ordinary_reports[1][0]["metrics"]["rollout_rel_l2"]),
                "low_lr_minus_pathsource_seed0": hr0["mean"]
                - float(path_reports[0][0]["metrics"]["rollout_rel_l2"]),
                "low_lr_minus_pathsource_seed1": hr1["mean"]
                - float(path_reports[1][0]["metrics"]["rollout_rel_l2"]),
            }
        )

    def flat_mean(rows: list[dict]) -> float:
        return statistics.mean(row["mean"] for row in rows)

    all_ordinary = ordinary_1e4[0] + ordinary_1e4[1]
    all_pathsource = pathsource[0] + pathsource[1]
    two_seed_low_hr = statistics.mean(low_seed0_hr + low_seed1_hr)
    two_seed_ordinary_hr = flat_mean(all_ordinary)
    two_seed_path_hr = flat_mean(all_pathsource)
    report = {
        "schema": "phase1_r5_lr3e5_seed1_replication_v1",
        "development_only": True,
        "confirmation_evidence": False,
        "query_read": False,
        "model_seeds": [0, 1],
        "support_seed": SUPPORT_SEED,
        "m": 4,
        "tasks": list(TASKS),
        "selected_by": "64x64 20-step LR-dev only; HR-dev run after all six LR fits completed",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "training_cost_gpu_hours_seed1": sum(train_costs),
        "hr_dev_cost_gpu_hours_seed1": sum(hr_costs),
        "combined_incremental_gpu_hours_seed1": sum(train_costs) + sum(hr_costs),
        "lr_dev_mean_seed1": statistics.mean(low_seed1),
        "hr_dev_mean_seed0_lr3e5": statistics.mean(low_seed0_hr),
        "hr_dev_mean_seed1_lr3e5": statistics.mean(low_seed1_hr),
        "hr_dev_mean_two_seed_lr3e5": two_seed_low_hr,
        "ordinary_full_comm256_lr1e4_hr_dev_mean_seed0": flat_mean(ordinary_1e4[0]),
        "ordinary_full_comm256_lr1e4_hr_dev_mean_seed1": flat_mean(ordinary_1e4[1]),
        "ordinary_full_comm256_lr1e4_hr_dev_mean_two_seed": two_seed_ordinary_hr,
        "pathsource_comm256_hr_dev_mean_seed0": flat_mean(pathsource[0]),
        "pathsource_comm256_hr_dev_mean_seed1": flat_mean(pathsource[1]),
        "pathsource_comm256_hr_dev_mean_two_seed": two_seed_path_hr,
        "two_seed_lr3e5_relative_reduction_vs_ordinary_lr1e4": 1.0
        - two_seed_low_hr / two_seed_ordinary_hr,
        "two_seed_lr3e5_relative_reduction_vs_pathsource_comm256": 1.0
        - two_seed_low_hr / two_seed_path_hr,
        "paired_task_differences": task_pairs,
        "tasks_detail": task_results,
        "limitations": [
            "Only two model seeds and one support draw; descriptive development evidence, not confirmation.",
            "The two-seed ordinary LR=1e-4 and pathsource comparisons are matched subsets, not their full three-seed Stage-2 aggregates.",
            "The Stage-2 pathsource arm uses a different adaptation strategy; this is a performance comparison, not a mechanism isolation.",
        ],
    }
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "output", "hr_dev_mean_seed0_lr3e5", "hr_dev_mean_seed1_lr3e5",
        "hr_dev_mean_two_seed_lr3e5", "two_seed_lr3e5_relative_reduction_vs_ordinary_lr1e4",
        "two_seed_lr3e5_relative_reduction_vs_pathsource_comm256",
        "training_cost_gpu_hours_seed1", "hr_dev_cost_gpu_hours_seed1",
        "combined_incremental_gpu_hours_seed1", "query_read",
    ) if key in report}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
