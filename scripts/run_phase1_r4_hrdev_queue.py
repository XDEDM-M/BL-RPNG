from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.paths import PROJECT_ROOT, sha256_file


TRIAL_IDS = (
    "pathsource_firstpath_ratio005_comm256-r4full",
    "pathsource_endpoint_ratio005_comm256-r4full",
)
SEEDS = (0, 1, 42)
TASKS = tuple(range(6))
SUPPORT_SEED = 271829
SUPPORT_BUDGET = 4
RESULTS_ROOT = PROJECT_ROOT / "results/phase1/R4"
REPORTS_ROOT = PROJECT_ROOT / "artifacts/reports/phase1_hrdev_evaluations"
EVALUATOR = PROJECT_ROOT / "scripts/evaluate_phase1_hrdev.py"


def _cell_paths(trial_id: str, seed: int, task: int) -> tuple[Path, Path]:
    run_dir = RESULTS_ROOT / f"{trial_id}_m{SUPPORT_BUDGET}_s{seed}_ss{SUPPORT_SEED}"
    result = run_dir / f"task{task:02d}.json"
    report = REPORTS_ROOT / f"{trial_id}_task{task:02d}_s{seed}_ss{SUPPORT_SEED}.json"
    return result, report


def _validate_frozen_lr_matrix() -> list[tuple[str, int, int, Path, Path]]:
    cells = []
    for trial_id in TRIAL_IDS:
        for seed in SEEDS:
            run_dir = RESULTS_ROOT / f"{trial_id}_m{SUPPORT_BUDGET}_s{seed}_ss{SUPPORT_SEED}"
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"R4 LR arm/seed is not complete: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            descriptor = summary.get("descriptor", {})
            if (
                summary.get("trial_id") != trial_id
                or int(descriptor.get("model_seed", -1)) != seed
                or int(descriptor.get("support_seed", -1)) != SUPPORT_SEED
                or int(descriptor.get("support_budget", -1)) != SUPPORT_BUDGET
                or descriptor.get("smoke_only") is not False
                or summary.get("query_opened_or_loaded") is not False
                or sorted(int(task) for task in descriptor.get("task_ids", [])) != list(TASKS)
            ):
                raise ValueError(f"R4 LR summary does not match the frozen matrix: {summary_path}")
            for task in TASKS:
                result_path, report_path = _cell_paths(trial_id, seed, task)
                if not result_path.exists():
                    raise FileNotFoundError(f"Missing completed LR result: {result_path}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    result.get("trial_id") != trial_id
                    or int(result.get("model_seed", -1)) != seed
                    or int(result.get("task", -1)) != task
                    or int(result.get("support_seed", -1)) != SUPPORT_SEED
                    or int(result.get("support_budget", -1)) != SUPPORT_BUDGET
                    or result.get("query_metric") is not None
                    or "LR" not in str(result.get("selection_data", "")).upper()
                    or result.get("checkpoint_sha256") is None
                ):
                    raise ValueError(f"LR result is not eligible for HR-dev evaluation: {result_path}")
                cells.append((trial_id, seed, task, result_path, report_path))
    if len(cells) != 36:
        raise AssertionError(f"Expected the frozen 36 new R4 cells, got {len(cells)}")
    return cells


def _validate_existing_report(report_path: Path, result_path: Path, trial_id: str, seed: int, task: int) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        report.get("model_trial_id") != trial_id
        or int(report.get("model_seed", -1)) != seed
        or int(report.get("task", -1)) != task
        or int(report.get("support_seed", -1)) != SUPPORT_SEED
        or int(report.get("support_budget", -1)) != SUPPORT_BUDGET
        or report.get("task_result_sha256") != sha256_file(result_path)
        or report.get("checkpoint_sha256") != result.get("checkpoint_sha256")
        or report.get("query_read") is not False
        or report.get("training_or_within_task_checkpoint_selection_used_hr") is not False
        or report.get("hr_validation_is_development_only") is not True
    ):
        raise ValueError(f"Existing HR-dev output does not match its frozen LR result: {report_path}")


def main() -> None:
    cells = _validate_frozen_lr_matrix()
    completed = 0
    for trial_id, seed, task, result_path, report_path in cells:
        if report_path.exists():
            _validate_existing_report(report_path, result_path, trial_id, seed, task)
            completed += 1
            print(f"SKIP verified existing HR-dev {trial_id} seed={seed} task={task}", flush=True)
            continue
        relative_result = result_path.relative_to(PROJECT_ROOT)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--task-result",
            str(relative_result),
            "--task",
            str(task),
            "--device",
            "cuda:0",
        ]
        print(f"START HR-dev {trial_id} seed={seed} task={task}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        _validate_existing_report(report_path, result_path, trial_id, seed, task)
        completed += 1
    print(json.dumps({"verified_hrdev_reports": completed, "expected": 36, "query_read": False}, indent=2))


if __name__ == "__main__":
    main()
