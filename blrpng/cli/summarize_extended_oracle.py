from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def _load(path: str | Path) -> dict[str, Any]:
    resolved = resolve_in_project(path, must_exist=True)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _legacy_path_report(task: int) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (PROJECT_ROOT / "artifacts/reports").glob(
            f"hr_convergence_task{task}_query256_h20_*.json"
        )
    )
    for path in candidates:
        payload = _load(path)
        if any(row.get("method") == "path" for row in payload.get("records", [])):
            return path, payload
    raise FileNotFoundError(f"No path-initialized m64 query report for task {task}.")


def _query_rows(task: int, budget: int) -> tuple[list[dict[str, Any]], Path]:
    if budget == 64:
        path, payload = _legacy_path_report(task)
        rows = [row for row in payload["records"] if row["method"] == "path"]
        return rows, path
    path = resolve_in_project(
        f"artifacts/reports/task{task}_m{budget}_hr_path_query256_h20.json",
        must_exist=True,
    )
    payload = _load(path)
    rows = []
    for row in payload["metrics"]["rollouts"]:
        rows.append(
            {
                "seed": 42,
                "boundary_id": int(row["boundary_id"]),
                "initial_condition_id": int(row["initial_condition_id"]),
                "relative_l2": float(row["relative_l2"]),
            }
        )
    return rows, path


def _validation_metric(task: int, budget: int) -> float:
    if budget == 64:
        _, payload = _legacy_path_report(task)
        candidates = [row for row in payload["candidates"] if row["label"] == "path"]
        if len(candidates) != 1:
            raise ValueError(f"Task {task} has {len(candidates)} path m64 candidates.")
        return float(candidates[0]["validation_metric"])
    payload = _load(
        f"artifacts/reports/v2_task{task:02d}_m{budget}_s42_"
        "hr_oracle_path_conv_path_lr3em04_e100_training.json"
    )
    return float(payload["best_validation_metric"])


def _mean(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("Cannot average an empty query result.")
    return sum(float(row["relative_l2"]) for row in rows) / len(rows)


def _paired_improvement(
    lower_rows: list[dict[str, Any]],
    higher_rows: list[dict[str, Any]],
    *,
    lower_budget: int,
    higher_budget: int,
    task: int,
    replicates: int,
) -> dict[str, Any]:
    records = []
    for method, rows in ((f"m{lower_budget}", lower_rows), (f"m{higher_budget}", higher_rows)):
        for row in rows:
            records.append({**row, "method": method})
    report = paired_hierarchical_bootstrap(
        records,
        method=f"m{higher_budget}",
        baseline=f"m{lower_budget}",
        replicates=replicates,
        seed=20260827 + task + higher_budget,
    )
    return {
        **report,
        "lower_budget": lower_budget,
        "higher_budget": higher_budget,
    }


def _reference_means(task: int) -> dict[str, float]:
    payload = _load("artifacts/reports/v2_seed42_query256_h20.json")
    output: dict[str, float] = {}
    for method in ("nlt", "lr_scratch"):
        rows = [
            row for row in payload["records"]
            if int(row["target_task_id"]) == task and row["method"] == method
        ]
        output[method] = _mean(rows)
    return output


def build_report(
    tasks: list[int], budgets: list[int], *, replicates: int
) -> dict[str, Any]:
    task_reports: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    complete_three_budget_curves = True
    data_plateau_pass = True
    for task in tasks:
        rows_by_budget: dict[int, list[dict[str, Any]]] = {}
        query_means: dict[str, float] = {}
        validation_means: dict[str, float] = {}
        for budget in budgets:
            try:
                rows, source = _query_rows(task, budget)
                validation = _validation_metric(task, budget)
            except FileNotFoundError:
                complete_three_budget_curves = False
                continue
            rows_by_budget[budget] = rows
            query_means[str(budget)] = _mean(rows)
            validation_means[str(budget)] = validation
            source_hashes[str(source.relative_to(PROJECT_ROOT))] = sha256_file(source)
        comparisons = []
        available = sorted(rows_by_budget)
        for lower, higher in zip(available, available[1:]):
            comparisons.append(
                _paired_improvement(
                    rows_by_budget[lower], rows_by_budget[higher],
                    lower_budget=lower, higher_budget=higher, task=task,
                    replicates=replicates,
                )
            )
        if len(available) < 3:
            complete_three_budget_curves = False
            data_plateau_pass = False
        elif not all(
            float(row["relative_improvement"]) < 0.01 for row in comparisons[-2:]
        ):
            data_plateau_pass = False
        references = _reference_means(task)
        largest = max(available)
        task_reports[str(task)] = {
            "query_relative_l2": query_means,
            "validation_relative_l2": validation_means,
            "paired_budget_improvements": comparisons,
            "references": references,
            "nlt_to_largest_hr_error_ratio": references["nlt"] / query_means[str(largest)],
            "lr_scratch_to_nlt_relative_reduction": (
                references["lr_scratch"] - references["nlt"]
            ) / references["lr_scratch"],
            "largest_completed_budget": largest,
        }
    return {
        "schema": "blrpng_extended_oracle_query_audit_v1",
        "seed": 42,
        "resolution": 256,
        "horizon": 20,
        "tasks": task_reports,
        "gates": {
            "all_representative_tasks_have_three_budgets": complete_three_budget_curves,
            "two_consecutive_query_improvements_below_1pct": data_plateau_pass,
            "saturated_hr_oracle": bool(
                complete_three_budget_curves and data_plateau_pass
            ),
        },
        "source_sha256": source_hashes,
        "interpretation": (
            "All support-budget comparisons use the same path initialization and the same "
            "frozen 256^2/20-step query initial conditions. Validation and query metrics "
            "remain separate. Missing budgets fail rather than pass the saturation gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired m64/m128/m256 HR oracle query curves."
    )
    parser.add_argument("--tasks", default="0,2,7")
    parser.add_argument("--budgets", default="64,128,256")
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument(
        "--output",
        default="artifacts/reports/oracle_extension_query_audit_seed42.json",
    )
    args = parser.parse_args()
    tasks = [int(value) for value in args.tasks.split(",") if value]
    budgets = [int(value) for value in args.budgets.split(",") if value]
    with workspace_audit("summarize_extended_oracle") as audit:
        report = build_report(tasks, budgets, replicates=args.replicates)
        output = write_json(args.output, report)
        audit["output"] = str(output)
    print(json.dumps({"output": str(output), "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
