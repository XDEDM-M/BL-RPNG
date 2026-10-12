from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


METHODS = (
    "nlt",
    "ordinary_nlt",
    "nlt_path",
    "nlt_commutator",
    "nlt_commutator_defect",
)


def _task_families() -> dict[int, str]:
    path = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["target_task_id"]): str(row["family"]) for row in payload["tasks"]}


def _checkpoint(task: int, method: str) -> Path:
    return resolve_in_project(
        f"artifacts/checkpoints/reconfirmation/"
        f"reconfirmation_task{task:02d}_m4_s42_{method}.adapter.pt",
        must_exist=True,
    )


def _best_validation(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    history = payload["metadata"]["history"]
    best = min(history, key=lambda row: float(row["validation"]["rollout_rel_l2"]))
    stratum = best["validation"]["strata"][0]
    return payload, {
        "best_epoch": int(best["epoch"]),
        "relative_l2": float(best["validation"]["rollout_rel_l2"]),
        "per_rollout_rel_l2": [float(value) for value in stratum["per_rollout_rel_l2"]],
        "initial_condition_ids": [int(value) for value in stratum["initial_condition_ids"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the reconfirmation method using frozen LR validation only."
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    families = _task_families()
    records: list[dict[str, Any]] = []
    task_metrics: dict[str, dict[str, float]] = defaultdict(dict)
    checkpoint_rows = []
    for task in sorted(families):
        for method in METHODS:
            path = _checkpoint(task, method)
            payload, validation = _best_validation(path)
            task_metrics[str(task)][method] = validation["relative_l2"]
            checkpoint_rows.append({
                "task": task,
                "family": families[task],
                "method": method,
                "checkpoint": str(path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(path),
                "best_epoch": validation["best_epoch"],
                "validation_relative_l2": validation["relative_l2"],
                "training_seconds": float(payload["metadata"]["elapsed_seconds"]),
                "hr_query_used": False,
            })
            for initial_id, value in zip(
                validation["initial_condition_ids"], validation["per_rollout_rel_l2"]
            ):
                records.append({
                    "seed": 42,
                    "boundary_id": task,
                    "initial_condition_id": initial_id,
                    "method": method,
                    "relative_l2": value,
                })
    comparisons = {}
    for method in METHODS[1:]:
        comparisons[f"{method}_vs_nlt"] = paired_hierarchical_bootstrap(
            records,
            method=method,
            baseline="nlt",
            replicates=args.replicates,
            seed=20260831 + METHODS.index(method),
        )
    family_metrics: dict[str, dict[str, float]] = {}
    for family in sorted(set(families.values())):
        tasks = [task for task, value in families.items() if value == family]
        family_metrics[family] = {
            method: sum(task_metrics[str(task)][method] for task in tasks) / len(tasks)
            for method in METHODS
        }
    path_comparison = comparisons["nlt_path_vs_nlt"]
    task_wins = sum(
        task_metrics[str(task)]["nlt_path"] < task_metrics[str(task)]["nlt"]
        for task in families
    )
    family_wins = sum(
        values["nlt_path"] < values["nlt"] for values in family_metrics.values()
    )
    promotion_checks = {
        "relative_improvement_at_least_2pct": (
            float(path_comparison["relative_improvement"]) >= 0.02
        ),
        "task_wins_at_least_32_of_48": task_wins >= 32,
        "family_wins_at_least_5_of_6": family_wins >= 5,
        "ci_lower_bound_positive": float(path_comparison["ci95"][0]) > 0.0,
    }
    promote_path = all(promotion_checks.values())
    selected_method = "nlt_path" if promote_path else "nlt"
    report_payload = {
        "schema": "blrpng_reconfirmation_lr_validation_selection_v1",
        "seed": 42,
        "support_budget": 4,
        "query_opened": False,
        "selection_data": "64x64 LR validation only",
        "methods": list(METHODS),
        "task_metrics": task_metrics,
        "family_metrics": family_metrics,
        "comparisons": comparisons,
        "path_promotion": {
            "predeclared_rule": (
                ">=2% paired improvement, >=32/48 task wins, >=5/6 family wins, "
                "and positive 95% CI lower bound"
            ),
            "checks": promotion_checks,
            "task_wins": task_wins,
            "family_wins": family_wins,
            "passed": promote_path,
        },
        "selected_primary_method": selected_method,
        "checkpoints": checkpoint_rows,
    }
    with workspace_audit("summarize_reconfirmation_validation") as audit:
        report = write_json(
            "artifacts/reports/allen_cahn_reconfirmation_lr_validation_selection.json",
            report_payload,
        )
        freeze = write_json(
            "artifacts/manifests/allen_cahn_reconfirmation_checkpoint_freeze.json",
            {
                "schema": "blrpng_reconfirmation_checkpoint_freeze_v1",
                "passed": True,
                "seed": 42,
                "support_budget": 4,
                "selected_primary_method": selected_method,
                "query_opened_during_selection": False,
                "selection_report": str(report.relative_to(PROJECT_ROOT)),
                "selection_report_sha256": sha256_file(report),
                "selected_checkpoints": [
                    row for row in checkpoint_rows if row["method"] == selected_method
                ],
            },
        )
        audit.update({"report": str(report), "freeze": str(freeze)})
    print(json.dumps({
        "selected_primary_method": selected_method,
        "path_promotion": report_payload["path_promotion"],
        "comparisons": {
            key: {
                "relative_improvement": value["relative_improvement"],
                "ci95": value["ci95"],
            }
            for key, value in comparisons.items()
        },
        "freeze": str(freeze),
    }, indent=2))


if __name__ == "__main__":
    main()
