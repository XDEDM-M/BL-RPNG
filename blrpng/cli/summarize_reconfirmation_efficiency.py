from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


BUDGETS = (1, 2, 4, 8)


def _task_families() -> dict[int, str]:
    path = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["target_task_id"]): str(row["family"])
        for row in payload["tasks"]
    }


def _best_validation(payload: dict) -> tuple[int, float]:
    history = payload["metadata"]["history"]
    if not history:
        raise ValueError("NLT checkpoint has no training history.")
    best = min(
        history,
        key=lambda row: float(row["validation"]["rollout_rel_l2"]),
    )
    return int(best["epoch"]), float(best["validation"]["rollout_rel_l2"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the NLT support-size curve using LR validation only."
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/reports/"
            "allen_cahn_reconfirmation_lr_validation_efficiency.json"
        ),
    )
    args = parser.parse_args()
    families = _task_families()
    rows: list[dict] = []
    by_budget: dict[int, list[float]] = defaultdict(list)
    by_family_budget: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task, family in sorted(families.items()):
        for budget in BUDGETS:
            checkpoint = resolve_in_project(
                "artifacts/checkpoints/reconfirmation/"
                f"reconfirmation_task{task:02d}_m{budget}_s42_nlt.adapter.pt",
                must_exist=True,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("format") != "blrpng_adapter_delta_v1":
                raise ValueError(f"Unexpected adapter format: {checkpoint}")
            if not payload.get("adapter_state"):
                raise ValueError(f"Empty adapter state: {checkpoint}")
            best_epoch, value = _best_validation(payload)
            by_budget[budget].append(value)
            by_family_budget[family][budget].append(value)
            rows.append(
                {
                    "task": task,
                    "family": family,
                    "support_budget": budget,
                    "best_epoch": best_epoch,
                    "validation_rollout_rel_l2": value,
                    "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "training_seconds": float(
                        payload["metadata"].get("elapsed_seconds", 0.0)
                    ),
                    "hr_query_used": False,
                }
            )
    means = {
        str(budget): sum(values) / len(values)
        for budget, values in sorted(by_budget.items())
    }
    family_means = {
        family: {
            str(budget): sum(values) / len(values)
            for budget, values in sorted(curves.items())
        }
        for family, curves in sorted(by_family_budget.items())
    }
    nonincreasing_task_count = sum(
        all(
            next(row["validation_rollout_rel_l2"] for row in rows
                 if row["task"] == task and row["support_budget"] == right)
            <= next(row["validation_rollout_rel_l2"] for row in rows
                    if row["task"] == task and row["support_budget"] == left)
            for left, right in zip(BUDGETS, BUDGETS[1:])
        )
        for task in families
    )
    payload = {
        "schema": "blrpng_reconfirmation_lr_efficiency_v1",
        "seed": 42,
        "method": "nlt",
        "selection_data": "64x64 LR validation only",
        "query_opened_for_selection": False,
        "budgets": list(BUDGETS),
        "tasks": len(families),
        "checkpoint_rows": rows,
        "mean_validation_rollout_rel_l2": means,
        "family_mean_validation_rollout_rel_l2": family_means,
        "nonincreasing_tasks": nonincreasing_task_count,
        "interpretation_limit": (
            "Development diagnostic only; confirmatory data-efficiency claims use "
            "the frozen HR query protocol."
        ),
    }
    with workspace_audit("summarize_reconfirmation_efficiency") as audit:
        output = write_json(args.output, payload)
        freeze = write_json(
            "artifacts/manifests/"
            "allen_cahn_reconfirmation_efficiency_checkpoint_freeze.json",
            {
                "schema": "blrpng_reconfirmation_efficiency_freeze_v1",
                "passed": True,
                "seed": 42,
                "method": "nlt",
                "budgets": list(BUDGETS),
                "query_used": False,
                "report": str(output.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(output),
                "checkpoints": [
                    {
                        "task": row["task"],
                        "support_budget": row["support_budget"],
                        "checkpoint": row["checkpoint"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                    for row in rows
                ],
            },
        )
        audit.update({"output": str(output), "freeze": str(freeze)})
    print(json.dumps({
        "output": str(output),
        "freeze": str(freeze),
        "mean_validation_rollout_rel_l2": means,
        "nonincreasing_tasks": nonincreasing_task_count,
    }, indent=2))


if __name__ == "__main__":
    main()
