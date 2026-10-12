from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


TASKS = (0, 4, 8, 12)
METHODS = ("final_nlt", "path_nlt", "final_nlt_multiscale", "path_nlt_multiscale")


def _adapter_path(task: int, method: str) -> Path:
    return resolve_in_project(
        f"artifacts/checkpoints/gray_scott_r2/"
        f"gray_scott_r2_task{task:02d}_m4_s42_{method}.adapter.pt",
        must_exist=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze query-blind Gray--Scott R2 method selection.")
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    protocol = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    task_specs = {int(row["target_task_id"]): row for row in protocol["targets"]}
    records = []
    checkpoints = {}
    task_metrics = {}
    for task in TASKS:
        checkpoints[str(task)] = {}
        task_metrics[str(task)] = {}
        for method in METHODS:
            path = _adapter_path(task, method)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            history = metadata["history"]
            best = float(metadata["best_validation_metric"])
            best_row = min(history, key=lambda row: float(row["validation"]["rollout_rel_l2"]))
            if abs(float(best_row["validation"]["rollout_rel_l2"]) - best) > 1e-12:
                raise RuntimeError(f"Best-history mismatch for task={task}, method={method}")
            strata = best_row["validation"]["strata"]
            if len(strata) != 1 or len(strata[0]["per_rollout_rel_l2"]) != 4:
                raise RuntimeError("R2 selection requires exactly four 50-step validation rollouts.")
            for initial_id, error in zip(
                strata[0]["initial_condition_ids"], strata[0]["per_rollout_rel_l2"]
            ):
                records.append({
                    "seed": 42,
                    "target_task_id": task,
                    "boundary_id": int(task_specs[task]["boundary_id"]),
                    "initial_condition_id": int(initial_id),
                    "boundary_regime": task_specs[task]["boundary_regime"],
                    "method": method,
                    "relative_l2": float(error),
                    "resolution": 64,
                    "horizon": 50,
                })
            task_metrics[str(task)][method] = best
            checkpoints[str(task)][method] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "best_epoch": int(best_row["epoch"]),
                "validation_rel_l2": best,
                "training_seconds": float(metadata["elapsed_seconds"]),
                "optimizer_steps": int(metadata["optimizer_steps"]),
            }

    means = {
        method: sum(task_metrics[str(task)][method] for task in TASKS) / len(TASKS)
        for method in METHODS
    }
    primary = min(means, key=means.get)
    comparisons = {}
    for index, baseline in enumerate(method for method in METHODS if method != primary):
        comparisons[f"{primary}_vs_{baseline}"] = paired_hierarchical_bootstrap(
            records,
            method=primary,
            baseline=baseline,
            replicates=args.replicates,
            seed=20260905 + index,
        )
    path_vs_final = comparisons["path_nlt_vs_final_nlt"]
    payload = {
        "schema": "blrpng_gray_scott_r2_target_selection_v1",
        "frozen": True,
        "passed": primary == "path_nlt",
        "seed": 42,
        "selection_resolution": 64,
        "selection_horizon": 50,
        "support_budget": 4,
        "tasks": list(TASKS),
        "method_means": means,
        "task_metrics": task_metrics,
        "comparisons": comparisons,
        "selected_method": primary,
        "checkpoints": checkpoints,
        "mechanism_conclusion": {
            "path_pretraining_mean_improvement_over_final_pretraining": (
                1.0 - means["path_nlt"] / means["final_nlt"]
            ),
            "path_pretraining_paired_ci95": path_vs_final["ci95"],
            "multiscale_promoted": False,
            "reason": (
                "Path-pretrain NLT has the lowest frozen LR/50-step mean; adding target "
                "multiscale loss does not improve the mean."
            ),
        },
        "selection_used_high_resolution_query": False,
        "new_query_authorized": primary == "path_nlt",
    }
    with workspace_audit("freeze_gray_scott_r2_target_selection") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_target_selection.json", payload
        )
        audit.update({"output": str(output), "passed": payload["passed"]})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
