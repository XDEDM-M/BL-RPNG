from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


METHODS = (
    "nlt",
    "ordinary_nlt",
    "nlt_path",
    "nlt_commutator",
    "nlt_commutator_defect",
    "lr_scratch",
)


def _families() -> dict[int, str]:
    manifest = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        int(row["target_task_id"]): str(row["family"])
        for row in payload["tasks"]
    }


def _load_report(
    task: int,
    method: str,
    budget: int,
    resolution: int,
    horizon: int,
) -> tuple[str, dict[str, Any]]:
    relative = (
        "artifacts/reports/reconfirmation/"
        f"task{task:02d}_m{budget}_s42_{method}_"
        f"query{resolution}_h{horizon}.json"
    )
    path = resolve_in_project(relative, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (task, method, budget, resolution, horizon)
    actual = (
        int(payload["task"]),
        str(payload["method"]),
        int(payload["support_budget"]),
        int(payload["resolution"]),
        int(payload["horizon"]),
    )
    if actual != expected:
        raise ValueError(f"Evaluation metadata mismatch in {path}: {actual} != {expected}")
    if payload.get("selection_used_hr_query") is not False:
        raise ValueError(f"Checkpoint selection leakage marker in {path}.")
    return relative, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze paired statistics for the new-boundary HR blind query."
    )
    parser.add_argument("--budget", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--resolution", type=int, default=256, choices=(128, 256))
    parser.add_argument("--horizon", type=int, default=20, choices=(20, 50))
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated frozen methods; nlt must be included.",
    )
    args = parser.parse_args()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not methods or methods[0] != "nlt" or any(value not in METHODS for value in methods):
        raise ValueError(f"Unsupported method set: {methods}")
    require_prerequisite_gate(
        "artifacts/manifests/allen_cahn_reconfirmation_checkpoint_freeze.json"
    )
    require_prerequisite_gate(
        "artifacts/manifests/allen_cahn_reconfirmation_query_protocol_freeze.json"
    )
    families = _families()
    records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    task_metrics: dict[int, dict[str, float]] = defaultdict(dict)
    family_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task, family in sorted(families.items()):
        for method in methods:
            relative, payload = _load_report(
                task, method, args.budget, args.resolution, args.horizon
            )
            rollout_rows = payload["metrics"]["rollouts"]
            units = {
                (int(row["boundary_id"]), int(row["initial_condition_id"]))
                for row in rollout_rows
            }
            if len(units) != len(rollout_rows):
                raise ValueError(f"Duplicate query statistical unit in {relative}.")
            # Initial-condition IDs differ across tasks; paired methods for a
            # given task must nevertheless contain exactly the same units.
            task_reference = {
                int(row["initial_condition_id"])
                for row in records
                if row["task"] == task and row["method"] == "nlt"
            }
            current_initials = {initial for _boundary, initial in units}
            if method != "nlt" and task_reference != current_initials:
                raise ValueError(f"Unpaired HR query units for task {task}, method {method}.")
            values = []
            for row in rollout_rows:
                value = float(row["relative_l2"])
                values.append(value)
                records.append({
                    "seed": 42,
                    "task": task,
                    "boundary_id": task,
                    "initial_condition_id": int(row["initial_condition_id"]),
                    "family": family,
                    "method": method,
                    "relative_l2": value,
                })
            mean = sum(values) / len(values)
            task_metrics[task][method] = mean
            family_values[family][method].extend(values)
            reports.append({
                "task": task,
                "family": family,
                "method": method,
                "path": relative,
                "sha256": sha256_file(resolve_in_project(relative)),
                "rollouts": len(values),
                "mean_relative_l2": mean,
                "evaluation_seconds": float(payload["metrics"]["evaluation_seconds"]),
                "boundary_error_max": float(payload["metrics"]["boundary_error_max"]),
            })
    comparisons = {}
    for index, baseline in enumerate(methods):
        if baseline == "nlt":
            continue
        comparisons[f"nlt_vs_{baseline}"] = paired_hierarchical_bootstrap(
            records,
            method="nlt",
            baseline=baseline,
            replicates=args.replicates,
            seed=20260901 + index,
        )
    method_means = {
        method: sum(row["relative_l2"] for row in records if row["method"] == method)
        / sum(row["method"] == method for row in records)
        for method in methods
    }
    family_means = {
        family: {
            method: sum(values) / len(values)
            for method, values in sorted(method_rows.items())
        }
        for family, method_rows in sorted(family_values.items())
    }
    task_wins = {
        baseline: sum(
            task_metrics[task]["nlt"] < task_metrics[task][baseline]
            for task in families
        )
        for baseline in methods
        if baseline != "nlt"
    }
    scratch = comparisons.get("nlt_vs_lr_scratch")
    ordinary = comparisons.get("nlt_vs_ordinary_nlt")
    partial_gate = {
        "scope": "accuracy only; target-cost gate is frozen separately",
        "scratch_improvement_at_least_75pct": bool(
            scratch and float(scratch["relative_improvement"]) >= 0.75
        ),
        "scratch_improvement_ci_lower_positive": bool(
            scratch and float(scratch["ci95"][0]) > 0.0
        ),
        "path_pretraining_improvement_ci_lower_positive": bool(
            ordinary and float(ordinary["ci95"][0]) > 0.0
        ),
        "boundary_error_below_1e_minus_6": all(
            row["boundary_error_max"] < 1e-6 for row in reports
        ),
    }
    partial_gate["passed"] = all(
        value for key, value in partial_gate.items() if key != "scope"
    )
    output_relative = (
        "artifacts/reports/"
        f"allen_cahn_reconfirmation_seed42_query{args.resolution}_"
        f"h{args.horizon}_m{args.budget}.json"
    )
    payload = {
        "schema": "blrpng_reconfirmation_query_summary_v1",
        "frozen": True,
        "seed": 42,
        "support_budget": args.budget,
        "resolution": args.resolution,
        "horizon": args.horizon,
        "primary_endpoint": (
            f"{args.resolution}x{args.resolution} {args.horizon}-step "
            "per-rollout relative L2 on unseen target-boundary query initial conditions"
        ),
        "selection_protocol": "checkpoint frozen from 64x64 validation before HR query generation",
        "methods": list(methods),
        "statistical_units": len(records) // len(methods),
        "method_mean_relative_l2": method_means,
        "family_mean_relative_l2": family_means,
        "task_mean_relative_l2": {str(key): value for key, value in task_metrics.items()},
        "task_wins_for_nlt": task_wins,
        "comparisons": comparisons,
        "accuracy_gate": partial_gate,
        "evaluation_reports": reports,
        "claim_limit": (
            "The m256 HR specialist is resource-capped and unsaturated; this report "
            "does not make saturated-oracle parity claims."
        ),
    }
    with workspace_audit("summarize_reconfirmation_query") as audit:
        output = write_json(output_relative, payload)
        freeze = write_json(
            "artifacts/manifests/"
            f"allen_cahn_reconfirmation_seed42_query{args.resolution}_"
            f"h{args.horizon}_m{args.budget}_freeze.json",
            {
                "schema": "blrpng_reconfirmation_query_freeze_v1",
                "passed": bool(partial_gate["passed"]),
                "scope": "accuracy_only",
                "report": str(output.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(output),
                "resource_capped_oracle": True,
                "saturated_oracle_claim": False,
            },
        )
        audit.update({"output": str(output), "freeze": str(freeze)})
    print(json.dumps({
        "output": str(output),
        "freeze": str(freeze),
        "method_mean_relative_l2": method_means,
        "comparisons": comparisons,
        "accuracy_gate": partial_gate,
    }, indent=2))


if __name__ == "__main__":
    main()
