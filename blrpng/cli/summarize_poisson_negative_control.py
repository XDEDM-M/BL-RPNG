from __future__ import annotations

import json
from pathlib import Path

from blrpng.analysis.bootstrap import (
    paired_equivalence_test,
    paired_hierarchical_bootstrap,
)
from blrpng.analysis.stage_gates import poisson_negative_control_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


ROOT = Path("artifacts/experiments/poisson_negative_control")


def _load(path: str | Path) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))


def _nlt_records(method: str, resolution: int) -> list[dict]:
    payload = _load(ROOT / f"seed42_{method}_frozen_query.json")
    records = []
    for evaluation in payload["evaluations"]:
        if int(evaluation["resolution"]) != resolution:
            continue
        for row in evaluation["metrics"]["rollouts"]:
            records.append({
                "seed": 42,
                "boundary_id": int(row["boundary_id"]),
                "initial_condition_id": int(row["initial_condition_id"]),
                "target_task_id": int(row["target_task_id"]),
                "method": method,
                "relative_l2": float(row["rollout_relative_l2"]),
            })
    return records


def _scratch_records(
    resolution: int, method: str = "lr_scratch_converged"
) -> list[dict]:
    records = []
    for task in range(30):
        name = (
            f"seed42_lr_scratch_task{task:02d}.json"
            if method == "lr_scratch_converged"
            else f"seed42_{method}_task{task:02d}.json"
        )
        payload = _load(ROOT / name)
        for row in payload["query"][str(resolution)]["rollouts"]:
            records.append({
                "seed": 42,
                "boundary_id": int(row["boundary_id"]),
                "initial_condition_id": int(row["initial_condition_id"]),
                "target_task_id": int(row["target_task_id"]),
                "method": method,
                "relative_l2": float(row["rollout_relative_l2"]),
            })
    return records


def _mean(records: list[dict], method: str) -> float:
    values = [row["relative_l2"] for row in records if row["method"] == method]
    return sum(values) / len(values)


def main() -> None:
    protocol = _load("artifacts/manifests/poisson_negative_control_seed42_protocol.json")
    family_by_task = {
        int(task["target_task_id"]): task["family"] for task in protocol["targets"]
    }
    resolution_reports = {}
    all_records = {}
    for resolution in (128, 256):
        records = (
            _nlt_records("ordinary_nlt", resolution)
            + _nlt_records("path_nlt", resolution)
            + _scratch_records(resolution)
            + _scratch_records(resolution, "boundary_augmented_lr_scratch")
        )
        all_records[str(resolution)] = records
        path_vs_ordinary = paired_hierarchical_bootstrap(
            records, method="path_nlt", baseline="ordinary_nlt",
            replicates=50_000, seed=20260901 + resolution,
        )
        ordinary_vs_scratch = paired_hierarchical_bootstrap(
            records, method="ordinary_nlt", baseline="lr_scratch_converged",
            replicates=50_000, seed=20261901 + resolution,
        )
        path_vs_scratch = paired_hierarchical_bootstrap(
            records, method="path_nlt", baseline="lr_scratch_converged",
            replicates=50_000, seed=20262901 + resolution,
        )
        ordinary_vs_boundary_augmented = paired_hierarchical_bootstrap(
            records, method="ordinary_nlt", baseline="boundary_augmented_lr_scratch",
            replicates=50_000, seed=20263401 + resolution,
        )
        equivalence = paired_equivalence_test(
            records, method="path_nlt", baseline="ordinary_nlt",
            equivalence_lower=0.95, noninferiority_margin=1.05,
            replicates=50_000, seed=20263901 + resolution,
        )
        family_means = {}
        for family in ("fourier", "bump", "piecewise_smooth"):
            selected = [
                row for row in records
                if family_by_task[row["target_task_id"]] == family
            ]
            family_means[family] = {
                method: _mean(selected, method)
                for method in (
                    "ordinary_nlt", "path_nlt", "lr_scratch_converged",
                    "boundary_augmented_lr_scratch",
                )
            }
        resolution_reports[str(resolution)] = {
            "means": {
                method: _mean(records, method)
                for method in (
                    "ordinary_nlt", "path_nlt", "lr_scratch_converged",
                    "boundary_augmented_lr_scratch",
                )
            },
            "family_means": family_means,
            "path_vs_ordinary": path_vs_ordinary,
            "path_ordinary_equivalence_5pct": equivalence,
            "ordinary_vs_scratch": ordinary_vs_scratch,
            "path_vs_scratch": path_vs_scratch,
            "ordinary_vs_boundary_augmented_scratch": ordinary_vs_boundary_augmented,
        }
    primary = resolution_reports["256"]
    negative_gate = poisson_negative_control_gate(
        primary["path_vs_ordinary"]["relative_improvement"]
    )
    payload = {
        "schema": "blrpng_poisson_negative_control_seed42_summary_v1",
        "frozen": True,
        "seed": 42,
        "support_budget": 4,
        "source_cases": {"train": 500, "validation": 100},
        "target_tasks": 30,
        "query_cases_per_task": 16,
        "resolutions": resolution_reports,
        "primary_negative_control_gate": negative_gate,
        "interpretation": {
            "linear_identity": "one Newton correction equals the final lifted solution target",
            "expected": "path pretraining should not materially outperform final-only pretraining",
            "mechanism_claim": (
                "A pass supports nonlinear Newton trajectory information, rather than an arbitrary "
                "auxiliary label, as the source of the Allen-Cahn path benefit."
            ),
            "scope": "seed42 only; multi-seed confirmation remains required",
        },
    }
    with workspace_audit("summarize_poisson_negative_control") as audit:
        output = write_json(ROOT / "seed42_summary.json", payload)
        gate = write_json(
            "artifacts/manifests/poisson_negative_control_seed42_stage_gate.json",
            {
                "schema": "blrpng_poisson_negative_control_stage_gate_v1",
                "passed": bool(negative_gate["passed"]),
                "seed": 42,
                "negative_control": negative_gate,
                "primary_mean_relative_l2": primary["means"],
                "ordinary_vs_converged_lr_scratch": primary["ordinary_vs_scratch"],
                "next_stage": "gray_scott_seed42" if negative_gate["passed"] else None,
                "multi_seed_required_for_paper_claim": True,
                "summary": str(output),
            },
        )
        audit.update({"output": str(output), "gate": str(gate), "passed": negative_gate["passed"]})
    print(json.dumps({
        "output": str(output), "gate": str(gate),
        "primary_256_means": primary["means"],
        "path_vs_ordinary": primary["path_vs_ordinary"],
        "path_ordinary_equivalence": primary["path_ordinary_equivalence_5pct"],
        "ordinary_vs_scratch": primary["ordinary_vs_scratch"],
        "ordinary_vs_boundary_augmented_scratch": primary[
            "ordinary_vs_boundary_augmented_scratch"
        ],
        "negative_control_gate": negative_gate,
    }, indent=2))


if __name__ == "__main__":
    main()
