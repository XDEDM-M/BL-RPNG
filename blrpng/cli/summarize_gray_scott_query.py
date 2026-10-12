from __future__ import annotations

import argparse
import json
from collections import defaultdict

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Gray--Scott query statistics.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    require_prerequisite_gate(
        f"artifacts/manifests/gray_scott_seed42_query{args.resolution}_data_gate.json"
    )
    part_paths = [
        resolve_in_project(
            f"artifacts/reports/gray_scott/query{args.resolution}_tasks00_08.json",
            must_exist=True,
        ),
        resolve_in_project(
            f"artifacts/reports/gray_scott/query{args.resolution}_tasks08_16.json",
            must_exist=True,
        ),
    ]
    records = []
    for path in part_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("frozen") or payload.get("selection_used_hr_query") is not False:
            raise PermissionError(f"Unfrozen or selection-leaking query report: {path}")
        records.extend(payload["records"])
    expected = 16 * 8 * len(METHODS) * 2
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} paired records, got {len(records)}.")
    units = {
        (row["target_task_id"], row["initial_condition_id"], row["method"], row["horizon"])
        for row in records
    }
    if len(units) != expected:
        raise RuntimeError("Duplicate Gray--Scott query statistical units.")

    method_means = {
        f"{method}_h{horizon}": sum(
            row["relative_l2"] for row in records
            if row["method"] == method and row["horizon"] == horizon
        ) / sum(
            row["method"] == method and row["horizon"] == horizon for row in records
        )
        for method in METHODS for horizon in (20, 50)
    }
    comparisons = {}
    for horizon in (20, 50):
        selected = [row for row in records if row["horizon"] == horizon]
        for baseline in ("ordinary_nlt", "safe_path_nlt"):
            comparisons[f"safe_path_nlt_gated_vs_{baseline}_h{horizon}"] = (
                paired_hierarchical_bootstrap(
                    selected,
                    method="safe_path_nlt_gated",
                    baseline=baseline,
                    replicates=args.replicates,
                    seed=20260902 + horizon + (baseline == "safe_path_nlt"),
                )
            )

    family_values = defaultdict(lambda: defaultdict(list))
    task_values = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["horizon"] != 20:
            continue
        family_values[row["boundary_regime"]][row["method"]].append(row["relative_l2"])
        task_values[int(row["target_task_id"])][row["method"]].append(row["relative_l2"])
    family_means = {
        family: {
            method: sum(values) / len(values)
            for method, values in methods.items()
        }
        for family, methods in family_values.items()
    }
    family_gaps = {
        family: values["safe_path_nlt_gated"] / values["ordinary_nlt"] - 1.0
        for family, values in family_means.items()
    }
    task_wins = sum(
        sum(values["safe_path_nlt_gated"]) / len(values["safe_path_nlt_gated"])
        < sum(values["ordinary_nlt"]) / len(values["ordinary_nlt"])
        for values in task_values.values()
    )
    h20 = comparisons["safe_path_nlt_gated_vs_ordinary_nlt_h20"]
    h50 = comparisons["safe_path_nlt_gated_vs_ordinary_nlt_h50"]
    gate_checks = {
        "h20_improvement_ci_lower_positive": h20["ci95"][0] > 0.0,
        "h50_point_degradation_below_5pct": h50["relative_improvement"] > -0.05,
        "minimum_task_wins_9_of_16": task_wins >= 9,
        "no_shift_family_degrades_over_5pct": max(family_gaps.values()) <= 0.05,
    }
    payload = {
        "schema": "blrpng_gray_scott_query_evaluation_gate_v1",
        "frozen": True,
        "passed": all(gate_checks.values()),
        "seed": 42,
        "resolution": args.resolution,
        "support_budget": 4,
        "query_checkpoint_selection_used": False,
        "method_means": method_means,
        "comparisons": comparisons,
        "h20_family_means": family_means,
        "h20_family_relative_gaps": family_gaps,
        "h20_task_wins": task_wins,
        "gate_checks": gate_checks,
        "source_reports": [
            {"path": str(path), "sha256": sha256_file(path)} for path in part_paths
        ],
        "authorized_next_experiment": (
            "Generate the frozen 256-grid query."
            if args.resolution == 128 and all(gate_checks.values())
            else None
        ),
    }
    with workspace_audit(f"summarize_gray_scott_query{args.resolution}") as audit:
        output = write_json(
            f"artifacts/manifests/gray_scott_seed42_query{args.resolution}_evaluation_gate.json",
            payload,
        )
        audit.update({"output": str(output), "passed": payload["passed"]})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
