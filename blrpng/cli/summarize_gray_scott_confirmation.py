from __future__ import annotations

import argparse
import json
from collections import defaultdict

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated", "lifted_scratch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Gray--Scott confirmation statistics.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    require_prerequisite_gate(
        f"artifacts/manifests/gray_scott_seed42_confirmation{args.resolution}_data_gate.json"
    )
    report_path = resolve_in_project(
        f"artifacts/reports/gray_scott/confirmation{args.resolution}.json",
        must_exist=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("selection_used_confirmation_query") is not False:
        raise PermissionError("Confirmation query was used for checkpoint selection.")
    records = report["records"]
    expected = 16 * 8 * len(METHODS) * 2
    units = {
        (row["target_task_id"], row["initial_condition_id"], row["method"], row["horizon"])
        for row in records
    }
    if len(records) != expected or len(units) != expected:
        raise RuntimeError("Confirmation evaluation is incomplete or duplicated.")
    means = {
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
        pairs = (
            ("ordinary_nlt", "lifted_scratch"),
            ("safe_path_nlt", "lifted_scratch"),
            ("safe_path_nlt_gated", "lifted_scratch"),
            ("safe_path_nlt", "ordinary_nlt"),
            ("safe_path_nlt_gated", "ordinary_nlt"),
            ("safe_path_nlt_gated", "safe_path_nlt"),
        )
        for index, (method, baseline) in enumerate(pairs):
            comparisons[f"{method}_vs_{baseline}_h{horizon}"] = paired_hierarchical_bootstrap(
                selected,
                method=method,
                baseline=baseline,
                replicates=args.replicates,
                seed=20260903 + horizon * 10 + index,
            )
    family_rows = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["horizon"] == 20:
            family_rows[row["boundary_regime"]][row["method"]].append(row["relative_l2"])
    family_means = {
        family: {
            method: sum(values) / len(values) for method, values in methods.items()
        }
        for family, methods in family_rows.items()
    }
    ordinary_scratch = comparisons["ordinary_nlt_vs_lifted_scratch_h20"]
    main_scratch = comparisons["safe_path_nlt_gated_vs_lifted_scratch_h20"]
    main_ordinary_h20 = comparisons["safe_path_nlt_gated_vs_ordinary_nlt_h20"]
    main_ordinary_h50 = comparisons["safe_path_nlt_gated_vs_ordinary_nlt_h50"]
    checks = {
        "ordinary_vs_scratch_point_at_least_75pct": ordinary_scratch["relative_improvement"] >= 0.75,
        "ordinary_vs_scratch_ci_lower_at_least_75pct": ordinary_scratch["ci95"][0] >= 0.75,
        "main_vs_scratch_point_at_least_75pct": main_scratch["relative_improvement"] >= 0.75,
        "main_vs_scratch_ci_lower_at_least_75pct": main_scratch["ci95"][0] >= 0.75,
        "main_degradation_vs_ordinary_h20_below_5pct": main_ordinary_h20["relative_improvement"] >= -0.05,
        "main_degradation_vs_ordinary_h50_below_5pct": main_ordinary_h50["relative_improvement"] >= -0.05,
    }
    passed = all(checks.values())
    payload = {
        "schema": "blrpng_gray_scott_confirmation_evaluation_gate_v1",
        "frozen": True,
        "passed": passed,
        "seed": 42,
        "resolution": args.resolution,
        "support_budget": 4,
        "main_method": "safe_path_nlt_gated",
        "query_checkpoint_selection_used": False,
        "method_means": means,
        "comparisons": comparisons,
        "h20_family_means": family_means,
        "gate_checks": checks,
        "evaluation_report": str(report_path),
        "evaluation_report_sha256": sha256_file(report_path),
        "authorized_next_experiment": (
            "Generate the frozen 256-grid confirmation query."
            if args.resolution == 128 and passed else None
        ),
    }
    with workspace_audit(f"summarize_gray_scott_confirmation{args.resolution}") as audit:
        output = write_json(
            f"artifacts/manifests/gray_scott_seed42_confirmation{args.resolution}_evaluation_gate.json",
            payload,
        )
        audit.update({"output": str(output), "passed": passed})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
