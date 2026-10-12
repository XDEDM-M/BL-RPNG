from __future__ import annotations

import argparse
import json
from collections import defaultdict

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


METHODS = ("final_nlt", "path_nlt", "final_nlt_multiscale", "path_nlt_multiscale")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Gray--Scott R2 blind-query statistics.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    report_path = resolve_in_project(
        f"artifacts/reports/gray_scott/r2_blind_query{args.resolution}.json",
        must_exist=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("selection_used_this_query") is not False:
        raise PermissionError("R2 checkpoint selection used the blind query.")
    records = report["records"]
    expected = 4 * 8 * len(METHODS) * 2
    if len(records) != expected:
        raise RuntimeError("R2 blind-query evaluation is incomplete.")
    means = {
        f"{method}_h{horizon}": sum(
            row["relative_l2"] for row in records
            if row["method"] == method and row["horizon"] == horizon
        ) / 32
        for method in METHODS for horizon in (20, 50)
    }
    comparisons = {}
    for horizon in (20, 50):
        selected = [row for row in records if row["horizon"] == horizon]
        for index, baseline in enumerate(
            ("final_nlt", "final_nlt_multiscale", "path_nlt_multiscale")
        ):
            comparisons[f"path_nlt_vs_{baseline}_h{horizon}"] = paired_hierarchical_bootstrap(
                selected,
                method="path_nlt",
                baseline=baseline,
                replicates=args.replicates,
                seed=20260906 + args.resolution * 10 + horizon + index,
            )
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["horizon"] == 50:
            grouped[row["boundary_regime"]][row["method"]].append(row["relative_l2"])
    family_means = {
        family: {method: sum(values) / len(values) for method, values in methods.items()}
        for family, methods in grouped.items()
    }
    primary20 = comparisons["path_nlt_vs_final_nlt_h20"]
    primary50 = comparisons["path_nlt_vs_final_nlt_h50"]
    payload = {
        "schema": "blrpng_gray_scott_r2_blind_evaluation_gate_v1",
        "frozen": True,
        "passed": (
            primary20["relative_improvement"] > 0
            and primary50["relative_improvement"] > 0
        ),
        "seed": 42,
        "resolution": args.resolution,
        "selected_method": "path_nlt",
        "method_means": means,
        "comparisons": comparisons,
        "h50_family_means": family_means,
        "selection_used_this_query": False,
        "statistical_scope": "seed42 development confirmation; not a multi-seed paper claim",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }
    with workspace_audit(f"summarize_gray_scott_r2_query{args.resolution}") as audit:
        output = write_json(
            f"artifacts/manifests/gray_scott_r2_blind_query{args.resolution}_gate.json",
            payload,
        )
        audit.update({"output": str(output), "passed": payload["passed"]})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
