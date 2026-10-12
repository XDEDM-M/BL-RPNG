from __future__ import annotations

import json
from pathlib import Path

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


METHODS = (
    "ordinary_nlt", "nlt", "nlt_path", "nlt_commutator", "nlt_commutator_defect"
)


def _load(path: str | Path) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))


def main() -> None:
    evaluation = _load("artifacts/reports/nlt_causal_seed42_query256_h20_exploratory.json")
    records = evaluation["records"]
    means = {
        method: sum(row["relative_l2"] for row in records if row["method"] == method)
        / sum(row["method"] == method for row in records)
        for method in METHODS
    }
    comparisons = {
        f"{method}_vs_nlt": paired_hierarchical_bootstrap(
            records, method=method, baseline="nlt", replicates=50_000, seed=20260820
        )
        for method in METHODS if method != "nlt"
    }
    path_rows = []
    for task in range(8):
        report = _load(
            f"artifacts/reports/v2_task{task:02d}_m4_s42_nlt_path_training.json"
        )
        for epoch in report["history"]:
            train = epoch["train"]
            path_rows.append({"task": task, "epoch": epoch["epoch"], **train})
    active = [row for row in path_rows if row.get("path_gate_active", 0.0) > 0.0]
    path_audit = {
        "epoch_task_rows": len(path_rows),
        "active_fraction": len(active) / max(len(path_rows), 1),
        "mean_gradient_ratio_when_active": (
            sum(row["gradient_ratio"] for row in active) / max(len(active), 1)
        ),
        "mean_path_weight_when_active": (
            sum(row["path_weight"] for row in active) / max(len(active), 1)
        ),
        "all_shared_parameter_counts_positive": all(
            row.get("path_shared_parameter_count", 0.0) > 0.0 for row in path_rows
        ),
        "all_final_shared_gradient_norms_positive": all(
            row.get("final_shared_gradient_norm", 0.0) > 0.0 for row in path_rows
        ),
    }
    payload = {
        "schema": "blrpng_nlt_causal_exploratory_v1",
        "status": "post_query_exploratory_not_primary",
        "seed": 42,
        "resolution": 256,
        "horizon": 20,
        "support_budget": 4,
        "means": means,
        "comparisons": comparisons,
        "path_mechanism_audit": path_audit,
        "interpretation": {
            "path_pretraining_effect_isolated_by": "ordinary_nlt vs nlt",
            "target_path_effect_isolated_by": "nlt vs nlt_path",
            "promotion_rule": (
                "nlt_path remains non-primary unless it wins on a fresh unopened blind set"
            ),
        },
    }
    with workspace_audit("summarize_nlt_causal"):
        output = write_json(
            "artifacts/reports/nlt_causal_seed42_summary_exploratory.json", payload
        )
    print(json.dumps({"output": str(output), "means": means, "path_audit": path_audit}, indent=2))


if __name__ == "__main__":
    main()
