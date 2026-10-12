from __future__ import annotations

import json
import numpy as np

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def _load(path: str) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text())


def main() -> None:
    summary = _load("artifacts/reports/v2_seed42_calibration_summary.json")
    primary = _load("artifacts/reports/v2_seed42_query256_h20.json")["records"]
    raw = _load("artifacts/reports/v2_seed42_query256_h20_raw_scratch.json")["records"]
    exploratory = _load("artifacts/reports/v2_seed42_query256_h20_defect_weight_sweep.json")["records"]
    all_records = primary + raw + exploratory
    means = {method: float(np.mean([row["relative_l2"] for row in all_records if row["method"] == method]))
             for method in sorted({row["method"] for row in all_records})}
    comparisons = {}
    for method, baseline in [
        ("blrpng_v2", "lr_scratch"), ("blrpng_v2", "ordinary_final"),
        ("pathpre_final", "ordinary_final"), ("nlt", "blrpng_v2"),
        ("blrpng_plus_d1e2", "blrpng_v2"), ("blrpng_v2", "raw_lr_scratch"),
    ]:
        comparisons[f"{method}_vs_{baseline}"] = paired_hierarchical_bootstrap(
            all_records, method=method, baseline=baseline, replicates=50_000, seed=20260813)
    report = {
        "schema": "blrpng_v2_seed42_frozen_report_v1", "frozen": True,
        "confirmatory": {
            "primary_method": "blrpng_v2", "means_256_h20": means,
            "oracle_ratio": summary["oracle_ratio"], "speedup": summary["speedup"],
            "gate": summary["gate"],
            "comparisons": {key: value for key, value in comparisons.items()
                            if "d1e2" not in key and not key.startswith("nlt")},
            "horizon50_means": summary["horizon50_means"],
        },
        "exploratory": {
            "not_eligible_for_primary_gate": True,
            "comparisons": {key: value for key, value in comparisons.items()
                            if "d1e2" in key or key.startswith("nlt")},
            "unlabeled_defect_selection": _load("artifacts/reports/v2_unlabeled_defect_adapter_selection.json"),
        },
        "solver_timing_repeats": _load("artifacts/reports/v2_solver_cost_benchmark.json"),
        "data_verification": _load("artifacts/reports/allen_cahn_v2_data_verification.json"),
        "figure_manifest": _load("artifacts/reports/v2_figure_manifest.json"),
    }
    with workspace_audit("freeze_seed42_report"):
        output = write_json("artifacts/reports/v2_seed42_frozen_report.json", report)
    print(json.dumps({"output": str(output), "gate_passed": report["confirmatory"]["gate"]["passed"]}, indent=2))


if __name__ == "__main__":
    main()
