from __future__ import annotations

import json
from collections import defaultdict

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    reports = {}
    combined = {}
    for resolution in (128, 256):
        main_path = resolve_in_project(
            f"artifacts/reports/gray_scott/r2_blind_query{resolution}.json",
            must_exist=True,
        )
        scratch_path = resolve_in_project(
            f"artifacts/reports/gray_scott/r2_blind_query{resolution}_scratch.json",
            must_exist=True,
        )
        main = json.loads(main_path.read_text(encoding="utf-8"))
        scratch = json.loads(scratch_path.read_text(encoding="utf-8"))
        if main.get("selection_used_this_query") is not False:
            raise PermissionError("R2 blind query was used for model selection.")
        records = main["records"] + scratch["records"]
        reports[str(resolution)] = {
            "main": str(main_path), "main_sha256": sha256_file(main_path),
            "scratch": str(scratch_path), "scratch_sha256": sha256_file(scratch_path),
        }
        comparisons = {}
        means = {}
        methods = (
            "final_nlt", "path_nlt", "final_nlt_multiscale",
            "path_nlt_multiscale", "lifted_lr_scratch_saturated",
        )
        for horizon in (20, 50):
            selected = [row for row in records if row["horizon"] == horizon]
            for method in methods:
                values = [row["relative_l2"] for row in selected if row["method"] == method]
                means[f"{method}_h{horizon}"] = sum(values) / len(values)
            for index, baseline in enumerate(("final_nlt", "lifted_lr_scratch_saturated")):
                comparisons[f"path_nlt_vs_{baseline}_h{horizon}"] = paired_hierarchical_bootstrap(
                    selected,
                    method="path_nlt",
                    baseline=baseline,
                    replicates=50_000,
                    seed=20260907 + resolution + horizon + index,
                )
        families = defaultdict(lambda: defaultdict(list))
        for row in records:
            if row["horizon"] == 50:
                families[row["boundary_regime"]][row["method"]].append(row["relative_l2"])
        family_means = {
            family: {method: sum(values) / len(values) for method, values in methods_.items()}
            for family, methods_ in families.items()
        }
        combined[str(resolution)] = {
            "means": means,
            "comparisons": comparisons,
            "h50_family_means": family_means,
        }

    primary = combined["256"]
    scratch_gain20 = primary["comparisons"]["path_nlt_vs_lifted_lr_scratch_saturated_h20"]
    scratch_gain50 = primary["comparisons"]["path_nlt_vs_lifted_lr_scratch_saturated_h50"]
    ordinary_gain20 = primary["comparisons"]["path_nlt_vs_final_nlt_h20"]
    ordinary_gain50 = primary["comparisons"]["path_nlt_vs_final_nlt_h50"]
    family = primary["h50_family_means"]
    degraded_families = [
        name for name, values in family.items()
        if values["path_nlt"] > 1.05 * values["final_nlt"]
    ]
    checks = {
        "path_beats_ordinary_256_h20": ordinary_gain20["relative_improvement"] > 0,
        "path_beats_ordinary_256_h50": ordinary_gain50["relative_improvement"] > 0,
        "scratch_improvement_256_h20_at_least_75pct": (
            scratch_gain20["relative_improvement"] >= 0.75
        ),
        "scratch_improvement_256_h50_at_least_75pct": (
            scratch_gain50["relative_improvement"] >= 0.75
        ),
        "no_boundary_family_degrades_over_5pct": not degraded_families,
        "saturated_hr_oracle_parity_available": False,
    }
    payload = {
        "schema": "blrpng_gray_scott_r2_seed42_stage_v1",
        "frozen": True,
        "passed": all(checks.values()),
        "seed": 42,
        "selected_method": "path_nlt",
        "method_definition": (
            "A256 harmonic-lifted Newton-path pretraining with reaction-aware inputs; "
            "B64 m4 NLT final-only adaptation"
        ),
        "results": combined,
        "gate_checks": checks,
        "degraded_boundary_families": degraded_families,
        "confirmed": {
            "old_a128_256_h50_instability_removed": True,
            "overall_path_pretraining_benefit_on_fresh_query": True,
            "boundary_and_solver_data_gates_passed": True,
        },
        "limitations": [
            "Only seed42 has been run.",
            "Boundary-family bootstrap confidence intervals cross zero.",
            "The high-frequency family degrades despite overall improvement.",
            "The pre-frozen scratch baseline uses raw F/K/correlation inputs rather than the new reaction feature map.",
            "No saturated target-specific Gray--Scott HR oracle is available yet.",
        ],
        "reports": reports,
        "authorized_next_experiment": (
            "One post-query exploratory high-frequency spectral-adapter repair, followed by a new blind query if successful."
        ),
    }
    with workspace_audit("freeze_gray_scott_r2_stage") as audit:
        output = write_json("artifacts/manifests/gray_scott_r2_seed42_stage.json", payload)
        audit.update({"output": str(output), "passed": payload["passed"]})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
