from __future__ import annotations

import csv
import json
from pathlib import Path

from blrpng.data.generate import workspace_audit
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    write_json,
)


SOURCES = (
    "artifacts/reports/allen_cahn_reconfirmation_seed42_query128_h20_m4.json",
    "artifacts/reports/allen_cahn_reconfirmation_seed42_query128_efficiency.json",
    "artifacts/reports/allen_cahn_reconfirmation_seed42_query256_h20_m4.json",
    "artifacts/reports/allen_cahn_reconfirmation_seed42_query256_efficiency.json",
    "artifacts/reports/allen_cahn_reconfirmation_seed42_target_cost_query256.json",
    "artifacts/manifests/allen_cahn_reconfirmation_query_256_data_gate.json",
    "artifacts/manifests/allen_cahn_reconfirmation_seed42_cost_query256_gate.json",
)


def _load(relative: str) -> tuple[Path, dict]:
    path = resolve_in_project(relative, must_exist=True)
    return path, json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    loaded = {relative: _load(relative) for relative in SOURCES}
    primary128 = loaded[SOURCES[0]][1]
    primary256 = loaded[SOURCES[2]][1]
    efficiency256 = loaded[SOURCES[3]][1]
    cost256 = loaded[SOURCES[4]][1]
    data_gate = loaded[SOURCES[5]][1]
    cost_gate = loaded[SOURCES[6]][1]

    means = primary256["method_mean_relative_l2"]
    comparisons = primary256["comparisons"]
    families = primary256["family_mean_relative_l2"]
    nlt = float(means["nlt"])
    ordinary = float(means["ordinary_nlt"])
    scratch = float(means["lr_scratch"])
    degrading_families = [
        family
        for family, values in families.items()
        if float(values["nlt"]) > 1.05 * float(values["ordinary_nlt"])
    ]
    target_variants = (
        "nlt_path",
        "nlt_commutator",
        "nlt_commutator_defect",
    )
    target_variant_cis_include_zero = {
        variant: (
            float(comparisons[f"nlt_vs_{variant}"]["ci95"][0]) <= 0.0
            <= float(comparisons[f"nlt_vs_{variant}"]["ci95"][1])
        )
        for variant in target_variants
    }
    checks = {
        "query256_data_gate_passed": bool(data_gate["passed"]),
        "query256_accuracy_gate_passed": bool(primary256["accuracy_gate"]["passed"]),
        "relative_improvement_over_lr_scratch_at_least_75pct": bool(
            float(comparisons["nlt_vs_lr_scratch"]["relative_improvement"]) >= 0.75
        ),
        "path_pretraining_improvement_ci_lower_positive": bool(
            float(comparisons["nlt_vs_ordinary_nlt"]["ci95"][0]) > 0.0
        ),
        "marginal_speedup_ci_lower_above_10x": bool(cost_gate["passed"]),
        "no_more_than_one_boundary_family_degrades_over_5pct": bool(
            len(degrading_families) <= 1
        ),
        "exact_boundary_trace": bool(
            primary256["accuracy_gate"]["boundary_error_below_1e_minus_6"]
        ),
    }
    passed = all(checks.values())
    source_records = {
        relative: sha256_file(path)
        for relative, (path, _payload) in loaded.items()
    }
    payload = {
        "schema": "blrpng_allen_cahn_reconfirmation_stage_v1",
        "as_of": "2026-08-31",
        "frozen": True,
        "seed": 42,
        "stage_passed": passed,
        "method": {
            "name": "BL-RPNG-NLT",
            "definition": (
                "harmonic boundary lifting + Newton-path privileged pretraining + "
                "target-specific NLT final-only adaptation"
            ),
            "target_adaptation_choice": "NLT final-only remains frozen primary",
            "target_variant_ci_includes_zero": target_variant_cis_include_zero,
        },
        "primary_256_h20": {
            "statistical_units": int(primary256["statistical_units"]),
            "nlt_mean_relative_l2": nlt,
            "ordinary_pretrain_nlt_mean_relative_l2": ordinary,
            "lifted_lr_scratch_mean_relative_l2": scratch,
            "nlt_vs_ordinary_relative_improvement": comparisons[
                "nlt_vs_ordinary_nlt"
            ],
            "nlt_vs_lr_scratch_relative_improvement": comparisons[
                "nlt_vs_lr_scratch"
            ],
            "task_wins_for_nlt": primary256["task_wins_for_nlt"],
            "family_mean_relative_l2": families,
        },
        "resolution_and_horizon": {
            "query128_h20_nlt_mean_relative_l2": float(
                primary128["method_mean_relative_l2"]["nlt"]
            ),
            "query256_h20_nlt_mean_relative_l2": nlt,
            "query256_to_query128_error_ratio": nlt
            / float(primary128["method_mean_relative_l2"]["nlt"]),
            "query256_h50_nlt_mean_relative_l2": float(
                efficiency256["mean_relative_l2"]["m4_h50"]
            ),
            "query256_h50_to_h20": efficiency256[
                "horizon50_to_horizon20_ratio"
            ],
        },
        "data_efficiency_256_h20": efficiency256["mean_relative_l2"],
        "target_cost": {
            "mean_lr_setup_seconds": sum(
                float(row["setup_seconds"]) for row in cost256["low_target_rows"]
            )
            / len(cost256["low_target_rows"]),
            "mean_hr_reference_setup_seconds": sum(
                float(row["setup_seconds"]) for row in cost256["high_reference_rows"]
            )
            / len(cost256["high_reference_rows"]),
            "setup_speedup": cost256["setup_speedup"],
            "q16_target_cost_speedup": cost256["q16_target_cost_speedup"],
            "shared_pretraining_training_seconds": cost256[
                "source_pretraining_training_seconds"
            ],
            "shared_a_generation_timing_status": (
                "legacy v3 A data did not record per-rollout solver timing; it is "
                "not silently assigned zero and must be instrumented prospectively"
            ),
        },
        "gate_checks": checks,
        "degrading_boundary_families_over_5pct": degrading_families,
        "claim_limits": {
            "multiseed": "seed42 confirmation only; seeds 0/1/2/3 remain pending",
            "oracle_parity": (
                "m256 HR reference is resource-capped and unsaturated; no saturated "
                "HR-oracle parity claim is made"
            ),
            "stage_transition": (
                "The user explicitly waived m512/m1024 oracle continuation; this gate "
                "supports proceeding to the Poisson negative control, not oracle parity"
            ),
        },
        "source_sha256": source_records,
    }
    with workspace_audit("freeze_reconfirmation_stage_20260831") as audit:
        table = ensure_artifact_path(
            "artifacts/reports/current_results_20260831.csv"
        )
        with table.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("scope", "group", "method_or_metric", "value"))
            for method, value in means.items():
                writer.writerow(("query256_h20", "all", method, value))
            for family, values in sorted(families.items()):
                for method in ("nlt", "ordinary_nlt", "lr_scratch"):
                    writer.writerow(("family", family, method, values[method]))
            for label, value in efficiency256["mean_relative_l2"].items():
                writer.writerow(("data_efficiency", "all", label, value))
            writer.writerow((
                "cost", "all", "setup_speedup",
                cost256["setup_speedup"]["point"],
            ))
            writer.writerow((
                "cost", "all", "q16_target_cost_speedup",
                cost256["q16_target_cost_speedup"]["point"],
            ))
        payload["result_table"] = str(table.relative_to(PROJECT_ROOT))
        payload["result_table_sha256"] = sha256_file(table)
        report = write_json(
            "artifacts/reports/allen_cahn_reconfirmation_seed42_stage_20260831.json",
            payload,
        )
        state = write_json("artifacts/reports/current_state_20260831.json", payload)
        gate = write_json(
            "artifacts/manifests/allen_cahn_reconfirmation_seed42_stage_gate.json",
            {
                "schema": "blrpng_stage_gate_v1",
                "passed": passed,
                "next_stage": "poisson_negative_control" if passed else None,
                "report": str(report.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(report),
                "current_state": str(state.relative_to(PROJECT_ROOT)),
                "current_state_sha256": sha256_file(state),
                "result_table": str(table.relative_to(PROJECT_ROOT)),
                "result_table_sha256": sha256_file(table),
                "saturated_oracle_claim": False,
            },
        )
        audit.update({
            "report": str(report), "state": str(state), "table": str(table),
            "gate": str(gate),
        })
    print(json.dumps({
        "report": str(report),
        "state": str(state),
        "table": str(table),
        "gate": str(gate),
        "stage_passed": passed,
        "gate_checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
