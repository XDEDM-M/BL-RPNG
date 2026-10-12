from __future__ import annotations

import json
from collections import defaultdict

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    report_paths = {
        128: resolve_in_project("artifacts/reports/gray_scott/confirmation128.json", must_exist=True),
        256: resolve_in_project("artifacts/reports/gray_scott/confirmation256.json", must_exist=True),
    }
    gate_paths = {
        128: resolve_in_project(
            "artifacts/manifests/gray_scott_seed42_confirmation128_evaluation_gate.json",
            must_exist=True,
        ),
        256: resolve_in_project(
            "artifacts/manifests/gray_scott_seed42_confirmation256_evaluation_gate.json",
            must_exist=True,
        ),
    }
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in report_paths.items()}
    gates = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in gate_paths.items()}
    if any(report.get("selection_used_confirmation_query") is not False for report in reports.values()):
        raise PermissionError("Confirmation query was used for checkpoint selection.")

    methods = ("ordinary_nlt", "safe_path_nlt_gated", "lifted_scratch")
    means: dict[int, dict[str, float]] = {}
    for resolution, report in reports.items():
        means[resolution] = {
            f"{method}_h{horizon}": _mean([
                float(row["relative_l2"]) for row in report["records"]
                if row["method"] == method and int(row["horizon"]) == horizon
            ])
            for method in methods for horizon in (20, 50)
        }

    amplification = {
        f"{method}_h{horizon}":
            means[256][f"{method}_h{horizon}"] / means[128][f"{method}_h{horizon}"]
        for method in methods for horizon in (20, 50)
    }
    family_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in reports[256]["records"]:
        if int(row["horizon"]) == 50:
            family_values[str(row["boundary_regime"])][str(row["method"])].append(
                float(row["relative_l2"])
            )
    family_h50 = {
        family: {method: _mean(values) for method, values in by_method.items()}
        for family, by_method in family_values.items()
    }
    main_vs_ordinary_family_h50 = {
        family: 1.0 - values["safe_path_nlt_gated"] / values["ordinary_nlt"]
        for family, values in family_h50.items()
    }
    c256 = gates[256]["comparisons"]
    payload = {
        "schema": "blrpng_gray_scott_256_failure_diagnosis_v1",
        "frozen": True,
        "seed": 42,
        "stage_passed": False,
        "main_method": "safe_path_nlt_gated",
        "query_checkpoint_selection_used": False,
        "means": {str(key): value for key, value in means.items()},
        "resolution_amplification_256_over_128": amplification,
        "main_vs_ordinary": {
            "h20": c256["safe_path_nlt_gated_vs_ordinary_nlt_h20"],
            "h50": c256["safe_path_nlt_gated_vs_ordinary_nlt_h50"],
        },
        "main_vs_scratch": {
            "h20": c256["safe_path_nlt_gated_vs_lifted_scratch_h20"],
            "h50": c256["safe_path_nlt_gated_vs_lifted_scratch_h50"],
        },
        "family_h50_means": family_h50,
        "main_vs_ordinary_family_h50_relative_improvement": main_vs_ordinary_family_h50,
        "diagnosis": {
            "short_horizon": (
                "The target-gated path remains beneficial at 256-grid h20, but the gain "
                "shrinks relative to 128-grid."
            ),
            "long_horizon": (
                "The candidate is unstable under 256-grid autoregression; amplitude OOD "
                "is the largest failure mode."
            ),
            "mechanistic_hypothesis": (
                "The target commutator was calibrated only on 64-to-128 predictions. Its "
                "one-step constraint does not control 128-to-256 mismatch or the multi-step "
                "Jacobian of the adapted operator."
            ),
        },
        "authorized_next_experiment": (
            "Use only frozen 64-grid support/validation and unlabeled 128/256 evaluations "
            "to develop a multiscale commutator plus multi-step stability repair. Freeze "
            "the repair before generating a new independent confirmation cohort."
        ),
        "prohibited": [
            "Selecting checkpoints or hyperparameters on either opened confirmation query",
            "Claiming the Gray--Scott stage gate passed",
            "Starting the next nonlinear PDE formal stage",
        ],
        "inputs": {
            f"report_{resolution}": {"path": str(path), "sha256": sha256_file(path)}
            for resolution, path in report_paths.items()
        } | {
            f"gate_{resolution}": {"path": str(path), "sha256": sha256_file(path)}
            for resolution, path in gate_paths.items()
        },
    }
    with workspace_audit("freeze_gray_scott_256_diagnosis") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_confirmation256_failure_diagnosis.json",
            payload,
        )
        audit.update({"output": str(output), "stage_passed": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
