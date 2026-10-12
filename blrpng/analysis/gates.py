from __future__ import annotations

from typing import Any


def development_gate(
    *,
    data_reports: list[dict[str, Any]],
    gradient_ratios: list[float],
    seen_blrpng: float,
    seen_pretrained: float,
    unseen_blrpng: float,
    unseen_pretrained: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    checks = {
        "all_nonlinear_solves_converged": all(
            report["solver_converged_fraction"] == 1.0 for report in data_reports
        ),
        "valid_correction_fraction": all(
            report["valid_correction_fraction"]
            >= float(gate["valid_correction_fraction_min"])
            for report in data_reports
        ),
        "boundary_and_lifting_error": all(
            max(
                report["boundary_error_max"],
                report["lifting_boundary_error_max"],
                report["correction_boundary_error_max"],
            )
            <= float(gate["boundary_error_max"])
            for report in data_reports
        ),
        "gradient_ratio": all(
            abs(value - 0.15) <= float(gate["gradient_ratio_tolerance"])
            for value in gradient_ratios
        ),
        "seen_non_degradation": seen_blrpng
        <= (1.0 + float(gate["max_relative_degradation"])) * seen_pretrained,
        "unseen_non_degradation": unseen_blrpng
        <= (1.0 + float(gate["max_relative_degradation"])) * unseen_pretrained,
    }
    return {"passed": all(checks.values()), "checks": checks}


def paper_claims(
    *,
    main_vs_scratch: dict[str, Any],
    path_pretrain_vs_final_pretrain: dict[str, Any],
    path_finetune_vs_final: dict[str, Any],
    main_four_boundary: float,
    ordinary_eight_boundary: float,
    frequency_ood_change: float,
    amplitude_ood_change: float,
) -> dict[str, Any]:
    claims = {
        "main_vs_scratch": main_vs_scratch["num_seed_wins"] >= 4
        and main_vs_scratch["ci95"][0] > 0,
        "path_pretraining_effect": path_pretrain_vs_final_pretrain["ci95"][0] > 0,
        "path_finetuning_effect": path_finetune_vs_final["ci95"][0] > 0,
        "data_efficiency": main_four_boundary <= 1.05 * ordinary_eight_boundary,
        "ood_robustness": not (
            frequency_ood_change > 0.05 and amplitude_ood_change > 0.05
        ),
    }
    return {"passed": all(claims.values()), "claims": claims}
