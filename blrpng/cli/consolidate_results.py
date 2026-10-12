from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json


DATE = "20260820"


def _load(path: str) -> dict:
    resolved = resolve_in_project(path, must_exist=True)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _adapter_summary(method: str) -> dict:
    paths = sorted(resolve_in_project("artifacts/checkpoints/v2").glob(
        f"v2_task*_m4_s42_{method}.adapter.pt"
    ))
    if not paths:
        return {}
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    config = payloads[0]["adapter_config"]
    return {
        "tasks": len(paths),
        "strategy": config["strategy"],
        "trainable_parameters": int(config["trainable_parameters"]),
        "total_parameters": int(config["total_parameters"]),
        "trainable_fraction": float(config["trainable_parameters"] / config["total_parameters"]),
        "mean_adaptation_seconds": float(sum(
            item["metadata"]["elapsed_seconds"] for item in payloads
        ) / len(payloads)),
        "mean_delta_mebibytes": float(sum(path.stat().st_size for path in paths) / len(paths) / 2**20),
    }


def main() -> None:
    frozen_path = resolve_in_project(
        "artifacts/reports/v2_seed42_frozen_report.json", must_exist=True
    )
    frozen = _load(str(frozen_path))
    means = frozen["confirmatory"]["means_256_h20"]
    nlt = float(means["nlt"])
    lifted_scratch = float(means["lr_scratch"])
    raw_scratch = float(means["raw_lr_scratch"])
    blrpng_v2 = float(means["blrpng_v2"])
    ordinary = float(means["ordinary_final"])
    path_fixed = float(means["pathpre_fixed"])
    causal_path = resolve_in_project(
        "artifacts/reports/nlt_causal_seed42_summary_exploratory.json"
    )
    causal = _load(str(causal_path)) if causal_path.exists() else None
    benchmark_path = resolve_in_project("artifacts/reports/v2_solver_cost_benchmark.json")
    benchmark = _load(str(benchmark_path)) if benchmark_path.exists() else None

    state = {
        "schema": "blrpng_current_state_v1",
        "as_of": "2026-08-20",
        "primary_candidate": {
            "name": "BL-RPNG-NLT",
            "definition": (
                "harmonic boundary lifting + Newton-path privileged pretraining + "
                "target-specific NLT final-only adaptation"
            ),
            "status": "exploratory_until_fresh_blind_multiseed_confirmation",
        },
        "confirmed": {
            "exact_dirichlet_boundary_by_construction": True,
            "seed42_nlt_mean_relative_l2_256_h20": nlt,
            "nlt_improvement_over_lifted_lr_scratch": 1.0 - nlt / lifted_scratch,
            "nlt_improvement_over_raw_lr_scratch": 1.0 - nlt / raw_scratch,
            "nlt_improvement_over_blrpng_v2_lora": 1.0 - nlt / blrpng_v2,
            "path_pretraining_is_useful_with_target_final_finetuning": bool(
                means["pathpre_final"] < ordinary
            ),
        },
        "exploratory": {
            "nlt": True,
            "fine_defect_weight_sweep": True,
            "current_budget_hr_specialist_comparison": True,
            "nlt_vs_blrpng_v2_bootstrap": frozen["exploratory"]["comparisons"][
                "nlt_vs_blrpng_v2"
            ],
        },
        "not_eligible_for_paper_claims": {
            "saturated_hr_oracle_parity": {
                "reason": "m32-to-m64 validation improvements remain far above 1%",
                "old_gate_interpretation": "current_budget_hr_specialist_only",
            },
            "nlt_path_finetuning": {
                "reason": (
                    "NLT parameters were absent from shared path parameters, causing "
                    "path_weight=gradient_ratio=0"
                )
            },
        },
        "reference_means_256_h20": {
            "nlt_final_only": nlt,
            "blrpng_v2_lora": blrpng_v2,
            "ordinary_pretrain_final": ordinary,
            "pathpre_fixed_full": path_fixed,
            "lifted_lr_scratch": lifted_scratch,
            "raw_lr_scratch": raw_scratch,
        },
        "adapter_efficiency_seed42_m4": {
            "nlt": _adapter_summary("nlt"),
            "lora_final_only": _adapter_summary("lora"),
            "lora_blrpng_v2": _adapter_summary("blrpng_v2"),
        },
        "source_reports": {
            str(frozen_path.relative_to(resolve_in_project("."))): sha256_file(frozen_path),
            "artifacts/reports/v2_oracle_curves.json": sha256_file(
                resolve_in_project("artifacts/reports/v2_oracle_curves.json", must_exist=True)
            ),
        },
    }
    if causal is not None:
        state["mechanism_audit"] = {
            "status": causal["status"],
            "ordinary_pretrain_to_nlt": causal["means"]["ordinary_nlt"],
            "path_pretrain_to_nlt": causal["means"]["nlt"],
            "corrected_target_nlt_path": causal["means"]["nlt_path"],
            "path_pretraining_relative_improvement": (
                1.0 - causal["means"]["nlt"] / causal["means"]["ordinary_nlt"]
            ),
            "target_path_relative_improvement": (
                1.0 - causal["means"]["nlt_path"] / causal["means"]["nlt"]
            ),
            "target_path_was_active": causal["path_mechanism_audit"],
            "source_sha256": sha256_file(causal_path),
        }
    if benchmark is not None:
        low = float(benchmark["summary"]["lr64"]["mean_seconds"])
        high = float(benchmark["summary"]["hr256"]["mean_seconds"])
        state["solver_cost_microbenchmark"] = {
            "lr64_seconds_per_rollout": low,
            "hr256_seconds_per_rollout": high,
            "hr256_over_lr64_ratio": high / low,
            "lr_timing_cv": benchmark["summary"]["lr64"]["cv"],
            "hr_timing_cv": benchmark["summary"]["hr256"]["cv"],
            "status": "confirmed_solver_only_not_saturated_oracle_deployment_cost",
            "source_sha256": sha256_file(benchmark_path),
        }

    rows = [
        {"method": key, "mean_relative_l2_256_h20": value}
        for key, value in state["reference_means_256_h20"].items()
    ]
    with workspace_audit("consolidate_results"):
        state_path = write_json(f"artifacts/reports/current_state_{DATE}.json", state)
        supersession = {
            "schema": "blrpng_claim_supersession_v1",
            "created": "2026-08-20",
            "supersedes_claim_interpretation_of": (
                "artifacts/manifests/allen_cahn_v2_seed42_gate.json"
            ),
            "preserves_original_artifact": True,
            "old_gate_valid_for": "current-budget m64/20-epoch HR scratch specialist",
            "old_gate_not_valid_for": "converged or saturated HR-full oracle",
            "replacement_state": str(state_path.relative_to(resolve_in_project("."))),
            "required_reopening_condition": (
                "optimization plateau plus two consecutive data-budget doublings below 1%"
            ),
        }
        write_json(
            f"artifacts/manifests/allen_cahn_v2_gate_supersession_{DATE}.json",
            supersession,
        )
        csv_path = ensure_artifact_path(f"artifacts/reports/current_results_{DATE}.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        markdown = [
            "# BL-RPNG consolidated results (2026-08-20)",
            "",
            "| Method | 256²/20-step mean relative L2 | Status |",
            "|---|---:|---|",
            f"| BL-RPNG-NLT | {nlt:.6f} | exploratory best |",
            f"| BL-RPNG v2 LoRA | {blrpng_v2:.6f} | frozen confirmatory |",
            f"| ordinary pretrain → final | {ordinary:.6f} | baseline |",
            f"| path-pretrain → fixed path full FT | {path_fixed:.6f} | baseline |",
            f"| lifted LR scratch | {lifted_scratch:.6f} | baseline |",
            f"| raw LR scratch | {raw_scratch:.6f} | baseline |",
            "",
            "The prior HR gate is a current-budget comparison, not a saturated-oracle result.",
            "The previous NLT+path runs had zero effective path weight and are invalid for",
            "assessing target-stage Newton-path supervision.",
            "",
        ]
        ensure_artifact_path(
            f"artifacts/reports/current_results_{DATE}.md"
        ).write_text("\n".join(markdown), encoding="utf-8")

    print(json.dumps({"state": str(state_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
