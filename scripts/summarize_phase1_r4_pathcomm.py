from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.paths import PROJECT_ROOT, sha256_file
from scripts.summarize_phase1_hrdev import (
    FINALIST_MODEL_SEEDS,
    FINALIST_SUPPORT_BUDGET,
    FINALIST_SUPPORT_SEED,
    TASK_FAMILIES,
    summarize,
    _write_json_durable,
)


ARMS = {
    "A0_pathsource_comm256": "pilot_pathsource_comm256",
    "A1_firstpath_comm256": "pathsource_firstpath_ratio005_comm256-r4full",
    "A2_endpoint_comm256": "pathsource_endpoint_ratio005_comm256-r4full",
}
SEED_TRIAL_ALIASES = {
    (ARMS["A0_pathsource_comm256"], 1): "pilot_pathsource_comm256-recovery1",
}
T_CRIT_95_DF2 = 4.302652729911275
PRACTICAL_RELATIVE_IMPROVEMENT = 0.01


def _paired_summary(reference: dict, candidate: dict) -> dict:
    ref_cells = {
        (int(row["seed"]), int(row["task"])): row
        for row in reference["cells"]
    }
    cand_cells = {
        (int(row["seed"]), int(row["task"])): row
        for row in candidate["cells"]
    }
    improvements = {}
    by_seed = {}
    by_family = {}
    task_rows = []
    for seed in FINALIST_MODEL_SEEDS:
        seed_diffs = []
        for task, family in TASK_FAMILIES.items():
            ref = ref_cells[(seed, task)]
            cand = cand_cells[(seed, task)]
            ref_provenance = _pairing_provenance(ref)
            cand_provenance = _pairing_provenance(cand)
            if ref_provenance != cand_provenance:
                raise ValueError(
                    f"R4 arms are not paired for seed={seed}, task={task}: "
                    f"{reference['trial_id']} vs {candidate['trial_id']}"
                )
            diff = float(ref["rollout_rel_l2"]) - float(cand["rollout_rel_l2"])
            seed_diffs.append(diff)
            improvements[(seed, task)] = diff
            task_rows.append({
                "seed": seed,
                "task": task,
                "family": family,
                "reference_hr_dev": float(ref["rollout_rel_l2"]),
                "candidate_hr_dev": float(cand["rollout_rel_l2"]),
                "improvement_reference_minus_candidate": diff,
                "relative_improvement": diff / float(ref["rollout_rel_l2"]),
                "reference_report": ref["report"],
                "reference_report_sha256": ref["report_sha256"],
                "candidate_report": cand["report"],
                "candidate_report_sha256": cand["report_sha256"],
                "paired_provenance": cand_provenance,
            })
        by_seed[str(seed)] = statistics.mean(seed_diffs)
    for task, family in TASK_FAMILIES.items():
        by_family[family] = statistics.mean(
            improvements[(seed, task)] for seed in FINALIST_MODEL_SEEDS
        )

    seed_values = list(by_seed.values())
    mean_improvement = statistics.mean(seed_values)
    half_width = T_CRIT_95_DF2 * statistics.stdev(seed_values) / (len(seed_values) ** 0.5)
    reference_mean = float(reference["mean_hr_dev_rollout_rel_l2"])
    candidate_mean = float(candidate["mean_hr_dev_rollout_rel_l2"])
    relative_improvement = (reference_mean - candidate_mean) / reference_mean
    return {
        "reference_trial_id": reference["trial_id"],
        "candidate_trial_id": candidate["trial_id"],
        "n_paired_cells": len(task_rows),
        "reference_macro_mean": reference_mean,
        "candidate_macro_mean": candidate_mean,
        "absolute_macro_improvement": reference_mean - candidate_mean,
        "relative_macro_improvement": relative_improvement,
        "paired_seed_macro_improvement": by_seed,
        "paired_seed_macro_improvement_mean": mean_improvement,
        "paired_seed_macro_improvement_95pct_t_interval_df2": [
            mean_improvement - half_width,
            mean_improvement + half_width,
        ],
        "paired_family_mean_improvement": by_family,
        "seed_wins": sum(value > 0 for value in seed_values),
        "family_wins": sum(value > 0 for value in by_family.values()),
        "task_seed_cells": task_rows,
    }


def _pairing_provenance(cell: dict) -> dict:
    report_path = PROJECT_ROOT / str(cell["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result_path = PROJECT_ROOT / str(report["task_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "support_index": result["support_index"],
        "support_index_sha256": result["support_index_sha256"],
        "lr_validation_index": result["validation_index"],
        "lr_validation_index_sha256": result["validation_index_sha256"],
        "hr_validation_index": report["hr_validation_index"],
        "hr_validation_index_sha256": report["hr_validation_index_sha256"],
        "source_checkpoint": result["source_checkpoint"],
        "source_checkpoint_sha256": result["source_checkpoint_sha256"],
    }


def _write_markdown(payload: dict, path: Path) -> None:
    ranked = payload["ranked_candidates"]
    candidates = {row["trial_id"]: row for row in ranked}
    a0 = candidates[ARMS["A0_pathsource_comm256"]]
    a1 = candidates[ARMS["A1_firstpath_comm256"]]
    a2 = candidates[ARMS["A2_endpoint_comm256"]]
    comparisons = payload["comparisons"]
    lines = [
        "# R4: target path supervision with commutator calibration",
        "",
        "Development-only paired analysis of the frozen R4 matrix. All LR checkpoints were selected on LR validation before these HR-dev evaluations. No confirmation query was read.",
        "",
        "## Overall HR-dev results",
        "",
        "| Arm | Mean rollout relative L2 | Worst family mean | Seed means (0 / 1 / 42) |",
        "|---|---:|---:|---|",
    ]
    for key, row in (("A0", a0), ("A1", a1), ("A2", a2)):
        seed_means = row["mean_by_model_seed"]
        lines.append(
            f"| {key} `{row['trial_id']}` | {row['mean_hr_dev_rollout_rel_l2']:.8f} | "
            f"{row['worst_task_mean_hr_dev_rollout_rel_l2']:.8f} | "
            f"{seed_means['0']:.8f} / {seed_means['1']:.8f} / {seed_means['42']:.8f} |"
        )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "Positive improvement means the candidate in the comparison label has lower error than its reference. Confidence intervals are paired t intervals over the three seed-level six-family means (df=2); they are descriptive and are not confirmation evidence.",
        "",
        "| Comparison | Absolute mean improvement | Relative improvement | Seed wins | Family wins | 95% paired t interval |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("A1_vs_A0", "A1_vs_A2", "A2_vs_A0"):
        row = comparisons[name]
        low, high = row["paired_seed_macro_improvement_95pct_t_interval_df2"]
        lines.append(
            f"| {name} | {row['absolute_macro_improvement']:.8f} | "
            f"{100*row['relative_macro_improvement']:.3f}% | "
            f"{row['seed_wins']}/3 | {row['family_wins']}/6 | [{low:.8f}, {high:.8f}] |"
        )
    lines.extend([
        "",
        f"A1 meets the pre-evaluation repeatable/practical development rule: **{payload['repeatable_practically_meaningful_A1_signal']}**. The rule requires at least 1% lower overall error than both comparators, all three seed means to favor A1, and at least four of six family means to favor A1 against each comparator.",
        "",
        "## Boundary-family means",
        "",
        "| Family | A0 | A1 | A2 | A1 improvement vs A0 | A1 improvement vs A2 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    a1_a0_fam = comparisons["A1_vs_A0"]["paired_family_mean_improvement"]
    a1_a2_fam = comparisons["A1_vs_A2"]["paired_family_mean_improvement"]
    for task, family in TASK_FAMILIES.items():
        lines.append(
            f"| {family} | {a0['mean_by_boundary_family'][family]:.8f} | "
            f"{a1['mean_by_boundary_family'][family]:.8f} | "
            f"{a2['mean_by_boundary_family'][family]:.8f} | "
            f"{a1_a0_fam[family]:.8f} | {a1_a2_fam[family]:.8f} |"
        )
    cell_maps = {
        key: {(int(cell["seed"]), int(cell["task"])): cell["rollout_rel_l2"] for cell in row["cells"]}
        for key, row in (("A0", a0), ("A1", a1), ("A2", a2))
    }
    lines.extend([
        "",
        "## Every paired task×seed result",
        "",
        "| Seed | Task | Family | A0 | A1 | A2 | A1 improvement vs A0 | A1 improvement vs A2 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for seed in FINALIST_MODEL_SEEDS:
        for task, family in TASK_FAMILIES.items():
            key = (seed, task)
            v0, v1, v2 = cell_maps["A0"][key], cell_maps["A1"][key], cell_maps["A2"][key]
            lines.append(
                f"| {seed} | {task} | {family} | {v0:.8f} | {v1:.8f} | {v2:.8f} | "
                f"{v0-v1:.8f} | {v2-v1:.8f} |"
            )
    lines.extend([
        "",
        "## Evidence and limits",
        "",
        "- Machine-readable paired summary: `artifacts/reports/phase1_r4_pathcomm_paired_summary_20261016.json`.",
        "- `scripts/summarize_phase1_hrdev.py` verifies each report task result, checkpoint, support/validation index seals, HR-dev manifest, and query flags. This R4 pair analysis additionally requires matching support/LR-validation/HR-validation index paths and seals plus recorded source-checkpoint paths and SHA values for each paired seed×task; it records those seals per cell.",
        "- Raw R4 task reports: `results/phase1/R4/`; HR-dev outputs: `artifacts/reports/phase1_hrdev_evaluations/`.",
        "- These six development families share one support draw. The three model seeds are repeated measurements, not 18 independent tasks. Do not generalize this comparison to the confirmation population.",
        "- Query access: `false`; HR-dev did not influence training or within-task checkpoint selection.",
        "",
    ])
    encoded = "\n".join(lines)
    if path.exists():
        raise FileExistsError(f"Preserve existing R4 markdown report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    if path.read_text(encoding="utf-8") != encoded:
        raise OSError(f"R4 markdown report failed read-back verification: {path}")


def main() -> None:
    base = summarize(
        trial_ids=list(ARMS.values()),
        seeds=list(FINALIST_MODEL_SEEDS),
        support_seed=FINALIST_SUPPORT_SEED,
        support_budget=FINALIST_SUPPORT_BUDGET,
        reports_root=PROJECT_ROOT / "artifacts/reports/phase1_hrdev_evaluations",
        seed_trial_aliases=SEED_TRIAL_ALIASES,
    )
    candidates = {row["trial_id"]: row for row in base["ranked_candidates"]}
    a0 = candidates[ARMS["A0_pathsource_comm256"]]
    a1 = candidates[ARMS["A1_firstpath_comm256"]]
    a2 = candidates[ARMS["A2_endpoint_comm256"]]
    comparisons = {
        "A1_vs_A0": _paired_summary(a0, a1),
        "A1_vs_A2": _paired_summary(a2, a1),
        "A2_vs_A0": _paired_summary(a0, a2),
    }
    a1_vs_a0 = comparisons["A1_vs_A0"]
    a1_vs_a2 = comparisons["A1_vs_A2"]
    repeatable_practical_signal = all(
        comparison["relative_macro_improvement"] >= PRACTICAL_RELATIVE_IMPROVEMENT
        and comparison["seed_wins"] == len(FINALIST_MODEL_SEEDS)
        and comparison["family_wins"] >= 4
        for comparison in (a1_vs_a0, a1_vs_a2)
    )
    payload = {
        "schema": "blrpng_phase1_r4_pathcomm_paired_summary_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "analysis_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "input_summary_schema": base["schema"],
        "model_seeds": list(FINALIST_MODEL_SEEDS),
        "support_seed": FINALIST_SUPPORT_SEED,
        "support_budget": FINALIST_SUPPORT_BUDGET,
        "seed_trial_aliases": {
            f"{candidate}:{seed}": effective_id
            for (candidate, seed), effective_id in SEED_TRIAL_ALIASES.items()
        },
        "query_read": False,
        "within_task_checkpoint_selection_used_hr": False,
        "decision_rule": {
            "source": "docs/experiment_cards/R4_PATHCOMM_INTERACTION.md",
            "relative_mean_improvement_threshold": PRACTICAL_RELATIVE_IMPROVEMENT,
            "all_seed_means_must_favor_a1": True,
            "minimum_family_wins_out_of_six": 4,
            "paired_interval_unit": "three model-seed macro means, t interval with df=2",
        },
        "repeatable_practically_meaningful_A1_signal": repeatable_practical_signal,
        "ranked_candidates": base["ranked_candidates"],
        "comparisons": comparisons,
    }
    output = PROJECT_ROOT / "artifacts/reports/phase1_r4_pathcomm_paired_summary_20261016.json"
    markdown_path = PROJECT_ROOT / "artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md"
    if output.exists():
        raise FileExistsError(f"Preserve existing R4 analysis: {output}")
    if markdown_path.exists():
        raise FileExistsError(f"Preserve existing R4 markdown report: {markdown_path}")
    _write_json_durable(output, payload)
    _write_markdown(payload, markdown_path)
    print(json.dumps({
        "output": str(output),
        "markdown_output": str(markdown_path),
        "ranked": [
            {
                "trial_id": row["trial_id"],
                "mean_hr_dev_rollout_rel_l2": row["mean_hr_dev_rollout_rel_l2"],
                "worst_task_mean_hr_dev_rollout_rel_l2": row["worst_task_mean_hr_dev_rollout_rel_l2"],
            }
            for row in base["ranked_candidates"]
        ],
        "A1_repeatable_practical_signal": repeatable_practical_signal,
        "query_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
