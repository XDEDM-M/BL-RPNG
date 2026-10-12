from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blrpng.paths import resolve_in_project


def _training_record(path: str | Path) -> dict[str, Any]:
    report_path = resolve_in_project(path, must_exist=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    history = payload["history"]
    values = [float(row["validation"]["rollout_rel_l2"]) for row in history]
    best_index = min(range(len(values)), key=values.__getitem__)
    best_row = history[best_index]
    strata = best_row["validation"].get("strata", [])
    return {
        "best": values[best_index],
        "best_epoch": int(best_row["epoch"]),
        "final": values[-1],
        "final_over_best": values[-1] / max(values[best_index], 1e-15),
        "seen": float(strata[0]["rollout_rel_l2"]) if len(strata) > 1 else None,
        "unseen": float(strata[1]["rollout_rel_l2"]) if len(strata) > 1 else None,
    }


def _relative_improvement(baseline: float, method: float) -> float:
    return (baseline - method) / baseline


def formal_seed_summary(
    *, experiment_name: str, seed: int, budgets: list[int]
) -> dict[str, Any]:
    a = {
        objective: _training_record(
            f"artifacts/reports/{experiment_name}_a_{objective}_seed{seed}_training.json"
        )
        for objective in ("final_only", "blrpng")
    }
    cells: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    for budget in budgets:
        budget_cells = {}
        for initialization in ("scratch", "final_pretrained", "blrpng_pretrained"):
            for objective in ("final_only", "blrpng"):
                method = f"{initialization}_to_{objective}"
                budget_cells[method] = _training_record(
                    f"artifacts/reports/b{budget}_s{seed}_{method}_training.json"
                )
        cells[str(budget)] = budget_cells
        scratch = budget_cells["scratch_to_final_only"]["best"]
        ordinary = budget_cells["final_pretrained_to_final_only"]["best"]
        path_pre_final = budget_cells["blrpng_pretrained_to_final_only"]["best"]
        main = budget_cells["blrpng_pretrained_to_blrpng"]["best"]
        effects[str(budget)] = {
            "main_vs_scratch_final": _relative_improvement(scratch, main),
            "main_vs_ordinary_pretrain_final": _relative_improvement(ordinary, main),
            "path_pretraining_representation": _relative_improvement(
                ordinary, path_pre_final
            ),
            "path_finetuning_given_path_pretraining": _relative_improvement(
                path_pre_final, main
            ),
            "path_finetuning_given_final_pretraining": _relative_improvement(
                ordinary,
                budget_cells["final_pretrained_to_blrpng"]["best"],
            ),
            "path_finetuning_from_scratch": _relative_improvement(
                scratch, budget_cells["scratch_to_blrpng"]["best"]
            ),
        }
    budget4_main = cells["4"]["blrpng_pretrained_to_blrpng"]["best"]
    budget8_ordinary = cells["8"]["final_pretrained_to_final_only"]["best"]
    return {
        "experiment": experiment_name,
        "seed": seed,
        "a_pretraining": a,
        "a_path_pretraining_improvement": _relative_improvement(
            a["final_only"]["best"], a["blrpng"]["best"]
        ),
        "cells": cells,
        "effects": effects,
        "budget4_main_relative_to_budget8_ordinary": (
            budget4_main / budget8_ordinary - 1.0
        ),
        "raw_scratch_b4": _training_record(
            f"artifacts/reports/ablation_b4_s{seed}_raw_scratch_final_training.json"
        ),
    }
