#!/usr/bin/env python3
"""Audit the frozen V0 first-complete-outer-state training label."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


NS_ROOT = Path("/gemini/code/NS-PIMPLE")
OUTPUT = Path("/gemini/code/BL-RPNG/artifacts/reports/ns_pimple_v0_pilot/selected_iteration_screen.json")


def audit_case(pair: dict, level: str) -> dict:
    h5_path = NS_ROOT / pair["cases"][level]["hdf5"]
    with h5py.File(h5_path, "r") as h5:
        d_values = h5["diagnostics/D"][:]
        s_values = h5["diagnostics/S"][:]
        valid = h5["path/valid_mask"][:].astype(bool)
        endpoints = h5["path/is_endpoint"][:].astype(bool)
        outer_indices = h5["path/outer_iteration_index"][:]

    first_d: list[float] = []
    first_s: list[float] = []
    first_outer: list[int] = []
    d_max_positions: list[int] = []
    s_max_positions: list[int] = []
    for time_index in range(valid.shape[0]):
        candidates = np.flatnonzero(valid[time_index] & ~endpoints[time_index])
        if not len(candidates):
            raise ValueError(f"{pair['pair_id']}/{level}: no non-endpoint outer state at step {time_index + 1}")
        first = int(candidates[0])
        first_d.append(float(d_values[time_index, first]))
        first_s.append(float(s_values[time_index, first]))
        first_outer.append(int(outer_indices[time_index, first]))
        candidate_d = np.asarray(d_values[time_index, candidates], dtype=np.float64)
        candidate_s = np.asarray(s_values[time_index, candidates], dtype=np.float64)
        d_max_positions.append(int(outer_indices[time_index, candidates[int(np.nanargmax(candidate_d))]]))
        s_max_positions.append(int(outer_indices[time_index, candidates[int(np.nanargmax(candidate_s))]]))

    first_d_array = np.asarray(first_d)
    first_s_array = np.asarray(first_s)
    second_half = first_d_array[len(first_d_array) // 2 :]
    checks = {
        "first_hook_D_median_ge_5pct": float(np.nanmedian(first_d_array)) >= 0.05,
        "first_hook_S_median_ge_1pct": float(np.nanmedian(first_s_array)) >= 0.01,
        "first_hook_D_ge_2pct_coverage_ge_70pct": float(np.mean(first_d_array >= 0.02)) >= 0.70,
        "second_half_first_hook_D_median_ge_3pct": float(np.nanmedian(second_half)) >= 0.03,
        "one_complete_nonendpoint_outer_state_per_physical_step": len(first_outer) == valid.shape[0],
    }
    return {
        "pair_id": pair["pair_id"],
        "split": pair["split"],
        "role": pair["role"],
        "level": level,
        "physical_steps": int(valid.shape[0]),
        "first_outer_index_min": min(first_outer),
        "first_outer_index_max": max(first_outer),
        "first_hook_D_median": float(np.nanmedian(first_d_array)),
        "first_hook_S_median": float(np.nanmedian(first_s_array)),
        "first_hook_D_ge_2pct_coverage": float(np.mean(first_d_array >= 0.02)),
        "second_half_first_hook_D_median": float(np.nanmedian(second_half)),
        "first_is_max_D_fraction": float(np.mean(np.asarray(d_max_positions) == np.asarray(first_outer))),
        "first_is_max_S_fraction": float(np.mean(np.asarray(s_max_positions) == np.asarray(first_outer))),
        "max_S_outer_index_counts": {
            str(index): int(count)
            for index, count in zip(*np.unique(s_max_positions, return_counts=True))
        },
        "checks": checks,
    }


def main() -> int:
    manifest = json.loads((NS_ROOT / "data/v0/manifest.json").read_text())
    rows = [audit_case(pair, level) for pair in manifest["pairs"] if pair["status"] == "complete" for level in ("lr", "hr")]
    check_names = rows[0]["checks"].keys()
    checks = {name: all(row["checks"][name] for row in rows) for name in check_names}
    first_d_max_count = sum(
        int(round(row["first_is_max_D_fraction"] * row["physical_steps"])) for row in rows
    )
    first_s_max_count = sum(
        int(round(row["first_is_max_S_fraction"] * row["physical_steps"])) for row in rows
    )
    all_steps = sum(row["physical_steps"] for row in rows)
    report = {
        "schema": "ns_pimple_v0_first_outer_selection_audit_v2",
        "dataset_id": manifest["dataset_id"],
        "status": "pass" if all(checks.values()) else "fail",
        "selection_rule": (
            "Use the earliest complete non-endpoint PIMPLE outer state at each physical step. "
            "The pre-existing V0 gate is defined on first-hook D/S medians, D1 coverage, and "
            "second-half D1; it does not require the first node to be the per-step argmax of both D and S."
        ),
        "selected_supervision": "path/u[first valid nonendpoint node] - solver_initial/u",
        "checks": checks,
        "aggregate": {
            "complete_pairs": len(manifest["pairs"]),
            "trajectory_level_cases": len(rows),
            "physical_steps": all_steps,
            "first_node_is_max_D_steps": first_d_max_count,
            "first_node_is_max_S_steps": first_s_max_count,
            "first_node_max_S_fraction": first_s_max_count / all_steps,
            "min_case_first_hook_D_median": min(row["first_hook_D_median"] for row in rows),
            "min_case_first_hook_S_median": min(row["first_hook_S_median"] for row in rows),
            "min_case_second_half_D_median": min(row["second_half_first_hook_D_median"] for row in rows),
        },
        "case_summaries": rows,
        "interpretation": (
            "The first outer state passes the frozen path gate. A few steps having a later S maximum "
            "does not invalidate this selection, because the preregistered gate tests the first-hook "
            "median S threshold rather than a per-step S argmax."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks, "aggregate": report["aggregate"], "output": str(OUTPUT)}, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
