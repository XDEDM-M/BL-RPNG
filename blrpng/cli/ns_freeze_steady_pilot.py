"""Freeze the two-boundary, three-resolution steady NS numerical pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from blrpng.data.shards import validate_shard, write_index
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


ROOT = "artifacts/data/navier_stokes_sfepy_steady_projected"
TASKS = (0, 6)
RESOLUTIONS = (64, 128, 256)


def _load(task: int, resolution: int, variant: str) -> tuple[dict, dict]:
    stem = "warmstart" if variant == "natural" else "re400"
    path = resolve_in_project(
        f"{ROOT}/task{task:02d}_{stem}_{resolution}.pt", must_exist=True
    )
    shard = torch.load(path, map_location="cpu", weights_only=False)
    validation = validate_shard(shard)
    metadata = shard["meta"]
    if metadata.get("trajectory_axis") != "reynolds_continuation_not_physical_time":
        raise ValueError(f"Unexpected trajectory semantics: {path}")
    if metadata.get("path_selection_mode") != "first_accepted":
        raise ValueError(f"Path target is not the first accepted Newton update: {path}")
    if "grid_projection" not in metadata:
        raise ValueError(f"Expected an audited projected grid shard: {path}")
    expected_blend = 1.0 if variant == "natural" else 0.5
    if float(metadata.get("newton_predictor_blend", -1)) != expected_blend:
        raise ValueError(f"Unexpected predictor blend for {variant}: {path}")
    ratios = np.asarray(metadata["selected_velocity_predictor_endpoint_discrepancy"])
    endpoint_norm = shard["predictor_endpoint_increment"][:, :2].flatten(1).norm(dim=1)
    state_norm = shard["trajectory"][1:, :2].flatten(1).norm(dim=1).clamp_min(1e-30)
    endpoint_state_fraction = (endpoint_norm / state_norm).tolist()
    row = {
        "path": str(path),
        "sha256": sha256_file(path),
        "resolution": resolution,
        "steps": validation["steps"],
        "boundary_id": shard["ids"]["boundary_id"],
        "initial_condition_id": shard["ids"]["initial_condition_id"],
        "target_task_id": shard["ids"]["target_task_id"],
        "pair_id": shard["ids"]["pair_id"],
        "reynolds_schedule": metadata["reynolds_schedule"],
        "solver_converged": metadata["solver_converged"],
        "valid_correction_fraction": float(shard["valid_correction_mask"].float().mean()),
        "max_relative_residual": max(metadata["relative_residuals"]),
        "boundary_error": validation["trace_error_max"],
        "pressure_gauge_error": metadata["pressure_gauge_error"],
        "max_native_fe_qp_divergence_rms": max(metadata["fe_divergence_qp_rms"]),
        "max_projected_grid_divergence_rms": max(metadata["grid_divergence_rms"]),
        "max_projection_relative_velocity_change": max(
            metadata["grid_projection"]["max_state_relative_velocity_change"],
            metadata["grid_projection"]["max_correction_relative_velocity_change"],
        ),
        "velocity_path_discrepancy_by_reynolds": ratios.tolist(),
        "velocity_path_discrepancy_median": float(np.median(ratios)),
        "velocity_path_discrepancy_p25": float(np.quantile(ratios, 0.25)),
        "predictor_endpoint_velocity_fraction_of_state_by_reynolds": endpoint_state_fraction,
        "median_newton_iterations": float(np.median(metadata["nonlinear_iterations"])),
        "peak_rss_gib": metadata["peak_rss_gib"],
        "lift_seconds": metadata["lift_time"],
        "solve_seconds": sum(metadata["solver_timing"]),
    }
    return shard, row


def _compare(coarse: dict, fine: dict) -> list[float]:
    fine_resolution = fine["trajectory"].shape[-1]
    upsampled = functional.interpolate(
        coarse["trajectory"][:, :2],
        size=(fine_resolution, fine_resolution),
        mode="bilinear",
        align_corners=True,
    ).double()
    reference = fine["trajectory"][:, :2].double()
    return [
        float((upsampled[step] - reference[step]).norm() / reference[step].norm().clamp_min(1e-30))
        for step in range(1, reference.shape[0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("natural", "blended"), default="natural")
    args = parser.parse_args()
    records = []
    comparisons = []
    for task in TASKS:
        shards = {}
        task_rows = []
        for resolution in RESOLUTIONS:
            shard, row = _load(task, resolution, args.variant)
            shards[resolution] = shard
            task_rows.append(row)
            records.append({"task": task, **row})
        if len({row["pair_id"] for row in task_rows}) != 1:
            raise ValueError(f"Coarse/fine pair mismatch for task {task}")
        if len({row["boundary_id"] for row in task_rows}) != 1:
            raise ValueError(f"Boundary mismatch for task {task}")
        if any(row["reynolds_schedule"] != [50.0, 100.0, 200.0, 400.0] for row in task_rows):
            raise ValueError(f"Unexpected Reynolds continuation schedule for task {task}")
        comparisons.append({
            "task": task,
            "velocity_rel_l2_64_to_128": _compare(shards[64], shards[128]),
            "velocity_rel_l2_128_to_256": _compare(shards[128], shards[256]),
        })
    numerical_gate = bool(
        all(
            row["solver_converged"]
            and row["valid_correction_fraction"] == 1.0
            and row["max_relative_residual"] <= 1e-8
            and row["boundary_error"] <= 1e-6
            and row["pressure_gauge_error"] <= 1e-7
            and row["max_projected_grid_divergence_rms"] <= 1e-4
            and row["max_projection_relative_velocity_change"] <= 0.01
            and row["velocity_path_discrepancy_median"] >= 0.05
            and min(row["predictor_endpoint_velocity_fraction_of_state_by_reynolds"]) >= 0.05
            and row["median_newton_iterations"] >= 3
            and (row["resolution"] == 64 or row["max_native_fe_qp_divergence_rms"] <= 1e-3)
            for row in records
        )
        and all(max(row["velocity_rel_l2_128_to_256"]) <= 0.01 for row in comparisons)
    )
    indexes = {}
    for resolution in RESOLUTIONS:
        rows = [
            {
                "path": str(Path(row["path"]).relative_to(PROJECT_ROOT)),
                "sha256": row["sha256"],
                "steps": row["steps"],
                "target_task_id": row["target_task_id"],
                "boundary_id": row["boundary_id"],
                "initial_condition_id": row["initial_condition_id"],
                "role": "numerical_pilot_only",
                "resolution": resolution,
            }
            for row in records if row["resolution"] == resolution
        ]
        index_name = (
            f"navier_stokes_sfepy_steady_natural_{resolution}.index.json"
            if args.variant == "natural" else
            f"navier_stokes_sfepy_steady_pilot_{resolution}.index.json"
        )
        index_path = write_index(
            f"artifacts/manifests/{index_name}",
            rows,
            metadata={
                "equation": "steady_navier_stokes_continuation",
                "trajectory_axis": "Reynolds_parameter_not_physical_time",
                "intended_use": "integration_smoke_and_numerical_pilot_only",
                "predictor_blend": 1.0 if args.variant == "natural" else 0.5,
            },
        )
        indexes[str(resolution)] = {"path": str(index_path), "sha256": sha256_file(index_path)}
    report = {
        "schema": "blrpng_steady_ns_numerical_pilot_v1",
        "equation": "steady_2d_incompressible_navier_stokes",
        "trajectory_axis": "Reynolds_parameter_continuation_not_physical_time",
        "solver": "SfePy_P2_P1_grad_div_Newton_line_search",
        "supervised_path": (
            "first_accepted_Newton_correction_from_previous_solution_warmstart"
            if args.variant == "natural" else
            "first_accepted_Newton_correction_from_blended_Stokes_predictor"
        ),
        "matched_endpoint_control": "predictor_endpoint_increment",
        "predictor_variant": args.variant,
        "predictor_blend": 1.0 if args.variant == "natural" else 0.5,
        "boundaries": len(TASKS),
        "records": records,
        "indexes": indexes,
        "spatial_refinement": comparisons,
        "numerical_pilot_gate_passed": numerical_gate,
        "model_effect_not_yet_tested": True,
        "transient_NS_supersession": (
            "Prior BDF3 progress-0.25 NS shards failed the actual supervised-path "
            "nonredundancy gate; they are not part of this pilot."
        ),
    }
    report_name = "natural_warmstart_gate.json" if args.variant == "natural" else "numerical_gate.json"
    output = write_json(
        f"artifacts/reports/navier_stokes_sfepy_steady_pilot/{report_name}",
        report,
    )
    print(json.dumps({
        "report": str(output),
        "sha256": sha256_file(output),
        "numerical_pilot_gate_passed": numerical_gate,
        "boundaries": len(TASKS),
        "rollout_shards": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
