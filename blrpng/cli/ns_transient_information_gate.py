"""Gate natural transient NS Newton information before any formal dataset.

The primary sweep changes only ``dt``.  Every solve starts from the previous
physical state (no Stokes blending or perturbation of the Newton predictor).
Each candidate is paired with a dt/2 trajectory at the same physical horizon.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import build_navier_stokes_protocol
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.paths import resolve_in_project, sha256_file, write_json


def _relative(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-30))


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    spacing = 1.0 / float(field.shape[-1] - 1)
    return torch.gradient(field[1], spacing=(spacing,), dim=(0,))[0] - torch.gradient(
        field[0], spacing=(spacing,), dim=(1,)
    )[0]


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _first_field_metrics(shard: dict, field: slice) -> dict[str, dict[str, float]]:
    correction = shard["newton_correction"][:, field].double().flatten(1)
    endpoint = shard["predictor_endpoint_increment"][:, field].double().flatten(1)
    state = shard["trajectory"][1:, field].double().flatten(1)
    endpoint_norm = endpoint.norm(dim=1).clamp_min(1e-30)
    correction_norm = correction.norm(dim=1)
    dot = (correction * endpoint).sum(dim=1)
    cosine = dot / (correction_norm * endpoint_norm).clamp_min(1e-30)
    coefficient = dot / endpoint_norm.square()
    orthogonal = (correction - coefficient[:, None] * endpoint).norm(dim=1) / endpoint_norm
    return {
        "endpoint_discrepancy": _quantiles(
            ((correction - endpoint).norm(dim=1) / endpoint_norm).tolist()
        ),
        "endpoint_cosine": _quantiles(cosine.clamp(-1.0, 1.0).tolist()),
        "orthogonal_fraction": _quantiles(orthogonal.tolist()),
        "norm_ratio": _quantiles((correction_norm / endpoint_norm).tolist()),
        "endpoint_fraction_of_state": _quantiles(
            (endpoint_norm / state.norm(dim=1).clamp_min(1e-30)).tolist()
        ),
    }


def _case_metrics(shard: dict, half: dict) -> dict:
    metadata = shard["meta"]
    paths = metadata["accepted_newton_path_metrics"]
    first = [path["updates"][0] for path in paths]
    turns = [value for path in paths for value in path["turn_cosines"]]
    final = shard["trajectory"][-1]
    reference = half["trajectory"][-1]
    result = {
        "solver_converged": bool(metadata["solver_converged"]),
        "steps": int(metadata["steps"]),
        "maximum_relative_residual": max(metadata["relative_residuals"]),
        "newton_iterations": _quantiles([float(v) for v in metadata["nonlinear_iterations"]]),
        "accepted_state_counts": _quantiles([
            float(len(path["updates"]) + 1) for path in paths
        ]),
        "first_endpoint_discrepancy": _quantiles([
            value["endpoint_discrepancy"] for value in first
        ]),
        "first_endpoint_cosine": _quantiles([value["endpoint_cosine"] for value in first]),
        "first_orthogonal_fraction": _quantiles([
            value["orthogonal_fraction"] for value in first
        ]),
        "first_norm_ratio": _quantiles([value["norm_ratio"] for value in first]),
        "first_residual_progress": _quantiles([
            value["residual_progress"] for value in first
        ]),
        "maximum_orthogonal_fraction": _quantiles([
            path["maximum_orthogonal_fraction"] for path in paths
        ]),
        "path_length_excess": _quantiles([path["path_length_excess"] for path in paths]),
        "turn_noncollinearity": _quantiles(
            [1.0 - value for value in turns] if turns else [0.0]
        ),
        "endpoint_velocity_fraction_of_state": _quantiles([
            path["endpoint_velocity_fraction_of_final_state"] for path in paths
        ]),
        "native_fe_divergence_rms_max": max(metadata["fe_divergence_qp_rms"]),
        "boundary_error": metadata["boundary_error"],
        "pressure_gauge_error": metadata["pressure_gauge_error"],
        "velocity_time_error": _relative(final[:2], reference[:2]),
        "pressure_time_error": _relative(final[2], reference[2]),
        "vorticity_time_error": _relative(_vorticity(final), _vorticity(reference)),
        "solve_seconds": float(sum(metadata["solver_timing"])),
        "peak_rss_gib": float(metadata["peak_rss_gib"]),
    }
    result["velocity_first_accepted"] = _first_field_metrics(shard, slice(0, 2))
    result["pressure_first_accepted"] = _first_field_metrics(shard, slice(2, 3))
    return result


def _aggregate(rows: list[dict]) -> dict:
    first_discrepancy = []
    first_orthogonal = []
    first_cosine = []
    residual_progress = []
    path_excess = []
    endpoint_fraction = []
    pressure_discrepancy = []
    pressure_orthogonal = []
    pressure_cosine = []
    pressure_endpoint_fraction = []
    for row in rows:
        # Aggregate per-case medians so no long/easy rollout dominates.
        first_discrepancy.append(row["metrics"]["first_endpoint_discrepancy"]["median"])
        first_orthogonal.append(row["metrics"]["first_orthogonal_fraction"]["median"])
        first_cosine.append(row["metrics"]["first_endpoint_cosine"]["median"])
        residual_progress.append(row["metrics"]["first_residual_progress"]["median"])
        path_excess.append(row["metrics"]["path_length_excess"]["median"])
        endpoint_fraction.append(row["metrics"]["endpoint_velocity_fraction_of_state"]["median"])
        pressure = row["metrics"]["pressure_first_accepted"]
        pressure_discrepancy.append(pressure["endpoint_discrepancy"]["median"])
        pressure_orthogonal.append(pressure["orthogonal_fraction"]["median"])
        pressure_cosine.append(pressure["endpoint_cosine"]["median"])
        pressure_endpoint_fraction.append(pressure["endpoint_fraction_of_state"]["median"])
    convergence_fraction = float(np.mean([row["metrics"]["solver_converged"] for row in rows]))
    result = {
        "cases": len(rows),
        "convergence_fraction": convergence_fraction,
        "first_endpoint_discrepancy": _quantiles(first_discrepancy),
        "first_orthogonal_fraction": _quantiles(first_orthogonal),
        "first_endpoint_cosine": _quantiles(first_cosine),
        "first_residual_progress": _quantiles(residual_progress),
        "path_length_excess": _quantiles(path_excess),
        "endpoint_velocity_fraction_of_state": _quantiles(endpoint_fraction),
        "pressure_first_endpoint_discrepancy": _quantiles(pressure_discrepancy),
        "pressure_first_orthogonal_fraction": _quantiles(pressure_orthogonal),
        "pressure_first_endpoint_cosine": _quantiles(pressure_cosine),
        "pressure_endpoint_fraction_of_state": _quantiles(pressure_endpoint_fraction),
        "velocity_time_error_max": max(row["metrics"]["velocity_time_error"] for row in rows),
        "pressure_time_error_max": max(row["metrics"]["pressure_time_error"] for row in rows),
        "vorticity_time_error_max": max(row["metrics"]["vorticity_time_error"] for row in rows),
        "relative_residual_max": max(row["metrics"]["maximum_relative_residual"] for row in rows),
        "native_fe_divergence_rms_max": max(row["metrics"]["native_fe_divergence_rms_max"] for row in rows),
        "peak_rss_gib_max": max(row["metrics"]["peak_rss_gib"] for row in rows),
        "solve_seconds_total": sum(row["metrics"]["solve_seconds"] for row in rows),
    }
    result["numerical_gate"] = bool(
        convergence_fraction >= 0.99
        and result["relative_residual_max"] <= 1e-8
        and result["native_fe_divergence_rms_max"] <= 5e-3
    )
    result["time_accuracy_gate"] = bool(
        result["velocity_time_error_max"] <= 0.01
        and result["pressure_time_error_max"] <= 0.02
        and result["vorticity_time_error_max"] <= 0.03
    )
    # A 5% endpoint difference alone is insufficient: at least 1% of the
    # endpoint norm must be genuinely orthogonal to the endpoint direction.
    velocity_path_gate = bool(
        result["first_endpoint_discrepancy"]["median"] >= 0.05
        and result["first_endpoint_discrepancy"]["p25"] >= 0.03
        and result["first_orthogonal_fraction"]["median"] >= 0.01
        and result["endpoint_velocity_fraction_of_state"]["p25"] >= 0.01
        and result["first_residual_progress"]["median"] < 0.8
    )
    pressure_path_gate = bool(
        result["pressure_first_endpoint_discrepancy"]["median"] >= 0.05
        and result["pressure_first_endpoint_discrepancy"]["p25"] >= 0.03
        and result["pressure_first_orthogonal_fraction"]["median"] >= 0.03
        and result["pressure_endpoint_fraction_of_state"]["p25"] >= 0.02
        and result["first_residual_progress"]["median"] < 0.8
    )
    result["velocity_path_information_gate"] = velocity_path_gate
    result["pressure_path_information_gate"] = pressure_path_gate
    result["path_information_gate"] = bool(velocity_path_gate or pressure_path_gate)
    result["eligible"] = bool(
        result["numerical_gate"]
        and result["time_accuracy_gate"]
        and result["path_information_gate"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dts", nargs="+", type=float, default=(0.04, 0.06, 0.08, 0.12, 0.16))
    parser.add_argument("--reynolds", type=float, default=800.0)
    parser.add_argument("--tasks", nargs="+", type=int, default=(0, 6))
    parser.add_argument("--initials", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--forcing-amplitude", type=float, default=0.05)
    parser.add_argument("--forcing-temporal-modulation", type=float, default=0.0)
    parser.add_argument("--forcing-temporal-period", type=float, default=1.4)
    parser.add_argument("--forcing-phase-amplitude", type=float, default=0.0)
    parser.add_argument("--namespace", default="bdf1_dt_primary")
    args = parser.parse_args()
    if args.initials < 1 or args.steps < 3:
        raise ValueError("Gate needs at least one initial condition and three time steps.")
    config = load_config("configs/navier_stokes_sfepy_seed42.yaml")
    config["solver"]["path_selection_mode"] = "first_accepted"
    config["resources"]["max_generation_workers"] = 1
    config["resources"]["sfepy_worker_address_space_gib"] = 8.0
    config["resources"]["sfepy_case_timeout_seconds"] = 180.0
    protocol = build_navier_stokes_protocol(config)
    rows = []
    with workspace_audit("ns_transient_information_gate"):
        require_capacity(estimated_new_bytes=2_000_000_000, maximum_gb=300.0)
        for dt_index, dt in enumerate(args.dts):
            for task in args.tasks:
                for initial_index in range(args.initials):
                    case = copy.deepcopy(protocol["targets"][str(task)]["support"][initial_index])
                    case_id = 9_910_000 + dt_index * 100 + task * 10 + initial_index
                    case.update({
                        "case_id": case_id,
                        "pair_id": case_id,
                        "rollout_id": case_id,
                        "role": "transient_information_gate",
                        "reynolds": float(args.reynolds),
                        "initial_perturbation_scale": 0.2,
                        "newton_predictor_blend": 1.0,
                        "grad_div_gamma": 100.0,
                    })
                    case["forcing"] = {
                        "active": args.forcing_amplitude > 0,
                        "amplitude": float(args.forcing_amplitude),
                        "kx": 2,
                        "ky": 2,
                        "phase": 0.37,
                        "temporal_mode": (
                            "sinusoidal" if args.forcing_temporal_modulation > 0 else "static"
                        ),
                        "temporal_bias": 1.0,
                        "temporal_modulation": float(args.forcing_temporal_modulation),
                        "temporal_omega": float(2.0 * np.pi / args.forcing_temporal_period),
                        "temporal_phase": 0.11,
                        "temporal_spatial_phase_amplitude": float(args.forcing_phase_amplitude),
                        "temporal_spatial_phase_omega": float(2.0 * np.pi / args.forcing_temporal_period),
                        "temporal_spatial_phase_offset": 0.23,
                    }
                    root = f"artifacts/data/navier_stokes_sfepy_transient_gate/{args.namespace}"
                    path = f"{root}/dt{dt:g}_task{task:02d}_initial{initial_index}.pt"
                    half_path = f"{root}/dt{0.5*dt:g}_task{task:02d}_initial{initial_index}_reference.pt"
                    run_sfepy_navier_stokes_case(
                        case, output_shard=path, output_resolution=args.resolution,
                        mesh_resolution=args.resolution // 2 + 1, steps=args.steps,
                        dt=dt, config=config, fidelity=0, time_integrator="BDF1",
                    )
                    half_case = copy.deepcopy(case)
                    half_case["case_id"] += 50_000
                    half_case["rollout_id"] = half_case["case_id"]
                    run_sfepy_navier_stokes_case(
                        half_case, output_shard=half_path, output_resolution=args.resolution,
                        mesh_resolution=args.resolution // 2 + 1, steps=2 * args.steps,
                        dt=0.5 * dt, config=config, fidelity=0, time_integrator="BDF1",
                    )
                    shard = torch.load(resolve_in_project(path), map_location="cpu", weights_only=False)
                    half = torch.load(resolve_in_project(half_path), map_location="cpu", weights_only=False)
                    row = {
                        "dt": dt, "reynolds": args.reynolds, "task": task,
                        "boundary_family": case["boundary"]["family"],
                        "initial_index": initial_index, "path": path,
                        "path_sha256": sha256_file(resolve_in_project(path)),
                        "reference_path": half_path,
                        "reference_sha256": sha256_file(resolve_in_project(half_path)),
                        "metrics": _case_metrics(shard, half),
                    }
                    rows.append(row)
                    print(json.dumps({"dt": dt, "task": task, "initial": initial_index,
                                      "metrics": row["metrics"]}, sort_keys=True), flush=True)
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row["dt"])].append(row)
        summary = {dt: _aggregate(values) for dt, values in grouped.items()}
        eligible = [float(dt) for dt, value in summary.items() if value["eligible"]]
        report = {
            "schema": "blrpng_ns_transient_information_gate_v1",
            "purpose": "natural_previous_state_BDF1_dt_only_primary_gate",
            "newton_predictor": "previous_physical_state_without_degradation",
            "trajectory_axis": "physical_time",
            "fixed": {
                "reynolds": args.reynolds, "time_integrator": "BDF1",
                "resolution": args.resolution, "mesh_resolution": args.resolution // 2 + 1,
                "steps": args.steps, "tasks": args.tasks, "initials_per_task": args.initials,
                "forcing_amplitude": args.forcing_amplitude, "grad_div_gamma": 100.0,
                "forcing_temporal_modulation": args.forcing_temporal_modulation,
                "forcing_temporal_period": args.forcing_temporal_period,
                "forcing_phase_amplitude": args.forcing_phase_amplitude,
            },
            "thresholds": {
                "velocity_time_error_max": 0.01, "pressure_time_error_max": 0.02,
                "vorticity_time_error_max": 0.03, "first_discrepancy_median_min": 0.05,
                "first_discrepancy_case_p25_min": 0.03,
                "first_orthogonal_fraction_median_min": 0.01,
                "pressure_first_discrepancy_median_min": 0.05,
                "pressure_first_orthogonal_fraction_median_min": 0.03,
                "first_residual_progress_median_max": 0.8,
            },
            "summary": summary, "eligible_dts": eligible,
            "decision": "confirm_at_128" if eligible else "run_bounded_dt_re_gate_or_single_step_high_order_study",
            "records": rows,
        }
        output = write_json(
            f"artifacts/reports/navier_stokes_sfepy_transient_gate/{args.namespace}.json",
            report,
        )
    print(json.dumps({"report": str(output), "sha256": sha256_file(output),
                      "eligible_dts": eligible, "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
