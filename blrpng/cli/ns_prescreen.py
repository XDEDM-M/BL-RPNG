"""Solver-only gate for the SfePy Taylor--Hood Navier--Stokes study.

This command deliberately runs before any neural-network training.  It checks
both numerical fidelity and whether the accepted Newton path contains a target
that is meaningfully different from the physical endpoint increment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import build_navier_stokes_protocol
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.paths import ensure_artifact_path, sha256_file, write_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/navier_stokes_sfepy_seed42.yaml"
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "dt64"),
        default="dt64",
        help="smoke runs one condition; dt64 runs the full solver/path prescreen.",
    )
    return parser.parse_args()


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    spacing = 1.0 / float(field.shape[-1] - 1)
    dv_dx = torch.gradient(field[1], spacing=(spacing,), dim=(0,))[0]
    du_dy = torch.gradient(field[0], spacing=(spacing,), dim=(1,))[0]
    return dv_dx - du_dy


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30))


def _step_records(payload: dict[str, Any], *, reynolds: int, dt: float) -> list[dict[str, Any]]:
    correction = payload["newton_correction"].double()
    endpoint = payload["endpoint_increment"].double()
    scales = correction.new_tensor((1.0, 1.0, 0.25))[None, :, None, None]
    scaled_correction = correction / scales
    scaled_endpoint = endpoint / scales
    correction_flat = scaled_correction.flatten(1)
    endpoint_flat = scaled_endpoint.flatten(1)
    correction_norm = correction_flat.norm(dim=1)
    endpoint_norm = endpoint_flat.norm(dim=1)
    cosine = (correction_flat * endpoint_flat).sum(1) / (
        correction_norm * endpoint_norm
    ).clamp_min(1e-30)
    discrepancy = (correction_flat - endpoint_flat).norm(dim=1) / endpoint_norm.clamp_min(1e-30)
    meta = payload["meta"]
    records = []
    for step in range(correction.shape[0]):
        records.append(
            {
                "reynolds": int(reynolds),
                "dt": float(dt),
                "step": int(step),
                "converged": bool(meta["solver_converged"]),
                "relative_residual": float(meta["relative_residuals"][step]),
                "newton_iterations": int(meta["nonlinear_iterations"][step]),
                "accepted_states": int(payload["accepted_newton_counts"][step]),
                "valid_correction": bool(payload["valid_correction_mask"][step].all()),
                "selected_progress": float(payload["residual_progress"][step, 0]),
                "correction_endpoint_discrepancy": float(discrepancy[step]),
                "correction_endpoint_cosine": float(cosine[step]),
                "boundary_error": float(meta["boundary_error"]),
                "pressure_gauge_error": float(meta["pressure_gauge_error"]),
                "weak_continuity": float(meta["weak_continuity"][step]),
                "grid_divergence_rms": float(meta["grid_divergence_rms"][step]),
                "net_flux": float(meta["net_flux"][step]),
                "solve_seconds": float(meta["solver_timing"][step]),
                "peak_rss_gib": float(meta["peak_rss_gib"]),
            }
        )
    return records


def _summary(
    records: list[dict[str, Any]], refinement: list[dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    discrepancy = np.asarray([row["correction_endpoint_discrepancy"] for row in records])
    iterations = np.asarray([row["newton_iterations"] for row in records])
    timings = np.asarray([row["solve_seconds"] for row in records])
    result = {
        "steps": len(records),
        "convergence_fraction": float(np.mean([row["converged"] for row in records])),
        "valid_correction_fraction": float(np.mean([row["valid_correction"] for row in records])),
        "maximum_relative_residual": float(max(row["relative_residual"] for row in records)),
        "maximum_boundary_error": float(max(row["boundary_error"] for row in records)),
        "maximum_pressure_gauge_error": float(max(row["pressure_gauge_error"] for row in records)),
        "maximum_weak_continuity": float(max(row["weak_continuity"] for row in records)),
        "maximum_grid_divergence_rms": float(max(row["grid_divergence_rms"] for row in records)),
        "maximum_net_flux_absolute": float(max(abs(row["net_flux"]) for row in records)),
        "correction_endpoint_discrepancy_median": float(np.median(discrepancy)),
        "correction_endpoint_discrepancy_p25": float(np.quantile(discrepancy, 0.25)),
        "correction_endpoint_cosine_mean": float(np.mean([row["correction_endpoint_cosine"] for row in records])),
        "newton_iterations_median": float(np.median(iterations)),
        "accepted_states_median": float(np.median([row["accepted_states"] for row in records])),
        "solve_seconds_mean": float(np.mean(timings)),
        "solve_seconds_cv": float(np.std(timings) / max(np.mean(timings), 1e-30)),
        "time_refinement_velocity_error_max": float(max(row["velocity_error"] for row in refinement)),
        "time_refinement_pressure_error_max": float(max(row["pressure_error"] for row in refinement)),
        "time_refinement_vorticity_error_max": float(max(row["vorticity_error"] for row in refinement)),
    }
    numerical_checks = {
        "convergence": result["convergence_fraction"] >= float(gates["convergence_fraction_min"]),
        "residual": result["maximum_relative_residual"] <= float(gates["final_relative_residual_max"]),
        "boundary": result["maximum_boundary_error"] <= float(gates["boundary_error_max"]),
        "pressure_gauge": result["maximum_pressure_gauge_error"] <= float(gates["pressure_gauge_error_max"]),
        "weak_continuity": result["maximum_weak_continuity"] <= float(gates["weak_continuity_error_max"]),
        "grid_divergence": result["maximum_grid_divergence_rms"] <= float(gates["grid_divergence_rms_max"]),
        "velocity_time_refinement": result["time_refinement_velocity_error_max"] <= float(gates["velocity_time_refinement_error_max"]),
        "pressure_time_refinement": result["time_refinement_pressure_error_max"] <= float(gates["pressure_time_refinement_error_max"]),
        "vorticity_time_refinement": result["time_refinement_vorticity_error_max"] <= float(gates["vorticity_time_refinement_error_max"]),
    }
    path_checks = {
        "valid_corrections": result["valid_correction_fraction"] >= float(gates["valid_correction_fraction_min"]),
        "median_discrepancy": result["correction_endpoint_discrepancy_median"] >= float(gates["correction_endpoint_discrepancy_median_min"]),
        "p25_discrepancy": result["correction_endpoint_discrepancy_p25"] >= float(gates["correction_endpoint_discrepancy_p25_min"]),
        "newton_iterations": result["newton_iterations_median"] >= float(gates["accepted_newton_updates_median_min"]),
    }
    result["numerical_checks"] = numerical_checks
    result["path_checks"] = path_checks
    result["numerical_gate_passed"] = bool(all(numerical_checks.values()))
    result["path_richness_gate_passed"] = bool(all(path_checks.values()))
    result["eligible"] = bool(result["numerical_gate_passed"] and result["path_richness_gate_passed"])
    return result


def _plot(report: dict[str, Any]) -> Path:
    summaries = report["summary_by_dt"]
    dts = np.asarray([float(value) for value in report["dt_candidates"]])
    rows = [summaries[str(value)] for value in report["dt_candidates"]]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(dts, [100.0 * row["correction_endpoint_discrepancy_median"] for row in rows], "o-", label="median")
    axes[0, 0].plot(dts, [100.0 * row["correction_endpoint_discrepancy_p25"] for row in rows], "s--", label="25th percentile")
    axes[0, 0].axhline(5.0, color="#D55E00", linestyle=":", label="median gate")
    axes[0, 0].set_ylabel("Newton vs endpoint discrepancy (%)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(dts, [row["newton_iterations_median"] for row in rows], "o-")
    axes[0, 1].axhline(3.0, color="#D55E00", linestyle=":")
    axes[0, 1].set_ylabel("Median Newton iterations")
    axes[1, 0].semilogy(dts, [row["maximum_relative_residual"] for row in rows], "o-", label="nonlinear residual")
    axes[1, 0].semilogy(dts, [row["maximum_weak_continuity"] for row in rows], "s--", label="weak continuity")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_ylabel("Maximum residual")
    axes[1, 1].plot(dts, [100.0 * row["time_refinement_velocity_error_max"] for row in rows], "o-", label="velocity")
    axes[1, 1].plot(dts, [100.0 * row["time_refinement_pressure_error_max"] for row in rows], "s--", label="pressure")
    axes[1, 1].plot(dts, [100.0 * row["time_refinement_vorticity_error_max"] for row in rows], "^--", label="vorticity")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_ylabel("dt vs dt/2 difference (%)")
    for axis in axes.flat:
        axis.set_xlabel("BDF1 dt")
        axis.grid(alpha=0.22)
    figure.suptitle("SfePy Navier–Stokes numerical and Newton-information prescreen")
    output = ensure_artifact_path("artifacts/figures/navier_stokes_sfepy_seed42/01_dt_newton_richness.png")
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    protocol = build_navier_stokes_protocol(config)
    dt_candidates = [float(value) for value in config["solver"]["dt_candidates"]]
    reynolds_values = [int(value) for value in config["solver"]["reynolds_prescreen"]]
    if args.stage == "smoke":
        dt_candidates = [dt_candidates[0]]
        reynolds_values = [reynolds_values[0]]
    steps = 1 if args.stage == "smoke" else int(config["solver"]["prescreen_steps"])
    base_cases = protocol["development_train"]
    records: list[dict[str, Any]] = []
    refinement: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    with workspace_audit("ns_prescreen"):
        for re_index, reynolds in enumerate(reynolds_values):
            template = copy.deepcopy(base_cases[re_index])
            template["reynolds"] = float(reynolds)
            template["role"] = "solver_prescreen"
            # Exercise low-frequency, high-frequency and localized boundary content.
            if re_index == 1:
                template["boundary"]["fourier_frequencies"] = [5, 6]
            elif re_index == 2:
                template["boundary"]["bump_amplitude"] = 0.45
                template["boundary"]["bump_sigma"] = 0.06
            for dt_index, dt in enumerate(dt_candidates):
                case = copy.deepcopy(template)
                case["case_id"] = 9_000_000 + re_index * 100 + dt_index
                case["pair_id"] = case["case_id"]
                case["rollout_id"] = case["case_id"]
                base_path = (
                    f"artifacts/data/navier_stokes_sfepy_prescreen/"
                    f"re{reynolds}_dt{dt:g}.pt"
                )
                row = run_sfepy_navier_stokes_case(
                    case,
                    output_shard=base_path,
                    output_resolution=64,
                    mesh_resolution=33,
                    steps=steps,
                    dt=dt,
                    config=config,
                    fidelity=0,
                )
                payload = torch.load(ensure_artifact_path(base_path), map_location="cpu", weights_only=False)
                records.extend(_step_records(payload, reynolds=reynolds, dt=dt))
                half_case = copy.deepcopy(case)
                half_case["case_id"] += 50_000
                half_case["pair_id"] = case["pair_id"]
                half_case["rollout_id"] = half_case["case_id"]
                half_path = (
                    f"artifacts/data/navier_stokes_sfepy_prescreen/"
                    f"re{reynolds}_dt{dt:g}_half.pt"
                )
                half_row = run_sfepy_navier_stokes_case(
                    half_case,
                    output_shard=half_path,
                    output_resolution=64,
                    mesh_resolution=33,
                    steps=2 * steps,
                    dt=0.5 * dt,
                    config=config,
                    fidelity=0,
                )
                half_payload = torch.load(ensure_artifact_path(half_path), map_location="cpu", weights_only=False)
                coarse_final = payload["trajectory"][-1].double()
                half_final = half_payload["trajectory"][-1].double()
                refinement.append(
                    {
                        "reynolds": reynolds,
                        "dt": dt,
                        "velocity_error": _relative(coarse_final[:2], half_final[:2]),
                        "pressure_error": _relative(coarse_final[2], half_final[2]),
                        "vorticity_error": _relative(_vorticity(coarse_final), _vorticity(half_final)),
                    }
                )
                shard_rows.extend((row, half_row))
                print(
                    json.dumps(
                        {
                            "reynolds": reynolds,
                            "dt": dt,
                            "median_discrepancy_percent": 100.0
                            * float(
                                np.median(
                                    [
                                        value["correction_endpoint_discrepancy"]
                                        for value in records
                                        if value["reynolds"] == reynolds and value["dt"] == dt
                                    ]
                                )
                            ),
                            "velocity_dt_error_percent": 100.0 * refinement[-1]["velocity_error"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        summaries: dict[str, Any] = {}
        for dt in dt_candidates:
            dt_records = [row for row in records if row["dt"] == dt]
            dt_refinement = [row for row in refinement if row["dt"] == dt]
            summaries[str(dt)] = _summary(dt_records, dt_refinement, config["gates"])
        eligible = [dt for dt in dt_candidates if summaries[str(dt)]["eligible"]]
        selected = max(eligible) if eligible else None
        report = {
            "schema": "blrpng_sfepy_navier_stokes_prescreen_v1",
            "config": config["_config_path"],
            "config_sha256": config["_config_sha256"],
            "protocol_sha256": protocol["sha256"],
            "stage": args.stage,
            "selection_uses_model_or_hr_query": False,
            "output_resolution": 64,
            "mesh_resolution": 33,
            "steps": steps,
            "reynolds": reynolds_values,
            "dt_candidates": dt_candidates,
            "summary_by_dt": summaries,
            "eligible_dts": eligible,
            "selected_dt": selected,
            "solver_gate_passed": selected is not None,
            "decision": (
                "proceed_to_128_development_generation"
                if selected is not None
                else "stop_before_neural_training; Newton_path_is_numerically_or_informationally_ineligible"
            ),
            "records": records,
            "time_refinement": refinement,
            "shards": shard_rows,
        }
        report_path = write_json(
            "artifacts/reports/navier_stokes_sfepy_seed42/prescreen.json", report
        )
        gate = {
            "schema": "blrpng_sfepy_navier_stokes_solver_gate_v1",
            "passed": bool(report["solver_gate_passed"]),
            "selected_dt": selected,
            "report": str(report_path.relative_to(Path(__file__).resolve().parents[2])),
            "report_sha256": sha256_file(report_path),
            "reason": report["decision"],
        }
        gate_path = write_json(
            "artifacts/manifests/navier_stokes_sfepy_solver_gate.json", gate
        )
        figure_path = _plot(report)
        write_json(
            figure_path.with_suffix(".json"),
            {
                "source": str(report_path.relative_to(Path(__file__).resolve().parents[2])),
                "source_sha256": sha256_file(report_path),
                "png_sha256": sha256_file(figure_path),
                "selection_performed": True,
                "selection_does_not_use_model_or_hr_query": True,
            },
        )
        print(
            json.dumps(
                {
                    "gate": str(gate_path),
                    "selected_dt": selected,
                    "eligible_dts": eligible,
                    "decision": report["decision"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
