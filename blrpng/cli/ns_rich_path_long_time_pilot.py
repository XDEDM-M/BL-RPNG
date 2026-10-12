"""Select a time-accurate dt for 128², 20-step Newton-rich NS rollouts."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import build_navier_stokes_protocol
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.paths import PROJECT_ROOT, sha256_file, write_json


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    h = 1.0 / float(field.shape[-1] - 1)
    return torch.gradient(field[1], spacing=(h,), dim=(0,))[0] - torch.gradient(
        field[0], spacing=(h,), dim=(1,)
    )[0]


def _relative(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value - reference).norm() / reference.norm().clamp_min(1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dts", nargs="+", type=float, default=(0.06, 0.08))
    parser.add_argument("--forcing-amplitude", type=float, default=0.15)
    parser.add_argument("--temporal-modulation", type=float, default=0.0)
    parser.add_argument("--temporal-period", type=float, default=0.64)
    parser.add_argument(
        "--time-integrator", choices=("BDF2", "BDF3"), default="BDF2"
    )
    parser.add_argument("--namespace", default="long_time")
    args = parser.parse_args()
    config = load_config("configs/navier_stokes_sfepy_seed42.yaml")
    protocol = build_navier_stokes_protocol(config)
    dts = tuple(float(value) for value in args.dts)
    tasks = (0, 6)
    steps = 20
    records = []
    with workspace_audit("ns_rich_path_long_time_pilot"):
        for dt_index, dt in enumerate(dts):
            for task in tasks:
                case = copy.deepcopy(protocol["targets"][str(task)]["support"][0])
                force_code = int(round(1000.0 * args.forcing_amplitude))
                case_id = 9_600_000 + force_code * 10_000 + dt_index * 1000 + task
                case.update(
                    {
                        "case_id": case_id,
                        "pair_id": case_id,
                        "rollout_id": case_id,
                        "role": "ns_rich_path_long_time_pilot",
                        "reynolds": 800.0,
                        "initial_perturbation_scale": 0.2,
                    }
                )
                case["forcing"] = {
                    "active": True,
                    "amplitude": float(args.forcing_amplitude),
                    "kx": 2,
                    "ky": 2,
                    "phase": 0.37,
                    "temporal_mode": (
                        "sinusoidal" if args.temporal_modulation > 0.0 else "static"
                    ),
                    "temporal_bias": 1.0,
                    "temporal_modulation": float(args.temporal_modulation),
                    "temporal_omega": float(2.0 * np.pi / args.temporal_period),
                    "temporal_phase": 0.11,
                }
                base_path = (
                    f"artifacts/data/navier_stokes_sfepy_rich_path_long/{args.namespace}/"
                    f"dt{dt:g}_task{task:02d}.pt"
                )
                base_row = run_sfepy_navier_stokes_case(
                    case,
                    output_shard=base_path,
                    output_resolution=128,
                    mesh_resolution=65,
                    steps=steps,
                    dt=dt,
                    config=config,
                    fidelity=1,
                    time_integrator=args.time_integrator,
                )
                base = torch.load(base_path, map_location="cpu", weights_only=False)
                half_case = copy.deepcopy(case)
                half_case["case_id"] += 500_000
                half_case["rollout_id"] = half_case["case_id"]
                half_path = (
                    f"artifacts/data/navier_stokes_sfepy_rich_path_long/{args.namespace}/"
                    f"dt{dt:g}_task{task:02d}_half.pt"
                )
                half_row = run_sfepy_navier_stokes_case(
                    half_case,
                    output_shard=half_path,
                    output_resolution=128,
                    mesh_resolution=65,
                    steps=2 * steps,
                    dt=0.5 * dt,
                    config=config,
                    fidelity=1,
                    time_integrator=args.time_integrator,
                )
                half = torch.load(half_path, map_location="cpu", weights_only=False)
                coarse = base["trajectory"][-1].double()
                reference = half["trajectory"][-1].double()
                startup_steps = 2 if args.time_integrator == "BDF3" else 1
                discrepancy = np.asarray(
                    base["meta"]["first_update_endpoint_discrepancy"][startup_steps:]
                )
                record = {
                    "dt": dt,
                    "task": task,
                    "boundary_family": case["boundary"]["family"],
                    "converged": bool(base["meta"]["solver_converged"]),
                    "discrepancy_median": float(np.median(discrepancy)),
                    "discrepancy_p25": float(np.quantile(discrepancy, 0.25)),
                    "discrepancy_min": float(np.min(discrepancy)),
                    "second_to_first_median": float(
                        np.median(
                            base["meta"]["second_to_first_correction_ratio"][startup_steps:]
                        )
                    ),
                    "velocity_time_error": _relative(coarse[:2], reference[:2]),
                    "pressure_time_error": _relative(coarse[2], reference[2]),
                    "vorticity_time_error": _relative(
                        _vorticity(coarse), _vorticity(reference)
                    ),
                    "peak_rss_gib": max(base_row["peak_rss_gib"], half_row["peak_rss_gib"]),
                    "maximum_relative_residual": max(base["meta"]["relative_residuals"]),
                }
                records.append(record)
                write_json(
                    f"artifacts/reports/navier_stokes_sfepy_rich_path/{args.namespace}_progress.json",
                    {"completed": len(records), "total": len(dts) * len(tasks), "latest": record},
                )
                print(json.dumps(record, sort_keys=True), flush=True)

        summaries = {}
        eligible = []
        for dt in dts:
            rows = [row for row in records if row["dt"] == dt]
            discrepancy = [row["discrepancy_median"] for row in rows]
            summary = {
                "tasks": len(rows),
                "discrepancy_median_across_tasks": float(np.median(discrepancy)),
                "discrepancy_min_task": float(min(discrepancy)),
                "velocity_time_error_max": max(row["velocity_time_error"] for row in rows),
                "pressure_time_error_max": max(row["pressure_time_error"] for row in rows),
                "vorticity_time_error_max": max(row["vorticity_time_error"] for row in rows),
            }
            summary["path_gate_passed"] = bool(
                summary["discrepancy_median_across_tasks"] >= 0.05
                and summary["discrepancy_min_task"] >= 0.03
            )
            summary["time_gate_passed"] = bool(
                summary["velocity_time_error_max"] <= 0.01
                and summary["pressure_time_error_max"] <= 0.02
                and summary["vorticity_time_error_max"] <= 0.03
            )
            summary["eligible"] = bool(
                summary["path_gate_passed"] and summary["time_gate_passed"]
            )
            summaries[str(dt)] = summary
            if summary["eligible"]:
                eligible.append(dt)
        selected = max(eligible) if eligible else None
        report = {
            "schema": "blrpng_ns_sfepy_rich_path_long_time_pilot_v1",
            "config_sha256": config["_config_sha256"],
            "resolution": 128,
            "mesh_resolution": 65,
            "steps": steps,
            "tasks": list(tasks),
            "fixed_setting": {
                "reynolds": 800,
                "initial_perturbation_scale": 0.2,
                "forcing_amplitude": float(args.forcing_amplitude),
                "forcing_temporal_modulation": float(args.temporal_modulation),
                "forcing_temporal_period": float(args.temporal_period),
                "time_integrator": args.time_integrator,
            },
            "summary": summaries,
            "eligible_dts": eligible,
            "selected_dt": selected,
            "decision": (
                "run_six_task_confirmation" if selected is not None else "redesign_time_dependent_forcing_or_steady_continuation"
            ),
            "records": records,
        }
        output = write_json(
            f"artifacts/reports/navier_stokes_sfepy_rich_path/{args.namespace}_pilot.json",
            report,
        )
    print(
        json.dumps(
            {
                "report": str(output),
                "report_sha256": sha256_file(output),
                "selected_dt": selected,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
