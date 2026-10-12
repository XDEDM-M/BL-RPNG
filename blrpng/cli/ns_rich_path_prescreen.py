"""Search for physically meaningful, time-accurate Navier--Stokes Newton paths."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import build_navier_stokes_protocol
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, write_json


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    h = 1.0 / float(field.shape[-1] - 1)
    return torch.gradient(field[1], spacing=(h,), dim=(0,))[0] - torch.gradient(
        field[0], spacing=(h,), dim=(1,)
    )[0]


def _relative(first: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(first - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def _key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        int(row["reynolds"]),
        float(row["dt"]),
        float(row["initial_perturbation_scale"]),
        float(row["forcing_amplitude"]),
    )


def _richness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    discrepancy = np.asarray([row["first_update_endpoint_discrepancy"] for row in rows])
    second_ratio = np.asarray([row["second_to_first_correction_ratio"] for row in rows])
    iterations = np.asarray([row["newton_iterations"] for row in rows])
    by_family = {
        family: float(
            np.median(
                [
                    row["first_update_endpoint_discrepancy"]
                    for row in rows
                    if row["boundary_family"] == family
                ]
            )
        )
        for family in sorted({row["boundary_family"] for row in rows})
    }
    result = {
        "samples": len(rows),
        "all_converged": all(row["converged"] for row in rows),
        "maximum_relative_residual": max(row["relative_residual"] for row in rows),
        "discrepancy_median": float(np.median(discrepancy)),
        "discrepancy_p25": float(np.quantile(discrepancy, 0.25)),
        "discrepancy_min": float(np.min(discrepancy)),
        "second_to_first_median": float(np.median(second_ratio)),
        "nonlinear_remainder_median": float(
            np.median([row["nonlinear_remainder_ratio"] for row in rows])
        ),
        "newton_iterations_median": float(np.median(iterations)),
        "family_discrepancy_medians": by_family,
    }
    result["path_gate_passed"] = bool(
        result["all_converged"]
        and result["maximum_relative_residual"] <= 1e-8
        and result["discrepancy_median"] >= 0.05
        and result["discrepancy_p25"] >= 0.01
        and result["newton_iterations_median"] >= 3
        and all(value >= 0.03 for value in by_family.values())
    )
    return result


def _case_records(payload: dict[str, Any], specification: dict[str, Any], exclude_startup: bool) -> list[dict[str, Any]]:
    metadata = payload["meta"]
    start = 1 if exclude_startup and metadata["time_integrator"] == "BDF2" else 0
    output = []
    for step in range(start, int(metadata["steps"])):
        output.append(
            {
                **specification,
                "step": step,
                "converged": bool(metadata["solver_converged"]),
                "relative_residual": float(metadata["relative_residuals"][step]),
                "newton_iterations": int(metadata["nonlinear_iterations"][step]),
                "first_update_endpoint_discrepancy": float(
                    metadata["first_update_endpoint_discrepancy"][step]
                ),
                "second_to_first_correction_ratio": float(
                    metadata["second_to_first_correction_ratio"][step]
                ),
                "nonlinear_remainder_ratio": float(
                    metadata["nonlinear_remainder_ratio"][step]
                ),
                "grid_divergence_rms": float(metadata["grid_divergence_rms"][step]),
                "weak_continuity": float(metadata["weak_continuity"][step]),
                "solve_seconds": float(metadata["solver_timing"][step]),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navier_stokes_sfepy_seed42.yaml")
    parser.add_argument("--max-settings", type=int, help="Deterministic debug truncation.")
    args = parser.parse_args()
    config = load_config(args.config)
    search = config["solver"]["richness_search"]
    protocol = build_navier_stokes_protocol(config)
    resolution = int(search["resolution"])
    mesh_resolution = int(search["mesh_resolution"])
    steps = int(search["steps"])
    integrator = str(search["time_integrator"])
    boundary_tasks = [int(value) for value in search["boundary_tasks"]]
    settings = list(
        itertools.product(
            [int(value) for value in search["reynolds"]],
            [float(value) for value in search["dt_candidates"]],
            [float(value) for value in search["initial_perturbation_scales"]],
            [float(value) for value in search["forcing_amplitudes"]],
        )
    )
    if args.max_settings is not None:
        settings = settings[: int(args.max_settings)]
    require_capacity(
        len(settings) * len(boundary_tasks) * 8 * 1024 * 1024,
        maximum_gb=float(config["resources"]["max_total_artifacts_gib"]),
    )
    records: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    progress_path = "artifacts/reports/navier_stokes_sfepy_rich_path/progress.json"
    with workspace_audit("ns_rich_path_prescreen"):
        serial = 0
        for setting_index, (reynolds, dt, initial_scale, forcing_amplitude) in enumerate(settings):
            for boundary_task in boundary_tasks:
                serial += 1
                template = copy.deepcopy(
                    protocol["targets"][str(boundary_task)]["support"][0]
                )
                case_id = 9_200_000 + setting_index * 10 + boundary_task
                template.update(
                    {
                        "case_id": case_id,
                        "pair_id": case_id,
                        "rollout_id": case_id,
                        "role": "ns_rich_path_prescreen",
                        "reynolds": float(reynolds),
                        "initial_perturbation_scale": float(initial_scale),
                    }
                )
                template["forcing"] = {
                    "active": forcing_amplitude > 0.0,
                    "amplitude": float(forcing_amplitude),
                    "kx": 2,
                    "ky": 2,
                    "phase": 0.37,
                }
                stem = (
                    f"re{reynolds}_dt{dt:g}_ic{initial_scale:g}_"
                    f"f{forcing_amplitude:g}_b{boundary_task:02d}"
                )
                path = f"artifacts/data/navier_stokes_sfepy_rich_path/{stem}.pt"
                row = run_sfepy_navier_stokes_case(
                    template,
                    output_shard=path,
                    output_resolution=resolution,
                    mesh_resolution=mesh_resolution,
                    steps=steps,
                    dt=dt,
                    config=config,
                    fidelity=0,
                    time_integrator=integrator,
                )
                payload = torch.load(path, map_location="cpu", weights_only=False)
                specification = {
                    "reynolds": reynolds,
                    "dt": dt,
                    "initial_perturbation_scale": initial_scale,
                    "forcing_amplitude": forcing_amplitude,
                    "boundary_task": boundary_task,
                    "boundary_family": str(template["boundary"]["family"]),
                }
                records.extend(
                    _case_records(
                        payload,
                        specification,
                        bool(search["exclude_startup_step_from_richness"]),
                    )
                )
                shard_rows.append(row)
                write_json(
                    progress_path,
                    {
                        "completed_cases": serial,
                        "total_cases": len(settings) * len(boundary_tasks),
                        "latest": specification,
                    },
                )
                print(
                    json.dumps(
                        {
                            **specification,
                            "case_median_discrepancy_percent": 100.0
                            * float(
                                np.median(
                                    payload["meta"]["first_update_endpoint_discrepancy"][1:]
                                )
                            ),
                            "iterations": payload["meta"]["nonlinear_iterations"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        grouped: dict[tuple[int, float, float, float], list[dict[str, Any]]] = {}
        for row in records:
            grouped.setdefault(_key(row), []).append(row)
        summaries = {
            "|".join(map(str, key)): {"setting": list(key), **_richness_summary(rows)}
            for key, rows in sorted(grouped.items())
        }
        preliminary = [
            tuple(summary["setting"])
            for summary in summaries.values()
            if summary["path_gate_passed"]
        ]

        refinement: list[dict[str, Any]] = []
        for reynolds, dt, initial_scale, forcing_amplitude in preliminary:
            for boundary_task in boundary_tasks:
                stem = (
                    f"re{int(reynolds)}_dt{dt:g}_ic{initial_scale:g}_"
                    f"f{forcing_amplitude:g}_b{boundary_task:02d}"
                )
                base_path = PROJECT_ROOT / f"artifacts/data/navier_stokes_sfepy_rich_path/{stem}.pt"
                base = torch.load(base_path, map_location="cpu", weights_only=False)
                half_case = copy.deepcopy(base["meta"]["case"])
                half_case["case_id"] = int(half_case["case_id"]) + 500_000
                half_case["rollout_id"] = half_case["case_id"]
                half_path = f"artifacts/data/navier_stokes_sfepy_rich_path/{stem}_half.pt"
                half_row = run_sfepy_navier_stokes_case(
                    half_case,
                    output_shard=half_path,
                    output_resolution=resolution,
                    mesh_resolution=mesh_resolution,
                    steps=2 * steps,
                    dt=0.5 * float(dt),
                    config=config,
                    fidelity=0,
                    time_integrator=integrator,
                )
                shard_rows.append(half_row)
                half = torch.load(half_path, map_location="cpu", weights_only=False)
                coarse_final = base["trajectory"][-1].double()
                fine_final = half["trajectory"][-1].double()
                refinement.append(
                    {
                        "reynolds": int(reynolds),
                        "dt": float(dt),
                        "initial_perturbation_scale": float(initial_scale),
                        "forcing_amplitude": float(forcing_amplitude),
                        "boundary_task": boundary_task,
                        "velocity_error": _relative(coarse_final[:2], fine_final[:2]),
                        "pressure_error": _relative(coarse_final[2], fine_final[2]),
                        "vorticity_error": _relative(
                            _vorticity(coarse_final), _vorticity(fine_final)
                        ),
                    }
                )

        eligible = []
        for setting in preliminary:
            rows = [row for row in refinement if _key(row) == setting]
            errors = {
                "velocity_max": max(row["velocity_error"] for row in rows),
                "pressure_max": max(row["pressure_error"] for row in rows),
                "vorticity_max": max(row["vorticity_error"] for row in rows),
            }
            key = "|".join(map(str, setting))
            summaries[key]["time_refinement"] = errors
            summaries[key]["time_gate_passed"] = bool(
                errors["velocity_max"] <= float(config["gates"]["velocity_time_refinement_error_max"])
                and errors["pressure_max"] <= float(config["gates"]["pressure_time_refinement_error_max"])
                and errors["vorticity_max"] <= float(config["gates"]["vorticity_time_refinement_error_max"])
            )
            if summaries[key]["time_gate_passed"]:
                eligible.append(setting)
        eligible.sort(
            key=lambda setting: summaries["|".join(map(str, setting))][
                "discrepancy_median"
            ],
            reverse=True,
        )
        report = {
            "schema": "blrpng_ns_sfepy_rich_path_prescreen_v1",
            "config": config["_config_path"],
            "config_sha256": config["_config_sha256"],
            "protocol_sha256": protocol["sha256"],
            "time_integrator": integrator,
            "resolution": resolution,
            "mesh_resolution": mesh_resolution,
            "steps": steps,
            "startup_step_excluded_from_richness": bool(
                search["exclude_startup_step_from_richness"]
            ),
            "settings_tested": len(settings),
            "boundary_tasks": boundary_tasks,
            "summary": summaries,
            "preliminary_path_eligible": [list(value) for value in preliminary],
            "eligible_time_accurate_settings": [list(value) for value in eligible],
            "selected_setting": list(eligible[0]) if eligible else None,
            "decision": (
                "run_128_grid_and_longer_rollout_confirmation"
                if eligible
                else "no_meaningful_time_accurate_path_found_in_R1"
            ),
            "records": records,
            "time_refinement": refinement,
            "shards": shard_rows,
        }
        report_path = write_json(
            "artifacts/reports/navier_stokes_sfepy_rich_path/prescreen.json", report
        )

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        values = sorted(
            summaries.values(), key=lambda value: value["discrepancy_median"], reverse=True
        )
        x = np.arange(len(values))
        axes[0].plot(x, [100 * value["discrepancy_median"] for value in values], "o", label="median")
        axes[0].plot(x, [100 * value["discrepancy_p25"] for value in values], ".", label="p25")
        axes[0].axhline(5.0, color="#D55E00", linestyle=":", label="median gate")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("settings ranked by median discrepancy")
        axes[0].set_ylabel("first Newton update vs endpoint (%)")
        axes[0].legend(frameon=False)
        axes[1].scatter(
            [value["newton_iterations_median"] for value in values],
            [100 * value["discrepancy_median"] for value in values],
            c=[value["setting"][2] for value in values],
            cmap="viridis",
        )
        axes[1].axhline(5.0, color="#D55E00", linestyle=":")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("median Newton iterations")
        axes[1].set_ylabel("median discrepancy (%)")
        axes[1].grid(alpha=0.2)
        figure.suptitle("BDF2 NS Newton-richness search")
        figure_path = ensure_artifact_path(
            "artifacts/figures/navier_stokes_sfepy_rich_path/01_richness_search.png"
        )
        figure.savefig(figure_path, dpi=300)
        plt.close(figure)
        write_json(
            figure_path.with_suffix(".json"),
            {
                "source": str(report_path.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256_file(report_path),
                "png_sha256": sha256_file(figure_path),
            },
        )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "preliminary_path_eligible": len(preliminary),
                "eligible_time_accurate_settings": len(eligible),
                "selected_setting": list(eligible[0]) if eligible else None,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
