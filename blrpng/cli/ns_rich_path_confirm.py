"""Confirm the selected Newton-rich NS regime at 128² and 20 steps."""
from __future__ import annotations

import copy
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

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


def _relative(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(value - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def _make_case(protocol: dict[str, Any], task: int, setting: list[float], case_id: int) -> dict[str, Any]:
    reynolds, _dt, initial_scale, forcing_amplitude = setting
    case = copy.deepcopy(protocol["targets"][str(task)]["support"][0])
    case.update(
        {
            "case_id": case_id,
            "pair_id": case_id,
            "rollout_id": case_id,
            "role": "ns_rich_path_128_confirmation",
            "reynolds": float(reynolds),
            "initial_perturbation_scale": float(initial_scale),
        }
    )
    case["forcing"] = {
        "active": float(forcing_amplitude) > 0.0,
        "amplitude": float(forcing_amplitude),
        "kx": 2,
        "ky": 2,
        "phase": 0.37,
    }
    return case


def main() -> None:
    config = load_config("configs/navier_stokes_sfepy_seed42.yaml")
    selection_path = PROJECT_ROOT / "artifacts/reports/navier_stokes_sfepy_rich_path/prescreen.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection["selected_setting"]:
        raise PermissionError("The 64² richness/time-accuracy gate did not select a setting.")
    setting = [float(value) for value in selection["selected_setting"]]
    setting[0] = int(setting[0])
    dt = float(setting[1])
    search = config["solver"]["richness_search"]
    confirmation = search["confirmation"]
    resolution = int(confirmation["resolution"])
    mesh_resolution = int(confirmation["mesh_resolution"])
    steps = int(confirmation["steps"])
    tasks = [int(value) for value in confirmation["boundary_tasks"]]
    reference_tasks = [int(value) for value in confirmation["time_reference_tasks"]]
    integrator = str(search["time_integrator"])
    require_capacity(
        len(tasks) * 200 * 1024 * 1024,
        maximum_gb=float(config["resources"]["max_total_artifacts_gib"]),
    )
    protocol = build_navier_stokes_protocol(config)
    rows = []
    payloads: dict[int, dict[str, Any]] = {}
    refinement = []
    grid_comparison = []
    with workspace_audit("ns_rich_path_confirm"):
        for task in tasks:
            case = _make_case(protocol, task, setting, 9_400_000 + task)
            path = f"artifacts/data/navier_stokes_sfepy_rich_path_128/task{task:02d}.pt"
            row = run_sfepy_navier_stokes_case(
                case,
                output_shard=path,
                output_resolution=resolution,
                mesh_resolution=mesh_resolution,
                steps=steps,
                dt=dt,
                config=config,
                fidelity=1,
                time_integrator=integrator,
            )
            rows.append(row)
            payloads[task] = torch.load(path, map_location="cpu", weights_only=False)
            write_json(
                "artifacts/reports/navier_stokes_sfepy_rich_path/confirmation_progress.json",
                {"completed_tasks": len(payloads), "total_tasks": len(tasks), "latest_task": task},
            )
            print(json.dumps({"task": task, "completed": len(payloads), "peak_rss_gib": row["peak_rss_gib"]}), flush=True)

        for task in reference_tasks:
            base = payloads[task]
            half_case = copy.deepcopy(base["meta"]["case"])
            half_case["case_id"] += 510_000
            half_case["rollout_id"] = half_case["case_id"]
            half_path = f"artifacts/data/navier_stokes_sfepy_rich_path_128/task{task:02d}_half.pt"
            half_row = run_sfepy_navier_stokes_case(
                half_case,
                output_shard=half_path,
                output_resolution=resolution,
                mesh_resolution=mesh_resolution,
                steps=2 * steps,
                dt=0.5 * dt,
                config=config,
                fidelity=1,
                time_integrator=integrator,
            )
            rows.append(half_row)
            half = torch.load(half_path, map_location="cpu", weights_only=False)
            coarse_final = base["trajectory"][-1].double()
            reference_final = half["trajectory"][-1].double()
            refinement.append(
                {
                    "task": task,
                    "velocity_error": _relative(coarse_final[:2], reference_final[:2]),
                    "pressure_error": _relative(coarse_final[2], reference_final[2]),
                    "vorticity_error": _relative(_vorticity(coarse_final), _vorticity(reference_final)),
                }
            )

            low_case = copy.deepcopy(base["meta"]["case"])
            low_case["case_id"] += 520_000
            low_case["rollout_id"] = low_case["case_id"]
            low_path = f"artifacts/data/navier_stokes_sfepy_rich_path_128/task{task:02d}_grid64.pt"
            low_row = run_sfepy_navier_stokes_case(
                low_case,
                output_shard=low_path,
                output_resolution=64,
                mesh_resolution=33,
                steps=steps,
                dt=dt,
                config=config,
                fidelity=0,
                time_integrator=integrator,
            )
            rows.append(low_row)
            low = torch.load(low_path, map_location="cpu", weights_only=False)
            restricted = F.interpolate(
                base["trajectory"][-1:].double(), size=(64, 64), mode="bilinear", align_corners=True
            )[0]
            low_final = low["trajectory"][-1].double()
            grid_comparison.append(
                {
                    "task": task,
                    "velocity_error": _relative(low_final[:2], restricted[:2]),
                    "pressure_error": _relative(low_final[2], restricted[2]),
                    "vorticity_error": _relative(_vorticity(low_final), _vorticity(restricted)),
                }
            )

        records = []
        task_summaries = {}
        for task, payload in payloads.items():
            meta = payload["meta"]
            values = np.asarray(meta["first_update_endpoint_discrepancy"][1:])
            seconds = np.asarray(meta["second_to_first_correction_ratio"][1:])
            task_summary = {
                "boundary_family": meta["case"]["boundary"]["family"],
                "converged": bool(meta["solver_converged"]),
                "discrepancy_median": float(np.median(values)),
                "discrepancy_p25": float(np.quantile(values, 0.25)),
                "discrepancy_min": float(np.min(values)),
                "second_to_first_median": float(np.median(seconds)),
                "newton_iterations_median": float(np.median(meta["nonlinear_iterations"][1:])),
                "maximum_relative_residual": float(max(meta["relative_residuals"])),
                "maximum_weak_continuity": float(max(meta["weak_continuity"])),
                "maximum_grid_divergence_rms": float(max(meta["grid_divergence_rms"])),
                "energy_growth": float(max(meta["energy"]) / max(meta["energy"][0], 1e-30)),
                "enstrophy_growth": float(max(meta["enstrophy"]) / max(meta["enstrophy"][0], 1e-30)),
                "peak_rss_gib": float(meta["peak_rss_gib"]),
            }
            task_summaries[str(task)] = task_summary
            for step in range(1, steps):
                records.append(
                    {
                        "task": task,
                        "step": step,
                        "discrepancy": float(meta["first_update_endpoint_discrepancy"][step]),
                        "second_to_first": float(meta["second_to_first_correction_ratio"][step]),
                        "energy": float(meta["energy"][step]),
                        "enstrophy": float(meta["enstrophy"][step]),
                    }
                )
        discrepancy = np.asarray([row["discrepancy"] for row in records])
        time_gate = all(
            row["velocity_error"] <= 0.01
            and row["pressure_error"] <= 0.02
            and row["vorticity_error"] <= 0.03
            for row in refinement
        )
        grid_gate = all(
            row["velocity_error"] <= 0.02
            and row["pressure_error"] <= 0.04
            and row["vorticity_error"] <= 0.06
            for row in grid_comparison
        )
        path_gate = bool(
            np.median(discrepancy) >= 0.05
            and np.quantile(discrepancy, 0.25) >= 0.01
            and all(summary["discrepancy_median"] >= 0.03 for summary in task_summaries.values())
        )
        report = {
            "schema": "blrpng_ns_sfepy_rich_path_128_confirmation_v1",
            "source_selection": str(selection_path.relative_to(PROJECT_ROOT)),
            "source_selection_sha256": sha256_file(selection_path),
            "config_sha256": config["_config_sha256"],
            "selected_setting": setting,
            "resolution": resolution,
            "mesh_resolution": mesh_resolution,
            "steps": steps,
            "tasks": tasks,
            "aggregate_discrepancy_median": float(np.median(discrepancy)),
            "aggregate_discrepancy_p25": float(np.quantile(discrepancy, 0.25)),
            "task_summaries": task_summaries,
            "time_refinement": refinement,
            "grid_comparison_64_to_128": grid_comparison,
            "path_gate_passed": path_gate,
            "time_gate_passed": time_gate,
            "grid_gate_passed": grid_gate,
            "confirmation_passed": bool(path_gate and time_gate and grid_gate),
            "pointwise_divergence_gate_passed": all(
                summary["maximum_grid_divergence_rms"] <= 1e-3
                for summary in task_summaries.values()
            ),
            "decision": (
                "freeze_rich_regime_and_redesign_formal_NS_protocol"
                if path_gate and time_gate and grid_gate
                else "revise_setting_before_formal_data"
            ),
            "records": records,
            "shards": rows,
        }
        report_path = write_json(
            "artifacts/reports/navier_stokes_sfepy_rich_path/confirmation_128.json", report
        )

        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        for task in tasks:
            selected = [row for row in records if row["task"] == task]
            axes[0].plot(
                [row["step"] for row in selected],
                [100 * row["discrepancy"] for row in selected],
                label=f"task {task}",
            )
            axes[1].plot(
                [row["step"] for row in selected],
                [100 * row["second_to_first"] for row in selected],
            )
            axes[2].plot(
                [row["step"] for row in selected],
                [row["enstrophy"] for row in selected],
            )
        axes[0].axhline(5.0, color="#D55E00", linestyle=":")
        axes[0].set_ylabel("first update vs endpoint (%)")
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        axes[1].set_ylabel("second/first correction (%)")
        axes[2].set_ylabel("enstrophy")
        for axis in axes:
            axis.set_xlabel("BDF2 rollout step")
            axis.grid(alpha=0.2)
        figure.suptitle("128² long-rollout confirmation of non-redundant Newton paths")
        figure_path = ensure_artifact_path(
            "artifacts/figures/navier_stokes_sfepy_rich_path/02_confirmation_128.png"
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
                "path_gate_passed": report["path_gate_passed"],
                "time_gate_passed": report["time_gate_passed"],
                "grid_gate_passed": report["grid_gate_passed"],
                "pointwise_divergence_gate_passed": report[
                    "pointwise_divergence_gate_passed"
                ],
                "confirmation_passed": report["confirmation_passed"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
