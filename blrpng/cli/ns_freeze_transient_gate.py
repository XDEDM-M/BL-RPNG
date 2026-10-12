"""Freeze the bounded transient-NS information gate and candidate protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


REPORT_ROOT = "artifacts/reports/navier_stokes_sfepy_transient_gate"
REPORTS = {
    "dt_primary_64": "bdf1_dt_primary.json",
    "re400_64": "bdf1_dt006_re400.json",
    "re1200_64": "bdf1_dt006_re1200.json",
    "re1600_64": "bdf1_dt006_re1600.json",
    "dt006_128": "bdf1_dt006_re800_confirm128.json",
    "dt007_64": "bdf1_dt007_candidate64.json",
    "dt007_128": "bdf1_dt007_confirm128.json",
    "no_force_64": "bdf1_dt007_no_force64.json",
    "static_long32_64": "bdf1_dt007_long32_64.json",
    "amplitude_periodic_long32_64": "bdf1_dt007_periodic_long32_64.json",
    "moving_force_long32_64": "bdf1_dt006_moving_force_long32_64.json",
}


def _load_report(name: str) -> tuple[dict, dict]:
    path = resolve_in_project(f"{REPORT_ROOT}/{REPORTS[name]}", must_exist=True)
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)
    }


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    spacing = 1.0 / float(field.shape[-1] - 1)
    return torch.gradient(field[:, 1], spacing=(spacing,), dim=(1,))[0] - torch.gradient(
        field[:, 0], spacing=(spacing,), dim=(2,)
    )[0]


def _relative(value: torch.Tensor, reference: torch.Tensor) -> list[float]:
    return (
        (value - reference).flatten(1).norm(dim=1)
        / reference.flatten(1).norm(dim=1).clamp_min(1e-30)
    ).tolist()


def _cross_resolution() -> list[dict]:
    rows = []
    for task in (0, 6):
        for initial in (0, 1):
            coarse_path = resolve_in_project(
                f"artifacts/data/navier_stokes_sfepy_transient_gate/"
                f"bdf1_dt007_candidate64/dt0.07_task{task:02d}_initial{initial}.pt",
                must_exist=True,
            )
            fine_path = resolve_in_project(
                f"artifacts/data/navier_stokes_sfepy_transient_gate/"
                f"bdf1_dt007_confirm128/dt0.07_task{task:02d}_initial{initial}.pt",
                must_exist=True,
            )
            coarse = torch.load(coarse_path, map_location="cpu", weights_only=False)["trajectory"].double()
            fine = torch.load(fine_path, map_location="cpu", weights_only=False)["trajectory"].double()
            upsampled = functional.interpolate(
                coarse, size=fine.shape[-2:], mode="bilinear", align_corners=True
            )
            rows.append({
                "task": task, "initial_index": initial,
                "coarse_path": str(coarse_path.relative_to(PROJECT_ROOT)),
                "coarse_sha256": sha256_file(coarse_path),
                "fine_path": str(fine_path.relative_to(PROJECT_ROOT)),
                "fine_sha256": sha256_file(fine_path),
                "velocity_rel_l2_by_step": _relative(upsampled[:, :2], fine[:, :2]),
                # Step zero pressure is identically zero, so exclude its undefined ratio.
                "pressure_rel_l2_by_step": _relative(upsampled[1:, 2:3], fine[1:, 2:3]),
                "vorticity_rel_l2_by_step": _relative(
                    _vorticity(upsampled), _vorticity(fine)
                ),
            })
    return rows


def _active_fraction(report: dict, horizon: int) -> dict:
    rows = []
    for record in report["records"]:
        shard = torch.load(resolve_in_project(record["path"], must_exist=True), map_location="cpu", weights_only=False)
        correction = shard["newton_correction"][:horizon, 2].double().flatten(1)
        endpoint = shard["predictor_endpoint_increment"][:horizon, 2].double().flatten(1)
        state = shard["trajectory"][1:horizon + 1, 2].double().flatten(1)
        endpoint_norm = endpoint.norm(dim=1).clamp_min(1e-30)
        dot = (correction * endpoint).sum(dim=1)
        discrepancy = (correction - endpoint).norm(dim=1) / endpoint_norm
        orthogonal = (
            correction - (dot / endpoint_norm.square())[:, None] * endpoint
        ).norm(dim=1) / endpoint_norm
        fraction = endpoint_norm / state.norm(dim=1).clamp_min(1e-30)
        active = (discrepancy >= 0.05) & (orthogonal >= 0.03) & (fraction >= 0.02)
        rows.append({
            "task": record["task"], "horizon": horizon,
            "active_steps": int(active.sum()), "total_steps": int(active.numel()),
            "active_fraction": float(active.float().mean()),
            "discrepancy_median": float(discrepancy.median()),
            "orthogonal_fraction_median": float(orthogonal.median()),
        })
    return {
        "rows": rows,
        "minimum_task_active_fraction": min(row["active_fraction"] for row in rows),
        "pooled_active_fraction": sum(row["active_steps"] for row in rows)
        / sum(row["total_steps"] for row in rows),
    }


def main() -> None:
    loaded = {}
    sources = {}
    for name in REPORTS:
        loaded[name], sources[name] = _load_report(name)
    cross_resolution = _cross_resolution()
    spatial_summary = {
        "velocity_final_max": max(row["velocity_rel_l2_by_step"][-1] for row in cross_resolution),
        "velocity_all_steps_max": max(max(row["velocity_rel_l2_by_step"]) for row in cross_resolution),
        "pressure_final_max": max(row["pressure_rel_l2_by_step"][-1] for row in cross_resolution),
        "vorticity_final_max": max(row["vorticity_rel_l2_by_step"][-1] for row in cross_resolution),
    }
    dt007_64 = loaded["dt007_64"]["summary"]["0.07"]
    dt007_128 = loaded["dt007_128"]["summary"]["0.07"]
    static20 = _active_fraction(loaded["static_long32_64"], 20)
    static32 = _active_fraction(loaded["static_long32_64"], 32)
    moving32 = _active_fraction(loaded["moving_force_long32_64"], 32)
    bounded_gate = bool(
        dt007_64["eligible"] and dt007_128["eligible"]
        and static20["minimum_task_active_fraction"] >= 0.30
        and loaded["static_long32_64"]["summary"]["0.07"]["time_accuracy_gate"]
    )
    report = {
        "schema": "blrpng_ns_transient_gate_freeze_v1",
        "status": "bounded_20_step_numerical_gate_passed" if bounded_gate else "gate_failed",
        "trajectory_axis": "physical_time",
        "sources": sources,
        "candidate": {
            "time_integrator": "BDF1", "dt": 0.07, "reynolds_center": 800,
            "newton_predictor": "previous_physical_state_without_degradation",
            "forcing": {
                "type": "static_divergence_free_body_force", "amplitude": 0.05,
                "spatial_wavenumbers": [2, 2], "phase": 0.37,
            },
            "grad_div_gamma": 100.0,
            "path_target": "first_accepted_Newton_state_displacement",
            "matched_control": "same_predictor_to_endpoint_increment",
            "path_information_mask": {
                "pressure_endpoint_discrepancy_min": 0.05,
                "pressure_orthogonal_fraction_min": 0.03,
                "pressure_endpoint_fraction_of_state_min": 0.02,
                "residual_progress_max": 0.8,
            },
        },
        "candidate_64_summary": dt007_64,
        "candidate_128_summary": dt007_128,
        "static_force_active_fraction_20_steps": static20,
        "static_force_active_fraction_32_steps": static32,
        "moving_force_active_fraction_32_steps": moving32,
        "cross_resolution_64_to_128": cross_resolution,
        "cross_resolution_summary": spatial_summary,
        "bounded_20_step_numerical_gate_passed": bounded_gate,
        "verified_conclusions": [
            "Natural BDF1 velocity Newton paths remain nearly endpoint-redundant in the tested regime.",
            "At dt=0.07 and Re=800, first-accepted pressure paths are non-collinear with matched endpoints on both 64 and 128 grids.",
            "The dt/2 time-accuracy gate passes for the tested 8-step 64/128 rollouts and the tested 32-step 64 rollouts.",
            "Static divergence-free forcing supplies useful early/transient pressure paths, but information decays as a rollout approaches a quasi-steady state.",
            "Increasing Reynolds number from 800 to 1600 gives negligible extra path information and is not justified for this purpose.",
            "The old progress-0.25 transient labels selected an almost-final state and are superseded for mechanism claims.",
        ],
        "limitations": [
            "The 20-step gate is based on two boundary families and two initial conditions at 64/128 resolution.",
            "Pressure has substantially larger 64-to-128 discretization discrepancy than velocity; low-resolution pressure paths must not supervise the main model.",
            "Path information is field-wise and intermittent; it is strong in pressure but not in velocity.",
            "No neural operator has yet been trained on this candidate protocol.",
            "A 256-grid transient generation benchmark and path confirmation are still required before formal source generation.",
        ],
        "formal_candidate_protocol": {
            "development_gate": {
                "resolution": 128, "train_rollouts": 24, "validation_rollouts": 8,
                "steps": 20, "reynolds": [600, 800, 1000], "dt": 0.07,
            },
            "source_pretrain": {
                "resolution": 256, "train_rollouts": 96, "validation_rollouts": 24,
                "steps": 20, "boundary_frequencies": [1, 2, 3, 4],
                "boundary_amplitude_range": [0.75, 1.25],
                "reynolds": [600, 800, 1000],
                "forcing_amplitude_range": [0.04, 0.06],
                "initial_perturbation_scale_range": [0.10, 0.25],
            },
            "target_support": {
                "resolution": 64, "budgets": [1, 2, 4, 8], "steps": 20,
                "adaptation_loss": "final_only_NLT", "use_low_resolution_path": False,
            },
            "target_query": {
                "resolutions": [128, 256], "initial_conditions_per_task": 8,
                "steps": 50, "blind": True,
            },
            "causal_controls": [
                "ordinary_final_pretrain", "matched_endpoint_pretrain",
                "first_accepted_Newton_pretrain", "pressure_only_Newton_pretrain",
            ],
            "fairness": (
                "Newton and endpoint controls use the identical path-information mask, "
                "initialization, batches, optimizer steps, and checkpoint rule."
            ),
        },
        "neural_hypotheses_not_yet_verified": [
            "Masked pressure-path pretraining improves unseen-boundary LR-to-HR adaptation over the matched endpoint control.",
            "The high-resolution pressure-path representation transfers despite low-resolution pressure discretization error.",
            "The method remains stable and advantageous over 50-step autoregressive rollout.",
        ],
    }
    with workspace_audit("ns_freeze_transient_gate"):
        output = write_json(f"{REPORT_ROOT}/frozen_gate_and_protocol.json", report)
    print(json.dumps({
        "report": str(output), "sha256": sha256_file(output),
        "bounded_20_step_numerical_gate_passed": bounded_gate,
        "model_effect_not_yet_tested": True,
    }, indent=2))


if __name__ == "__main__":
    main()
