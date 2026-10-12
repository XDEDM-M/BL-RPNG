"""Bounded check that steady-NS Newton richness survives natural warm starts."""
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import build_navier_stokes_protocol
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.paths import resolve_in_project, sha256_file, write_json


def _summary(path: str, blend: float) -> dict:
    artifact = resolve_in_project(path, must_exist=True)
    shard = torch.load(artifact, map_location="cpu", weights_only=False)
    metadata = shard["meta"]
    ratios = np.asarray(metadata["selected_velocity_predictor_endpoint_discrepancy"], dtype=float)
    return {
        "predictor_blend": blend,
        "resolution": int(shard["trajectory"].shape[-1]),
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "per_reynolds_discrepancy": ratios.tolist(),
        "median_discrepancy": float(np.median(ratios)),
        "p25_discrepancy": float(np.quantile(ratios, 0.25)),
        "newton_iterations": metadata["nonlinear_iterations"],
        "max_relative_residual": max(metadata["relative_residuals"]),
        "max_native_fe_divergence_rms": max(metadata["fe_divergence_qp_rms"]),
        "solver_converged": bool(metadata["solver_converged"]),
    }


def main() -> None:
    config = load_config("configs/navier_stokes_sfepy_seed42.yaml")
    config["solver"]["path_selection_mode"] = "first_accepted"
    config["resources"]["max_generation_workers"] = 1
    config["resources"]["sfepy_worker_address_space_gib"] = 8.0
    config["resources"]["sfepy_case_timeout_seconds"] = 180.0
    protocol = build_navier_stokes_protocol(config)
    rows = []
    with workspace_audit("ns_steady_predictor_sensitivity"):
        require_capacity(estimated_new_bytes=250_000_000, maximum_gb=300.0)
        for resolution in (64, 128):
            case = copy.deepcopy(protocol["targets"]["0"]["support"][0])
            case.update({
                "case_id": 99092300 + resolution,
                "pair_id": 99092300,
                "rollout_id": 99092300 + resolution,
                "role": "steady_natural_warm_start_sensitivity",
                "reynolds_schedule": [50, 100, 200, 400],
                "newton_predictor_blend": 1.0,
                "grad_div_gamma": 100.0,
                "forcing": {"active": False},
            })
            path = f"artifacts/data/navier_stokes_sfepy_steady_pilot/task00_warmstart_{resolution}.pt"
            run_sfepy_navier_stokes_case(
                case,
                output_shard=path,
                output_resolution=resolution,
                mesh_resolution=resolution // 2 + 1,
                steps=4,
                dt=1.0,
                config=config,
                fidelity={64: 0, 128: 1}[resolution],
                time_integrator="STEADY_CONTINUATION",
            )
            rows.append(_summary(path, 1.0))
            rows.append(_summary(
                f"artifacts/data/navier_stokes_sfepy_steady_projected/task00_re400_{resolution}.pt",
                0.5,
            ))
    report = {
        "schema": "blrpng_ns_steady_predictor_sensitivity_v1",
        "equation": "steady_2d_incompressible_navier_stokes",
        "trajectory_axis": "Reynolds_parameter_not_physical_time",
        "task": 0,
        "matched_control": "predictor_endpoint_increment",
        "rows": rows,
        "natural_warm_start_richness_passed": all(
            row["solver_converged"]
            and row["max_relative_residual"] <= 1e-8
            and row["median_discrepancy"] >= 0.05
            for row in rows if row["predictor_blend"] == 1.0
        ),
    }
    output = write_json(
        "artifacts/reports/navier_stokes_sfepy_steady_pilot/predictor_sensitivity.json",
        report,
    )
    print(json.dumps({"report": str(output), "sha256": sha256_file(output),
                      "natural_warm_start_richness_passed": report["natural_warm_start_richness_passed"]},
                     indent=2))


if __name__ == "__main__":
    main()
