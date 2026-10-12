"""Read-only numerical audit of the bounded post-crash BDF3 NS pilots."""
from __future__ import annotations

import json

import numpy as np
import torch

from blrpng.data.shards import validate_shard
from blrpng.paths import resolve_in_project, sha256_file, write_json


ROOT = "artifacts/data/navier_stokes_sfepy_post_crash"
PAIRS = (
    (0, "long20_128_bdf3.pt", "half40_128_bdf3.pt"),
    (6, "task06_long20_128_bdf3.pt", "task06_half40_128_bdf3.pt"),
)
GRAD_DIV_64 = (
    (0.0, "grad_div_gamma0_control64.pt"),
    (1.0, "grad_div_gamma1_pilot64.pt"),
    (10.0, "grad_div_gamma10_pilot64.pt"),
)


def _load(name: str) -> tuple[dict, str]:
    path = resolve_in_project(f"{ROOT}/{name}", must_exist=True)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    validate_shard(shard)
    return shard, sha256_file(path)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    h = 1.0 / (field.shape[-1] - 1)
    return torch.gradient(field[1], spacing=(h,), dim=(0,))[0] - torch.gradient(
        field[0], spacing=(h,), dim=(1,)
    )[0]


def _pair(task: int, base_name: str, half_name: str) -> dict:
    base, base_hash = _load(base_name)
    half, half_hash = _load(half_name)
    if (
        base["ids"]["pair_id"] != half["ids"]["pair_id"]
        or base["ids"]["initial_condition_id"] != half["ids"]["initial_condition_id"]
        or base["ids"]["boundary_id"] != half["ids"]["boundary_id"]
    ):
        raise ValueError(f"Task {task}: dt/dt2 pair does not match")
    if base["meta"]["steps"] != 20 or half["meta"]["steps"] != 40:
        raise ValueError(f"Task {task}: unexpected rollout lengths")
    if abs(base["meta"]["dt"] - 2 * half["meta"]["dt"]) > 1e-14:
        raise ValueError(f"Task {task}: inconsistent physical time")
    a, b = base["trajectory"][-1].double(), half["trajectory"][-1].double()
    metadata = base["meta"]
    discrepancy = np.asarray(metadata["first_update_endpoint_discrepancy"][2:])
    selected = base["newton_correction"][2:].double().numpy()
    endpoint = base["endpoint_increment"][2:].double().numpy()
    def discrepancy_ratio(channel: slice) -> np.ndarray:
        path = selected[:, channel].reshape(len(selected), -1)
        target = endpoint[:, channel].reshape(len(endpoint), -1)
        return np.linalg.norm(path - target, axis=1) / np.maximum(
            np.linalg.norm(target, axis=1), 1e-30
        )

    selected_velocity = discrepancy_ratio(slice(0, 2))
    selected_pressure = discrepancy_ratio(slice(2, 3))
    record = {
        "task_id": task,
        "base_path": f"{ROOT}/{base_name}",
        "base_sha256": base_hash,
        "half_path": f"{ROOT}/{half_name}",
        "half_sha256": half_hash,
        "velocity_time_error": _relative(a[:2], b[:2]),
        "pressure_time_error": _relative(a[2], b[2]),
        "vorticity_time_error": _relative(_vorticity(a), _vorticity(b)),
        "first_update_discrepancy_median": float(np.median(discrepancy)),
        "first_update_discrepancy_p25": float(np.quantile(discrepancy, 0.25)),
        "selected_velocity_discrepancy_median": float(np.median(selected_velocity)),
        "selected_velocity_discrepancy_p25": float(np.quantile(selected_velocity, 0.25)),
        "selected_pressure_discrepancy_median": float(np.median(selected_pressure)),
        "selected_progress_median": float(np.median(base["residual_progress"][2:].numpy())),
        "median_newton_iterations": float(np.median(metadata["nonlinear_iterations"][2:])),
        "max_relative_residual": max(metadata["relative_residuals"]),
        "max_grid_divergence_rms": max(metadata["grid_divergence_rms"]),
        "max_weak_continuity": max(metadata["weak_continuity"]),
        "max_net_flux": max(abs(value) for value in metadata["net_flux"]),
        "boundary_error": metadata["boundary_error"],
        "peak_rss_gib": max(metadata["peak_rss_gib"], half["meta"]["peak_rss_gib"]),
        "converged": bool(metadata["solver_converged"] and half["meta"]["solver_converged"]),
    }
    record["path_gate"] = bool(
        record["selected_velocity_discrepancy_median"] >= 0.05
        and record["selected_velocity_discrepancy_p25"] >= 0.01
        and record["median_newton_iterations"] >= 3
    )
    record["time_gate"] = bool(
        record["velocity_time_error"] <= 0.01
        and record["pressure_time_error"] <= 0.02
        and record["vorticity_time_error"] <= 0.03
    )
    record["strong_divergence_gate"] = bool(record["max_grid_divergence_rms"] <= 1e-3)
    return record


def main() -> None:
    records = [_pair(*entry) for entry in PAIRS]
    stabilization = []
    for gamma, name in GRAD_DIV_64:
        shard, checksum = _load(name)
        metadata = shard["meta"]
        stabilization.append(
            {
                "grad_div_gamma": gamma,
                "path": f"{ROOT}/{name}",
                "sha256": checksum,
                "steps": metadata["steps"],
                "grid_divergence_rms_max": max(metadata["grid_divergence_rms"]),
                "solver_converged": metadata["solver_converged"],
                "peak_rss_gib": metadata["peak_rss_gib"],
            }
        )
    report = {
        "schema": "blrpng_ns_post_crash_bounded_audit_v1",
        "purpose": "numerical prescreen only; no neural-network training or HR query opened",
        "records": records,
        "grad_div_64_short_pilot": stabilization,
        "all_path_gates": all(row["path_gate"] for row in records),
        "all_time_gates": all(row["time_gate"] for row in records),
        "all_strong_divergence_gates": all(row["strong_divergence_gate"] for row in records),
        "formal_data_allowed": all(
            row["path_gate"] and row["time_gate"] and row["strong_divergence_gate"]
            for row in records
        ),
        "incident_note": (
            "The prior host restart has no persistent kernel journal. A zero-byte shard "
            "was quarantined; its cause is not established. These pilots used one worker, "
            "a 16-GiB worker address-space cap and a 180-second per-case timeout."
        ),
        "mechanism_note": (
            "The earlier first-update/endpoint discrepancy is not the supervised "
            "progress-0.25 correction. The selected correction is nearly the "
            "endpoint increment, so the current NS path labels fail the mechanism gate."
        ),
    }
    output = write_json(
        "artifacts/reports/navier_stokes_sfepy_rich_path/post_crash_bdf3_audit.json",
        report,
    )
    print(json.dumps({"report": str(output), **{key: report[key] for key in (
        "all_path_gates", "all_time_gates", "all_strong_divergence_gates", "formal_data_allowed"
    )}}, indent=2))


if __name__ == "__main__":
    main()
