from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blrpng.data.shards import SHARD_SCHEMA, validate_shard
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    sha256_json,
    write_json,
)
from blrpng.utils import atomic_torch_save


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _quarantine_empty_shard(path: Path) -> Path | None:
    """Recover only an empty crash artifact; never replace nonempty user data."""
    if not path.exists() or path.stat().st_size != 0:
        return None
    quarantine = ensure_artifact_path(
        f"artifacts/quarantine/navier_stokes_sfepy/"
        f"{path.name}.{time.time_ns()}.empty"
    )
    path.replace(quarantine)
    write_json(
        quarantine.with_suffix(quarantine.suffix + ".json"),
        {
            "reason": "zero_byte_shard_after_interrupted_process",
            "original_path": _relative(path),
            "quarantine_path": _relative(quarantine),
            "recoverable": True,
        },
    )
    return quarantine


def run_sfepy_navier_stokes_case(
    case: dict[str, Any],
    *,
    output_shard: str | Path,
    output_resolution: int,
    mesh_resolution: int,
    steps: int,
    dt: float,
    config: dict[str, Any],
    fidelity: int,
    linear_solver: str = "direct",
    time_integrator: str | None = None,
) -> dict[str, Any]:
    output = ensure_artifact_path(output_shard)
    requested_integrator = str(
        time_integrator or config["solver"].get("time_integrator", "BDF1")
    )
    requested_path_mode = str(config["solver"].get("path_selection_mode", "progress"))
    _quarantine_empty_shard(output)
    if output.exists():
        try:
            payload = torch.load(output, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(
                f"Existing nonempty NS shard is unreadable; refusing to overwrite: {output}"
            ) from exc
        validation = validate_shard(payload)
        metadata = payload["meta"]
        compatible = (
            int(payload["ids"]["case_id"]) == int(case["case_id"])
            and int(metadata["output_resolution"]) == int(output_resolution)
            and int(metadata["mesh_resolution"]) == int(mesh_resolution)
            and int(metadata["steps"]) == int(steps)
            and abs(float(metadata["dt"]) - float(dt)) <= 1e-15
            and str(metadata.get("time_integrator", "BDF1")) == requested_integrator
            and str(metadata.get("path_selection_mode", "progress")) == requested_path_mode
            and str(metadata.get("linear_solver", "direct")) == linear_solver
            and abs(float(metadata.get("residual_progress_threshold", 0.25))
                    - float(config["solver"]["residual_progress_threshold"])) <= 1e-15
            and metadata.get("case") == case
        )
        if not compatible:
            raise ValueError(f"Existing shard conflicts with requested case: {output}")
        return {
            "path": _relative(output),
            "sha256": sha256_file(output),
            "steps": validation["steps"],
            "target_task_id": int(case["target_task_id"]),
            "boundary_id": int(case["boundary_id"]),
            "initial_condition_id": int(case["initial_condition_id"]),
            "pair_id": int(case["pair_id"]),
            "role": str(case["role"]),
            "resolution": int(output_resolution),
            "solver_converged": bool(payload["meta"]["solver_converged"]),
            "valid_fraction": float(payload["valid_correction_mask"].float().mean()),
            "peak_rss_gib": float(payload["meta"].get("peak_rss_gib", 0.0)),
            "reused": True,
        }

    stem = f"ns_{int(case['case_id'])}_{output_resolution}_{os.getpid()}"
    case_path = ensure_artifact_path(f"artifacts/tmp/{stem}.case.json")
    npz_path = ensure_artifact_path(f"artifacts/tmp/{stem}.npz")
    meta_path = ensure_artifact_path(f"artifacts/tmp/{stem}.meta.json")
    log_path = ensure_artifact_path(f"artifacts/tmp/{stem}.log")
    progress_path = ensure_artifact_path(
        f"artifacts/reports/navier_stokes_sfepy_worker_progress/"
        f"case_{int(case['case_id'])}_{output_resolution}.json"
    )
    write_json(case_path, case)
    solver = config["solver"]
    command = [
        str(config["resources"]["sfepy_python"]),
        "-m",
        "blrpng.solvers.sfepy_navier_stokes_worker",
        "--case-json",
        str(case_path),
        "--output-npz",
        str(npz_path),
        "--output-meta",
        str(meta_path),
        "--output-resolution",
        str(int(output_resolution)),
        "--mesh-resolution",
        str(int(mesh_resolution)),
        "--steps",
        str(int(steps)),
        "--dt",
        str(float(dt)),
        "--progress-threshold",
        str(float(solver["residual_progress_threshold"])),
        "--path-selection-mode",
        requested_path_mode,
        "--linear-solver",
        linear_solver,
        "--max-newton-iterations",
        str(int(solver["max_newton_iterations"])),
        "--newton-atol",
        str(float(solver["newton_atol"])),
        "--newton-rtol",
        str(float(solver["newton_rtol"])),
        "--time-integrator",
        requested_integrator,
        "--max-address-space-gib",
        str(float(config["resources"].get("sfepy_worker_address_space_gib", 16.0))),
        "--progress-json",
        str(progress_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        environment[name] = str(int(config["resources"]["worker_threads"]))
    with log_path.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(config["resources"].get("sfepy_case_timeout_seconds", 600.0)),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"SfePy Navier--Stokes worker timed out for case {case['case_id']}; "
                f"diagnostic log: {log_path}"
            ) from exc
    if completed.returncode != 0:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"SfePy Navier--Stokes worker failed for case {case['case_id']}:\n{log}"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    payload = {
        "schema": SHARD_SCHEMA,
        "trajectory": torch.from_numpy(arrays["trajectory"]),
        "boundary_values": torch.from_numpy(arrays["boundary_values"]),
        "boundary_lift": torch.from_numpy(arrays["boundary_lift"]),
        "constraint_mask": torch.from_numpy(arrays["constraint_mask"]),
        "newton_correction": torch.from_numpy(arrays["newton_correction"]),
        "endpoint_increment": torch.from_numpy(arrays["endpoint_increment"]),
        "predictor_endpoint_increment": torch.from_numpy(
            arrays["predictor_endpoint_increment"]
        ),
        "residual_progress": torch.from_numpy(arrays["residual_progress"]),
        "valid_correction_mask": torch.from_numpy(arrays["valid_correction_mask"]),
        "condition_fields": torch.from_numpy(arrays["condition_fields"]),
        "accepted_newton_residuals": torch.from_numpy(
            arrays["accepted_newton_residuals"]
        ),
        "accepted_newton_counts": torch.from_numpy(arrays["accepted_newton_counts"]),
        "ids": {
            key: int(case[key])
            for key in (
                "case_id",
                "pair_id",
                "boundary_id",
                "initial_condition_id",
                "rollout_id",
                "target_task_id",
            )
        },
        "meta": {
            **metadata,
            "role": str(case["role"]),
            "resolution": int(output_resolution),
            "fidelity": int(fidelity),
            "field_names": ["u", "v", "p"],
            "field_scales": [1.0, 1.0, 0.25],
            "condition_names": metadata.get(
                "condition_names", ["viscosity", "dt", "force_x", "force_y"]
            ),
            "boundary_lift_sha256": sha256_json(
                arrays["boundary_lift"].astype(np.float32).tolist()
            ),
        },
    }
    if "path_information_mask" in arrays:
        payload["path_information_mask"] = torch.from_numpy(
            arrays["path_information_mask"]
        )
    validate_shard(payload)
    atomic_torch_save(payload, output)
    # A caller must never observe a successful return for a truncated shard.
    saved = torch.load(output, map_location="cpu", weights_only=False)
    if validate_shard(saved) != validate_shard(payload):
        raise RuntimeError(f"Post-write shard validation mismatch: {output}")
    row = {
        "path": _relative(output),
        "sha256": sha256_file(output),
        "steps": int(steps),
        "target_task_id": int(case["target_task_id"]),
        "boundary_id": int(case["boundary_id"]),
        "initial_condition_id": int(case["initial_condition_id"]),
        "pair_id": int(case["pair_id"]),
        "role": str(case["role"]),
        "resolution": int(output_resolution),
        "solver_converged": bool(metadata["solver_converged"]),
        "valid_fraction": float(np.mean(arrays["valid_correction_mask"])),
        "peak_rss_gib": float(metadata["peak_rss_gib"]),
        "reused": False,
    }
    write_json(
        progress_path,
        {
            "schema": "blrpng_sfepy_ns_worker_progress_v1",
            "case_id": int(case["case_id"]),
            "state": "shard_verified",
            "completed_steps": int(steps),
            "requested_steps": int(steps),
            "resolution": int(output_resolution),
            "mesh_resolution": int(mesh_resolution),
            "dt": float(dt),
            "time_integrator": requested_integrator,
            "peak_rss_gib": float(metadata["peak_rss_gib"]),
            "shard_path": _relative(output),
            "shard_sha256": row["sha256"],
        },
    )
    for path in (case_path, npz_path, meta_path, log_path):
        path.unlink(missing_ok=True)
    return row


def worker_case_metadata(path: str | Path) -> dict[str, Any]:
    payload = torch.load(resolve_in_project(path, must_exist=True), map_location="cpu", weights_only=False)
    validate_shard(payload)
    return payload["meta"]
