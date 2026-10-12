"""Audited no-slip-preserving grid projection for SfePy NS shards."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from blrpng.analysis.ns_grid_projection import grid_divergence, project_no_slip_divergence
from blrpng.data.shards import validate_shard
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file
from blrpng.utils import atomic_torch_save


def project_steady_shard(source_path: str | Path, output_path: str | Path) -> dict:
    source = resolve_in_project(source_path, must_exist=True)
    output = ensure_artifact_path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite projected NS shard: {output}")
    original = torch.load(source, map_location="cpu", weights_only=False)
    validate_shard(original)
    if original["meta"].get("time_integrator") != "STEADY_CONTINUATION":
        raise ValueError("Grid projection is restricted to audited steady NS continuation shards")
    if original["meta"].get("grid_projection"):
        raise ValueError("This NS shard has already been projected")
    projected = copy.deepcopy(original)
    blend = float(original["meta"]["newton_predictor_blend"])
    state_changes: list[float] = []
    projected_states = []
    for state in original["trajectory"]:
        velocity, diagnostic = project_no_slip_divergence(state[:2].numpy())
        changed = state.numpy().astype(np.float64).copy()
        changed[:2] = velocity
        projected_states.append(changed)
        state_changes.append(float(diagnostic["relative_velocity_change"]))
    projected_states = np.stack(projected_states)
    lift = original["boundary_lift"].numpy().astype(np.float64).copy()
    lift_velocity, lift_diagnostic = project_no_slip_divergence(lift[:2])
    lift[:2] = lift_velocity
    projected_corrections = []
    correction_changes: list[float] = []
    for correction in original["newton_correction"]:
        velocity, diagnostic = project_no_slip_divergence(correction[:2].numpy())
        changed = correction.numpy().astype(np.float64).copy()
        changed[:2] = velocity
        projected_corrections.append(changed)
        correction_changes.append(float(diagnostic["relative_velocity_change"]))
    projected_corrections = np.stack(projected_corrections)
    predictor_endpoint = original["predictor_endpoint_increment"].numpy().astype(np.float64).copy()
    for step in range(len(predictor_endpoint)):
        predictor_velocity = lift_velocity + blend * (projected_states[step, :2] - lift_velocity)
        predictor_endpoint[step, :2] = projected_states[step + 1, :2] - predictor_velocity
    physical_endpoint = projected_states[1:] - projected_states[:-1]
    output_divergence = [
        float(np.sqrt(np.mean(grid_divergence(state[:2]) ** 2)))
        for state in projected_states[1:]
    ]
    # A projection that makes a large change would no longer be a faithful
    # representation of the SfePy solution; fail instead of masking it.
    maximum_change = max([*state_changes, *correction_changes, float(lift_diagnostic["relative_velocity_change"])])
    if maximum_change > 0.01 or max(output_divergence) > 1e-4:
        raise ValueError(
            f"NS projection fidelity gate failed: change={maximum_change:.3g}, "
            f"divergence={max(output_divergence):.3g}"
        )
    projected["trajectory"] = torch.from_numpy(projected_states.astype(np.float32))
    projected["boundary_lift"] = torch.from_numpy(lift.astype(np.float32))
    projected["newton_correction"] = torch.from_numpy(projected_corrections.astype(np.float32))
    projected["endpoint_increment"] = torch.from_numpy(physical_endpoint.astype(np.float32))
    projected["predictor_endpoint_increment"] = torch.from_numpy(predictor_endpoint.astype(np.float32))
    selected_velocity_discrepancy = []
    for step in range(len(predictor_endpoint)):
        correction = projected_corrections[step, :2]
        reference = predictor_endpoint[step, :2]
        selected_velocity_discrepancy.append(
            float(np.linalg.norm(correction - reference)) / max(float(np.linalg.norm(reference)), 1e-30)
        )
    metadata = projected["meta"]
    metadata["native_grid_divergence_rms"] = metadata["grid_divergence_rms"]
    metadata["grid_divergence_rms"] = output_divergence
    metadata["selected_velocity_predictor_endpoint_discrepancy"] = selected_velocity_discrepancy
    metadata["grid_projection"] = {
        "method": "least_L2_interior_correction_central_difference_divergence_v1",
        "source_sha256": sha256_file(source),
        "max_state_relative_velocity_change": max(state_changes),
        "max_correction_relative_velocity_change": max(correction_changes),
        "lift_relative_velocity_change": float(lift_diagnostic["relative_velocity_change"]),
        "max_divergence_rms_after": max(output_divergence),
        "preserves_dirichlet_boundary": True,
        "native_fe_qp_divergence_preserved_in_metadata": True,
    }
    validate_shard(projected)
    atomic_torch_save(projected, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=False)
    validate_shard(reloaded)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "max_relative_velocity_change": maximum_change,
        "max_divergence_rms_after": max(output_divergence),
        "selected_velocity_discrepancy": selected_velocity_discrepancy,
    }
