"""Render an existing SfePy Navier--Stokes rollout as a time-evolution PNG."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm

from blrpng.data.shards import validate_shard
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="artifacts/data/navier_stokes_sfepy_post_crash/long20_128_bdf3.pt",
    )
    parser.add_argument(
        "--output",
        default="artifacts/figures/navier_stokes_sfepy/task00_bdf3_rollout_evolution.png",
    )
    args = parser.parse_args()
    source = resolve_in_project(args.input, must_exist=True)
    output = ensure_artifact_path(args.output)
    if output.suffix.lower() != ".png":
        raise ValueError("This scientific-figure CLI writes PNG only")
    shard = torch.load(source, map_location="cpu", weights_only=False)
    validate_shard(shard)
    fields = shard["trajectory"].detach().cpu().numpy().astype(np.float64)
    if fields.ndim != 4 or fields.shape[1] != 3 or fields.shape[-1] != fields.shape[-2]:
        raise ValueError("Expected NS trajectory [T+1,3,H,W] on a square grid")
    steps = fields.shape[0] - 1
    indices = sorted(set((0, 1, steps // 4, steps // 2, 3 * steps // 4, steps)))
    if len(indices) < 4:
        raise ValueError("At least four distinct rollout times are required")
    dt = float(shard["meta"]["dt"])
    resolution = fields.shape[-1]
    spacing = 1.0 / (resolution - 1)
    selected = fields[indices]
    u, v, pressure = selected[:, 0], selected[:, 1], selected[:, 2]
    speed = np.hypot(u, v)
    vorticity = np.gradient(v, spacing, axis=1) - np.gradient(u, spacing, axis=2)
    divergence = np.gradient(u, spacing, axis=1) + np.gradient(v, spacing, axis=2)
    row_data = (speed, pressure, vorticity, divergence)
    row_names = (
        r"Speed $|\mathbf{u}|$ + streamlines",
        r"Pressure $p$ (zero-mean gauge)",
        r"Vorticity $\omega=\partial_x v-\partial_y u$",
        r"Grid divergence $\nabla\cdot\mathbf{u}$ (diagnostic)",
    )
    signed_limits = [
        None,
        max(0.1, float(np.quantile(np.abs(pressure), 0.995))),
        max(1.0, float(np.quantile(np.abs(vorticity), 0.995))),
        max(0.1, float(np.quantile(np.abs(divergence), 0.995))),
    ]
    norms = (
        Normalize(vmin=0, vmax=float(np.max(speed))),
        TwoSlopeNorm(vmin=-signed_limits[1], vcenter=0, vmax=signed_limits[1]),
        SymLogNorm(linthresh=2.0, vmin=-signed_limits[2], vmax=signed_limits[2]),
        SymLogNorm(linthresh=0.01, vmin=-signed_limits[3], vmax=signed_limits[3]),
    )
    cmaps = ("viridis", "coolwarm", "RdBu_r", "PuOr_r")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig = plt.figure(figsize=(18.5, 12), layout="constrained")
    grid = fig.add_gridspec(4, len(indices) + 1, width_ratios=[1] * len(indices) + [0.045])
    axis = np.linspace(0, 1, resolution)
    for row, data in enumerate(row_data):
        for column, step in enumerate(indices):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(
                data[column].T,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmaps[row],
                norm=norms[row],
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                ax.streamplot(
                    axis, axis, u[column].T, v[column].T,
                    density=0.65, color=(1, 1, 1, 0.68), linewidth=0.35,
                    arrowsize=0.48,
                )
                ax.set_title(f"step {step}  ·  t={step * dt:.2f}")
            if column == 0:
                ax.set_ylabel(row_names[row], labelpad=9)
                ax.set_yticks((0, 0.5, 1))
            else:
                ax.set_yticks(())
            if row == len(row_data) - 1:
                ax.set_xlabel("x")
                ax.set_xticks((0, 0.5, 1))
            else:
                ax.set_xticks(())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        cax = fig.add_subplot(grid[row, -1])
        fig.colorbar(image, cax=cax)
    fig.suptitle(
        "SfePy Navier–Stokes pilot: nonuniform lid-driven cavity  "
        f"·  task {shard['ids']['target_task_id']}  ·  Re={shard['meta']['reynolds']:g}  "
        f"·  BDF3  ·  Δt={dt:g}  ·  {resolution}² grid\n"
        "Numerical prescreen only — projected-grid divergence gate not passed",
        fontsize=14,
    )
    fig.savefig(output, dpi=300, format="png", metadata={"Software": "BL-RPNG"})
    plt.close(fig)
    manifest = {
        "schema": "blrpng_ns_pilot_rollout_visualization_v1",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "time_step_indices": indices,
        "times": [round(step * dt, 12) for step in indices],
        "shared_color_scale_per_row": True,
        "signed_color_limits_are_99_5_percentile_clipped": True,
        "physical_status": "numerical pilot; strong grid-divergence gate not passed",
    }
    write_json(output.with_suffix(".json"), manifest)
    print(json.dumps({"output": str(output), "sha256": manifest["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
