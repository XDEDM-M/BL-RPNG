"""Visualize accepted transient SfePy-NS gate samples over physical time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import validate_shard
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    write_json,
)


DEFAULT_INPUTS = (
    "artifacts/data/navier_stokes_sfepy_transient_gate/"
    "bdf1_dt007_long32_64/dt0.07_task00_initial0.pt",
    "artifacts/data/navier_stokes_sfepy_transient_gate/"
    "bdf1_dt007_long32_64/dt0.07_task06_initial0.pt",
)


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64, copy=False)


def _signed_limit(values: np.ndarray, quantile: float = 0.995) -> float:
    limit = float(np.quantile(np.abs(values), quantile))
    return max(limit, np.finfo(np.float64).eps)


def _vorticity(trajectory: np.ndarray) -> np.ndarray:
    spacing = 1.0 / float(trajectory.shape[-1] - 1)
    u = trajectory[:, 0]
    v = trajectory[:, 1]
    return np.gradient(v, spacing, axis=1) - np.gradient(u, spacing, axis=2)


def _save(
    figure: plt.Figure,
    output: Path,
    *,
    source: Path | list[Path],
    payload: dict[str, Any],
) -> dict[str, Any]:
    figure.savefig(
        output,
        dpi=300,
        format="png",
        bbox_inches="tight",
        metadata={"Software": "BL-RPNG", "Title": figure._suptitle.get_text() if figure._suptitle else output.stem},
    )
    plt.close(figure)
    sources = [source] if isinstance(source, Path) else source
    record = {
        "schema": "blrpng_ns_transient_sample_figure_v1",
        "output": str(output.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output),
        "sources": [
            {
                "path": str(item.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(item),
            }
            for item in sources
        ],
        "checkpoint_selection_performed": False,
        "model_prediction_shown": False,
        **payload,
    }
    write_json(output.with_suffix(".json"), record)
    return record


def _evolution_figure(shard: dict[str, Any], source: Path, output_dir: Path) -> dict[str, Any]:
    trajectory = _as_numpy(shard["trajectory"])
    total_steps = trajectory.shape[0] - 1
    requested = (0, 1, 4, 8, 16, 24, total_steps)
    indices = sorted({min(total_steps, value) for value in requested})
    selected = trajectory[indices]
    u, v, pressure = selected[:, 0], selected[:, 1], selected[:, 2]
    speed = np.hypot(u, v)
    vorticity = _vorticity(selected)
    speed_change = np.hypot(u - u[0], v - v[0])
    fields = (speed, pressure, vorticity, speed_change)
    labels = (
        r"Speed $|\mathbf{u}|$ + streamlines",
        r"Pressure $p$ (zero-mean gauge)",
        r"Vorticity $\omega=\partial_x v-\partial_y u$",
        r"Velocity change $|\mathbf{u}(t)-\mathbf{u}(0)|$",
    )
    speed_max = max(float(np.quantile(speed, 0.999)), np.finfo(float).eps)
    change_max = max(float(np.quantile(speed_change, 0.999)), np.finfo(float).eps)
    p_limit = _signed_limit(pressure)
    omega_limit = _signed_limit(vorticity)
    norms = (
        Normalize(0.0, speed_max),
        TwoSlopeNorm(vmin=-p_limit, vcenter=0.0, vmax=p_limit),
        TwoSlopeNorm(vmin=-omega_limit, vcenter=0.0, vmax=omega_limit),
        Normalize(0.0, change_max),
    )
    cmaps = ("viridis", "RdBu_r", "RdBu_r", "magma")
    dt = float(shard["meta"]["dt"])
    resolution = int(trajectory.shape[-1])
    task = int(shard["ids"]["target_task_id"])
    family = str(shard["meta"]["case"]["boundary"]["family"])
    reynolds = float(shard["meta"]["reynolds"])
    axis = np.linspace(0.0, 1.0, resolution)

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
        }
    )
    figure = plt.figure(figsize=(19.5, 10.8), layout="constrained")
    grid = figure.add_gridspec(
        len(fields), len(indices) + 1,
        width_ratios=[1.0] * len(indices) + [0.045],
    )
    for row, field in enumerate(fields):
        last_image = None
        for column, step in enumerate(indices):
            panel = figure.add_subplot(grid[row, column])
            last_image = panel.imshow(
                field[column].T,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=cmaps[row],
                norm=norms[row],
                interpolation="bilinear",
                aspect="equal",
            )
            if row == 0:
                panel.streamplot(
                    axis,
                    axis,
                    u[column].T,
                    v[column].T,
                    density=0.72,
                    color=(1.0, 1.0, 1.0, 0.70),
                    linewidth=0.38,
                    arrowsize=0.52,
                )
                panel.set_title(f"step {step}\n$t={step * dt:.2f}$")
            if column == 0:
                panel.set_ylabel(labels[row], labelpad=8)
                panel.set_yticks((0.0, 0.5, 1.0))
            else:
                panel.set_yticks(())
            if row == len(fields) - 1:
                panel.set_xlabel("x")
                panel.set_xticks((0.0, 0.5, 1.0))
            else:
                panel.set_xticks(())
            panel.set_xlim(0.0, 1.0)
            panel.set_ylim(0.0, 1.0)
        color_axis = figure.add_subplot(grid[row, -1])
        figure.colorbar(last_image, cax=color_axis)
    figure.suptitle(
        "Transient SfePy Navier–Stokes numerical sample over physical time\n"
        f"task {task:02d} · {family.replace('_', ' ')} lid · Re={reynolds:g} · "
        f"BDF1 $\Delta t={dt:g}$ · {resolution}² output · natural previous-state Newton start\n"
        "Ground-truth solver rollout (not a neural-network prediction)",
    )
    output = ensure_artifact_path(output_dir / f"task{task:02d}_time_evolution.png")
    return _save(
        figure,
        output,
        source=source,
        payload={
            "kind": "physical_time_evolution",
            "target_task_id": task,
            "boundary_family": family,
            "time_step_indices": indices,
            "physical_times": [step * dt for step in indices],
            "shared_color_scale_per_row": True,
            "signed_scales_clipped_at_absolute_quantile": 0.995,
        },
    )


def _path_figure(shard: dict[str, Any], source: Path, output_dir: Path) -> dict[str, Any]:
    correction = _as_numpy(shard["newton_correction"])[:, 2]
    endpoint = _as_numpy(shard["endpoint_increment"])[:, 2]
    extra = correction - endpoint
    progress = _as_numpy(shard["residual_progress"])[:, 0]
    total_steps = correction.shape[0]
    steps = sorted({min(total_steps, value) for value in (1, 2, 4, 8, 16, 24, total_steps)})
    locations = [step - 1 for step in steps]
    dt = float(shard["meta"]["dt"])
    task = int(shard["ids"]["target_task_id"])
    family = str(shard["meta"]["case"]["boundary"]["family"])

    c = correction[locations]
    e = endpoint[locations]
    x = extra[locations]
    e_flat = e.reshape(len(e), -1)
    c_flat = c.reshape(len(c), -1)
    e_norm = np.linalg.norm(e_flat, axis=1).clip(min=np.finfo(float).eps)
    discrepancy = np.linalg.norm(c_flat - e_flat, axis=1) / e_norm
    alpha = np.sum(c_flat * e_flat, axis=1) / np.square(e_norm)
    orthogonal = np.linalg.norm(c_flat - alpha[:, None] * e_flat, axis=1) / e_norm
    main_limit = _signed_limit(np.concatenate((c.reshape(-1), e.reshape(-1))))
    extra_limit = _signed_limit(x)
    fields = (e, c, x)
    labels = (
        r"Physical endpoint increment $\Delta p$",
        r"First accepted Newton correction $\delta p_1$",
        r"Non-endpoint component $\delta p_1-\Delta p$",
    )
    limits = (main_limit, main_limit, extra_limit)

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "figure.titlesize": 14})
    figure = plt.figure(figsize=(19.5, 8.1), layout="constrained")
    grid = figure.add_gridspec(3, len(steps) + 1, width_ratios=[1.0] * len(steps) + [0.045])
    for row, field in enumerate(fields):
        last_image = None
        for column, step in enumerate(steps):
            panel = figure.add_subplot(grid[row, column])
            last_image = panel.imshow(
                field[column].T,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-limits[row], vcenter=0.0, vmax=limits[row]),
                interpolation="bilinear",
                aspect="equal",
            )
            if row == 0:
                panel.set_title(
                    f"step {step} · $t={step * dt:.2f}$\n"
                    f"difference {100 * discrepancy[column]:.1f}% · "
                    f"orthogonal {100 * orthogonal[column]:.1f}%"
                )
            if column == 0:
                panel.set_ylabel(labels[row], labelpad=8)
                panel.set_yticks((0.0, 0.5, 1.0))
            else:
                panel.set_yticks(())
            if row == 2:
                panel.set_xlabel(f"residual progress {progress[locations[column]]:.2f}")
                panel.set_xticks((0.0, 0.5, 1.0))
            else:
                panel.set_xticks(())
        color_axis = figure.add_subplot(grid[row, -1])
        figure.colorbar(last_image, cax=color_axis)
    figure.suptitle(
        "Newton-path information inside each physical BDF1 time step\n"
        f"task {task:02d} · {family.replace('_', ' ')} lid · pressure channel · "
        "natural previous-state initialization\n"
        "Top and middle rows share a scale; the small difference field uses its own shared scale",
    )
    output = ensure_artifact_path(output_dir / f"task{task:02d}_newton_path_over_time.png")
    return _save(
        figure,
        output,
        source=source,
        payload={
            "kind": "newton_path_over_physical_time",
            "target_task_id": task,
            "boundary_family": family,
            "time_step_indices": steps,
            "physical_times": [step * dt for step in steps],
            "pressure_discrepancy": discrepancy.tolist(),
            "pressure_orthogonal_fraction": orthogonal.tolist(),
            "selected_residual_progress": progress[locations].tolist(),
            "correction_semantics": shard["meta"]["correction_semantics"],
            "endpoint_semantics": shard["meta"]["endpoint_semantics"],
        },
    )


def _diagnostic_figure(
    shards: list[dict[str, Any]], sources: list[Path], output_dir: Path
) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), layout="constrained")
    colors = ("#0072B2", "#D55E00")
    records: list[dict[str, Any]] = []
    for shard, color in zip(shards, colors):
        task = int(shard["ids"]["target_task_id"])
        family = str(shard["meta"]["case"]["boundary"]["family"])
        label = f"task {task:02d}: {family.replace('_', ' ')}"
        dt = float(shard["meta"]["dt"])
        trajectory = _as_numpy(shard["trajectory"])
        state_times = np.arange(trajectory.shape[0]) * dt
        step_times = np.arange(1, trajectory.shape[0]) * dt
        energy = np.asarray(shard["meta"]["energy"], dtype=np.float64)
        enstrophy = np.asarray(shard["meta"]["enstrophy"], dtype=np.float64)
        energy_times = state_times if len(energy) == len(state_times) else step_times
        enstrophy_times = state_times if len(enstrophy) == len(state_times) else step_times
        if len(energy_times) != len(energy) or len(enstrophy_times) != len(enstrophy):
            raise ValueError("Energy/enstrophy history does not match state or step count")
        correction = _as_numpy(shard["newton_correction"])[:, 2].reshape(len(step_times), -1)
        endpoint = _as_numpy(shard["endpoint_increment"])[:, 2].reshape(len(step_times), -1)
        endpoint_norm = np.linalg.norm(endpoint, axis=1).clip(min=np.finfo(float).eps)
        discrepancy = np.linalg.norm(correction - endpoint, axis=1) / endpoint_norm
        alpha = np.sum(correction * endpoint, axis=1) / np.square(endpoint_norm)
        orthogonal = np.linalg.norm(correction - alpha[:, None] * endpoint, axis=1) / endpoint_norm
        progress = _as_numpy(shard["residual_progress"])[:, 0]
        iterations = _as_numpy(shard["accepted_newton_counts"])
        axes[0, 0].plot(energy_times, energy, color=color, linewidth=1.8, label=label)
        axes[0, 1].plot(enstrophy_times, enstrophy, color=color, linewidth=1.8, label=label)
        axes[1, 0].plot(step_times, 100.0 * discrepancy, color=color, linewidth=1.8, label=f"{label}: discrepancy")
        axes[1, 0].plot(step_times, 100.0 * orthogonal, color=color, linestyle="--", linewidth=1.5, label=f"{label}: orthogonal")
        axes[1, 1].step(step_times, iterations, where="mid", color=color, linewidth=1.5, label=f"{label}: accepted states")
        axes[1, 1].plot(step_times, progress, color=color, linestyle="--", linewidth=1.5, label=f"{label}: selected progress")
        records.append(
            {
                "target_task_id": task,
                "boundary_family": family,
                "pressure_discrepancy_percent_median": float(100.0 * np.median(discrepancy)),
                "pressure_discrepancy_percent_max": float(100.0 * np.max(discrepancy)),
                "pressure_orthogonal_percent_median": float(100.0 * np.median(orthogonal)),
                "accepted_newton_states_median": float(np.median(iterations)),
            }
        )
    axes[0, 0].set_title("Kinetic energy")
    axes[0, 0].set_ylabel("domain mean")
    axes[0, 1].set_title("Enstrophy")
    axes[0, 1].set_ylabel("domain mean")
    axes[1, 0].set_title("Pressure path information")
    axes[1, 0].set_ylabel("relative to endpoint increment (%)")
    axes[1, 0].axhline(5.0, color="0.35", linestyle=":", linewidth=1.0, label="5% discrepancy reference")
    axes[1, 0].axhline(3.0, color="0.55", linestyle=":", linewidth=1.0, label="3% orthogonal reference")
    axes[1, 1].set_title("Accepted Newton path and selected progress")
    axes[1, 1].set_ylabel("count / normalized progress")
    for panel in axes.flat:
        panel.set_xlabel("physical time")
        panel.grid(alpha=0.2)
        panel.legend(frameon=False, fontsize=7, ncol=1)
    figure.suptitle(
        "Transient NS sample diagnostics across the full 32-step rollout\n"
        "Solid/dashed pairing separates magnitude discrepancy from non-collinearity",
    )
    output = ensure_artifact_path(output_dir / "transient_sample_diagnostics.png")
    return _save(
        figure,
        output,
        source=sources,
        payload={"kind": "rollout_diagnostics", "task_summaries": records},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--output-dir",
        default="artifacts/figures/navier_stokes_sfepy_transient_gate/samples",
    )
    args = parser.parse_args()
    sources = [resolve_in_project(value, must_exist=True) for value in args.inputs]
    output_dir = ensure_artifact_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    for source in sources:
        shard = torch.load(source, map_location="cpu", weights_only=False)
        validate_shard(shard)
        if shard["trajectory"].ndim != 4 or shard["trajectory"].shape[1] != 3:
            raise ValueError(f"Expected NS trajectory [T+1,3,H,W]: {source}")
        if shard["trajectory"].shape[0] < 17:
            raise ValueError(f"A time-evolution figure needs at least 16 steps: {source}")
        shards.append(shard)

    with workspace_audit("ns_visualize_transient_samples"):
        figures: list[dict[str, Any]] = []
        for shard, source in zip(shards, sources):
            figures.append(_evolution_figure(shard, source, output_dir))
            figures.append(_path_figure(shard, source, output_dir))
        figures.append(_diagnostic_figure(shards, sources, output_dir))
        index = {
            "schema": "blrpng_ns_transient_sample_figure_index_v1",
            "status": "numerical_ground_truth_samples_only",
            "interpretation": (
                "These figures visualize accepted SfePy trajectories and Newton internals; "
                "they do not contain neural-network predictions."
            ),
            "figures": [
                {
                    "path": item["output"],
                    "sha256": item["output_sha256"],
                    "kind": item["kind"],
                    "target_task_id": item.get("target_task_id"),
                }
                for item in figures
            ],
        }
        index_path = write_json(output_dir / "index.json", index)
    print(json.dumps({"index": str(index_path), "figures": index["figures"]}, indent=2))


if __name__ == "__main__":
    main()
