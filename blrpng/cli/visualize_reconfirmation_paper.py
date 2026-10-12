from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    write_json,
)


METHODS = ("nlt", "ordinary_nlt", "lr_scratch")
METHOD_LABELS = {
    "nlt": "BL-RPNG-NLT",
    "ordinary_nlt": "Final-pretrain + NLT",
    "lr_scratch": "Lifted LR scratch",
}
METHOD_COLORS = {
    "nlt": "#0072B2",
    "ordinary_nlt": "#E69F00",
    "lr_scratch": "#D55E00",
}
FAMILY_LABELS = {
    "high_frequency": "High frequency",
    "amplitude_ood": "Amplitude OOD",
    "matern_gp": "Matérn / GP",
    "localized_bump": "Localized bump",
    "mixed_fourier": "Mixed Fourier",
    "piecewise_smooth": "Piecewise smooth",
}
FAMILY_COLORS = {
    "high_frequency": "#0072B2",
    "amplitude_ood": "#E69F00",
    "matern_gp": "#009E73",
    "localized_bump": "#CC79A7",
    "mixed_fourier": "#56B4E9",
    "piecewise_smooth": "#D55E00",
}
REPRESENTATIVE_TIMES = (0, 1, 5, 10, 20, 35, 50)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads(
        resolve_in_project(relative, must_exist=True).read_text(encoding="utf-8")
    )


def _save_png(figure: plt.Figure, stem: str) -> dict[str, Any]:
    path = ensure_artifact_path(f"artifacts/figures/{stem}.png")
    figure.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _task_descriptors() -> tuple[list[dict[str, Any]], dict[int, str]]:
    payload = _load_json(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json"
    )
    tasks = sorted(payload["tasks"], key=lambda row: int(row["target_task_id"]))
    families = {
        int(row["target_task_id"]): str(row["family"])
        for row in tasks
    }
    return tasks, families


def _report(task: int, method: str, *, resolution: int = 256) -> dict[str, Any]:
    return _load_json(
        "artifacts/reports/reconfirmation/"
        f"task{task:02d}_m4_s42_{method}_query{resolution}_h20.json"
    )


def _aggregate_reports(
    families: dict[int, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, dict[str, dict[str, Any]]]]:
    rollout_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reports: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for task, family in sorted(families.items()):
        reference_ids: list[int] | None = None
        for method in METHODS:
            payload = _report(task, method)
            reports[task][method] = payload
            rows = payload["metrics"]["rollouts"]
            ids = [int(row["initial_condition_id"]) for row in rows]
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError(f"Unpaired visualization reports for task {task}.")
            for row in rows:
                rollout_rows[method].append({
                    "task": task,
                    "family": family,
                    "initial_condition_id": int(row["initial_condition_id"]),
                    "rollout_id": int(row["rollout_id"]),
                    "relative_l2": float(row["relative_l2"]),
                    "final_relative_l2": float(row["final_relative_l2"]),
                })
    return rollout_rows, reports


def _task_means(
    reports: dict[int, dict[str, dict[str, Any]]]
) -> dict[int, dict[str, float]]:
    return {
        task: {
            method: float(payload["metrics"]["primary_rollout_rel_l2"])
            for method, payload in methods.items()
        }
        for task, methods in reports.items()
    }


def _overview(
    primary128: dict[str, Any],
    primary256: dict[str, Any],
    efficiency128: dict[str, Any],
    efficiency256: dict[str, Any],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    methods = list(METHODS)
    values = [100.0 * primary256["method_mean_relative_l2"][m] for m in methods]
    bars = axes[0, 0].bar(
        np.arange(len(methods)), values,
        color=[METHOD_COLORS[m] for m in methods], width=0.65,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("20-step rollout relative L2 (%)")
    axes[0, 0].set_xticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=18, ha="right")
    axes[0, 0].set_title("(a) Frozen 256² blind-query performance")
    axes[0, 0].grid(axis="y", alpha=0.22, which="both")
    for bar, value in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    families = list(primary256["family_mean_relative_l2"])
    x = np.arange(len(families))
    width = 0.25
    for offset, method in zip((-1, 0, 1), methods):
        axes[0, 1].bar(
            x + offset * width,
            [100.0 * primary256["family_mean_relative_l2"][family][method] for family in families],
            width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x, [FAMILY_LABELS[f].replace(" ", "\n") for f in families], rotation=8)
    axes[0, 1].set_ylabel("20-step rollout relative L2 (%)")
    axes[0, 1].set_title("(b) Generalization across six unseen boundary families")
    axes[0, 1].legend(frameon=False, ncols=1)
    axes[0, 1].grid(axis="y", alpha=0.22, which="both")

    budgets = (1, 2, 4, 8)
    for resolution, payload, color, marker in (
        (128, efficiency128, "#56B4E9", "s"),
        (256, efficiency256, "#0072B2", "o"),
    ):
        curve = [100.0 * payload["mean_relative_l2"][f"m{budget}_h20"] for budget in budgets]
        axes[1, 0].plot(budgets, curve, marker=marker, linewidth=2.2, markersize=7, color=color, label=f"{resolution}² query")
    axes[1, 0].axvline(4, color="#666666", linestyle="--", linewidth=1.1, label="Pre-frozen primary m=4")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(budgets, [str(v) for v in budgets])
    axes[1, 0].set_xlabel("64² support rollouts per target boundary")
    axes[1, 0].set_ylabel("20-step rollout relative L2 (%)")
    axes[1, 0].set_title("(c) Few-shot data efficiency and resolution transfer")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.22)

    horizon_values = {
        "128²": (
            100.0 * efficiency128["mean_relative_l2"]["m4_h20"],
            100.0 * efficiency128["mean_relative_l2"]["m4_h50"],
        ),
        "256²": (
            100.0 * efficiency256["mean_relative_l2"]["m4_h20"],
            100.0 * efficiency256["mean_relative_l2"]["m4_h50"],
        ),
    }
    xh = np.arange(2)
    for offset, (label, vals), color in zip((-0.18, 0.18), horizon_values.items(), ("#56B4E9", "#0072B2")):
        axes[1, 1].bar(xh + offset, vals, width=0.34, label=label, color=color)
    axes[1, 1].set_xticks(xh, ("20 steps", "50 steps"))
    axes[1, 1].set_ylabel("Rollout relative L2 (%)")
    axes[1, 1].set_title("(d) Long-horizon stability")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.22)
    return figure


def _paired_effects(
    task_means: dict[int, dict[str, float]], families: dict[int, str]
) -> plt.Figure:
    figure = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    comparisons = (("ordinary_nlt", "Final-pretrain + NLT"), ("lr_scratch", "Lifted LR scratch"))
    for axis, (baseline, label) in zip(axes, comparisons):
        for family in FAMILY_LABELS:
            tasks = [task for task, value in families.items() if value == family]
            axis.scatter(
                [100.0 * task_means[t][baseline] for t in tasks],
                [100.0 * task_means[t]["nlt"] for t in tasks],
                s=42, color=FAMILY_COLORS[family], alpha=0.88, edgecolor="white", linewidth=0.45,
                label=FAMILY_LABELS[family],
            )
        values = [100.0 * value for task in task_means.values() for value in (task[baseline], task["nlt"])]
        lower, upper = min(values) * 0.82, max(values) * 1.18
        axis.plot([lower, upper], [lower, upper], "--", color="#333333", linewidth=1.2, label="Equal error")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel(f"{label} task-mean relL2 (%)")
        axis.set_ylabel("BL-RPNG-NLT task-mean relL2 (%)")
        axis.grid(alpha=0.22, which="both")
        wins = sum(task_means[t]["nlt"] < task_means[t][baseline] for t in task_means)
        axis.set_title(f"BL-RPNG-NLT wins {wins}/48 tasks vs {label}")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=FAMILY_COLORS[f], label=FAMILY_LABELS[f])
        for f in FAMILY_LABELS
    ]
    axes[0].legend(handles=handles, frameon=False, ncols=2, loc="upper left")

    ordered_families = list(FAMILY_LABELS)
    heat_axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    for axis, (baseline, label) in zip(heat_axes, comparisons):
        matrix = np.empty((len(ordered_families), 8), dtype=float)
        annotations = np.empty_like(matrix, dtype=object)
        for row, family in enumerate(ordered_families):
            tasks = sorted(task for task, value in families.items() if value == family)
            for column, task in enumerate(tasks):
                gain = 100.0 * (1.0 - task_means[task]["nlt"] / task_means[task][baseline])
                matrix[row, column] = gain
                annotations[row, column] = f"{gain:.0f}"
        image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=min(0.0, float(matrix.min())), vmax=float(matrix.max()))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if matrix[row, column] > 0.62 * matrix.max() else "black"
                axis.text(column, row, annotations[row, column], ha="center", va="center", fontsize=8.5, color=color)
        axis.set_xticks(np.arange(8), [f"T{i + 1}" for i in range(8)])
        axis.set_yticks(np.arange(len(ordered_families)), [FAMILY_LABELS[f] for f in ordered_families])
        axis.set_title(f"Paired error reduction vs {label} (%)")
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="Relative error reduction (%)")
    return figure


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def _rollout_distributions(
    rollout_rows: dict[str, list[dict[str, Any]]]
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.7, 4.9), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "relative_l2", "Per-rollout mean error over 20 steps"),
        (axes[1], "final_relative_l2", "Per-rollout error at step 20"),
    ):
        for method in METHODS:
            values = 100.0 * np.asarray([row[key] for row in rollout_rows[method]])
            x, y = _ecdf(values)
            axis.plot(x, y, linewidth=2.3, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            median = float(np.median(values))
            axis.scatter([median], [0.5], color=METHOD_COLORS[method], s=35, zorder=3)
        axis.set_xscale("log")
        axis.set_xlabel("Relative L2 (%)")
        axis.set_ylabel("Fraction of 768 query rollouts")
        axis.set_title(title)
        axis.grid(alpha=0.22, which="both")
        axis.set_ylim(0.0, 1.01)
    axes[0].legend(frameon=False, loc="lower right")
    return figure


def _physics_diagnostics(
    reports: dict[int, dict[str, dict[str, Any]]],
    primary128: dict[str, Any],
    primary256: dict[str, Any],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), constrained_layout=True)
    metrics = (
        ("one_step_rel_l2", "One-step"),
        ("primary_rollout_rel_l2", "20-step rollout"),
        ("near_boundary_rel_l2", "Near boundary"),
        ("interior_rel_l2", "Interior"),
    )
    x = np.arange(len(metrics))
    width = 0.24
    for offset, method in zip((-1, 0, 1), METHODS):
        arrays = [
            np.asarray([100.0 * reports[t][method]["metrics"][key] for t in sorted(reports)])
            for key, _label in metrics
        ]
        means = [float(v.mean()) for v in arrays]
        low = [float(np.quantile(v, 0.1)) for v in arrays]
        high = [float(np.quantile(v, 0.9)) for v in arrays]
        axes[0, 0].bar(x + offset * width, means, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[0, 0].errorbar(
            x + offset * width, means,
            yerr=[np.asarray(means) - np.asarray(low), np.asarray(high) - np.asarray(means)],
            fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x, [label for _key, label in metrics])
    axes[0, 0].set_ylabel("Relative L2 (%)")
    axes[0, 0].set_title("(a) Accuracy by diagnostic region (10–90% task range)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.22, which="both")

    pde_values = {
        method: np.asarray([reports[t][method]["metrics"]["pde_residual_relative_error"] for t in sorted(reports)])
        for method in METHODS
    }
    bp = axes[0, 1].boxplot(
        [pde_values[m] for m in METHODS], patch_artist=True, showfliers=False,
        tick_labels=[METHOD_LABELS[m] for m in METHODS],
    )
    for patch, method in zip(bp["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Relative Allen–Cahn residual error")
    axes[0, 1].set_title("(b) PDE-residual consistency across target tasks")
    axes[0, 1].tick_params(axis="x", rotation=14)
    axes[0, 1].grid(axis="y", alpha=0.22, which="both")
    axes[0, 1].text(
        0.02, 0.02,
        "A low local residual can coexist with the wrong phase / attractor;\n"
        "residual-only ranking is not trajectory-accuracy ranking.",
        transform=axes[0, 1].transAxes, va="bottom", fontsize=8.2,
        color="#444444",
    )

    shells = np.arange(1, 9)
    for method in METHODS:
        values = np.asarray([
            reports[t][method]["metrics"]["fourier_shell_relative_error"]
            for t in sorted(reports)
        ])
        mean = values.mean(axis=0)
        low, high = np.quantile(values, (0.1, 0.9), axis=0)
        axes[1, 0].plot(shells, mean, "o-", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1, 0].fill_between(shells, low, high, color=METHOD_COLORS[method], alpha=0.14)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Fourier shell (low → high spatial frequency)")
    axes[1, 0].set_ylabel("Relative spectral error")
    axes[1, 0].set_title("(c) Fourier-shell error (10–90% task range)")
    axes[1, 0].grid(alpha=0.22, which="both")

    families = list(primary256["family_mean_relative_l2"])
    x128 = [100.0 * primary128["family_mean_relative_l2"][f]["nlt"] for f in families]
    y256 = [100.0 * primary256["family_mean_relative_l2"][f]["nlt"] for f in families]
    for family, xv, yv in zip(families, x128, y256):
        axes[1, 1].scatter(xv, yv, s=75, color=FAMILY_COLORS[family], label=FAMILY_LABELS[family], edgecolor="white")
    lower = min([*x128, *y256]) * 0.9
    upper = max([*x128, *y256]) * 1.1
    axes[1, 1].plot([lower, upper], [lower, upper], "--", color="#333333")
    axes[1, 1].set_xlim(lower, upper)
    axes[1, 1].set_ylim(lower, upper)
    axes[1, 1].set_xlabel("128² family-mean relL2 (%)")
    axes[1, 1].set_ylabel("256² family-mean relL2 (%)")
    axes[1, 1].set_title("(d) Resolution transfer by boundary family")
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(frameon=False, fontsize=8.2, ncols=2)
    return figure


def _cost_points(
    reports: dict[int, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    points = []
    for method in METHODS:
        costs = []
        errors = []
        for task in sorted(reports):
            payload = reports[task][method]
            support = payload["support_cost"]
            setup = float(support["solver_seconds"]) + float(support["assembly_seconds"]) + float(payload["training_seconds"])
            q16 = setup + float(payload["metrics"]["evaluation_seconds"]) / 2.0
            costs.append(q16)
            errors.append(float(payload["metrics"]["primary_rollout_rel_l2"]))
        points.append({
            "name": METHOD_LABELS[method], "method": method,
            "cost_seconds": float(np.mean(costs)),
            "relative_l2": float(np.mean(errors)),
            "context": "48 new-boundary target tasks",
        })
    for budget in (1, 2, 8):
        costs, errors = [], []
        for task in sorted(reports):
            payload = _load_json(
                "artifacts/reports/reconfirmation/"
                f"task{task:02d}_m{budget}_s42_nlt_query256_h20.json"
            )
            support = payload["support_cost"]
            setup = float(support["solver_seconds"]) + float(support["assembly_seconds"]) + float(payload["training_seconds"])
            costs.append(setup + float(payload["metrics"]["evaluation_seconds"]) / 2.0)
            errors.append(float(payload["metrics"]["primary_rollout_rel_l2"]))
        points.append({
            "name": f"BL-RPNG-NLT m={budget}", "method": f"nlt_m{budget}",
            "cost_seconds": float(np.mean(costs)), "relative_l2": float(np.mean(errors)),
            "context": "48 new-boundary target tasks",
        })
    cost = _load_json(
        "artifacts/reports/allen_cahn_reconfirmation_seed42_target_cost_query256.json"
    )
    oracle_errors = [
        _load_json(f"artifacts/reports/task{task}_m256_hr_path_query256_h20.json")["metrics"]["primary_rollout_rel_l2"]
        for task in (0, 2, 7)
    ]
    points.append({
        "name": "Resource-capped HR specialist m=256",
        "method": "hr_reference",
        "cost_seconds": float(np.mean([row["target_cost_q16_seconds"] for row in cost["high_reference_rows"]])),
        "relative_l2": float(np.mean(oracle_errors)),
        "context": "3 legacy calibration tasks; contextual and not paired to the 48 new tasks",
    })
    return points


def _accuracy_cost(points: list[dict[str, Any]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.2, 6.4), constrained_layout=True)
    nlt_points = sorted(
        [row for row in points if row["method"].startswith("nlt")],
        key=lambda row: row["cost_seconds"],
    )
    axis.plot(
        [row["cost_seconds"] for row in nlt_points],
        [100.0 * row["relative_l2"] for row in nlt_points],
        color=METHOD_COLORS["nlt"], linewidth=1.2, alpha=0.55, zorder=2,
    )
    for row in points:
        if row["method"] == "hr_reference":
            marker, color, size = "*", "#000000", 210
        elif row["method"].startswith("nlt"):
            marker, color, size = "o", METHOD_COLORS["nlt"], 75
        else:
            marker, color, size = "s", METHOD_COLORS[row["method"]], 75
        axis.scatter(row["cost_seconds"], 100.0 * row["relative_l2"], marker=marker, color=color, s=size, edgecolor="white", linewidth=0.7, zorder=3)
        if row["method"].startswith("nlt"):
            continue
        else:
            label = row["name"]
            offset = (7, 7) if row["method"] != "lr_scratch" else (7, -16)
        axis.annotate(
            label, (row["cost_seconds"], 100.0 * row["relative_l2"]),
            xytext=offset, textcoords="offset points", fontsize=8.8,
        )
    hr = next(row for row in points if row["method"] == "hr_reference")
    axis.axvline(hr["cost_seconds"] / 10.0, color="#666666", linestyle="--", linewidth=1.3, label="10× target-cost threshold")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Marginal target cost for support + adaptation + 16 queries (s)")
    axis.set_ylabel("256² 20-step relative L2 (%)")
    axis.set_title("Accuracy–cost trade-off for target-boundary adaptation")
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False, loc="upper right")
    nlt_center_x = float(np.exp(np.mean(np.log([row["cost_seconds"] for row in nlt_points]))))
    nlt_center_y = float(np.mean([100.0 * row["relative_l2"] for row in nlt_points]))
    axis.annotate(
        "BL-RPNG-NLT, m=1/2/4/8",
        (nlt_center_x, nlt_center_y), xytext=(22, -22),
        textcoords="offset points", color=METHOD_COLORS["nlt"], fontsize=9.2,
        arrowprops={"arrowstyle": "-", "color": METHOD_COLORS["nlt"], "lw": 1.0},
    )
    axis.text(
        0.01, 0.02,
        "HR star: resource-capped, unsaturated, and evaluated on 3 legacy calibration tasks;\n"
        "shown as a contextual engineering reference, not a paired accuracy comparison.",
        transform=axis.transAxes, fontsize=8.5, color="#444444", va="bottom",
    )
    return figure


def _checkpoint(task: int, method: str) -> Path:
    suffix = "inference.pt" if method == "lr_scratch" else "adapter.pt"
    return resolve_in_project(
        "artifacts/checkpoints/reconfirmation/"
        f"reconfirmation_task{task:02d}_m4_s42_{method}.{suffix}",
        must_exist=True,
    )


def _representative_shard(task: int, rollout_id: int) -> tuple[dict[str, Any], str]:
    index_path = (
        f"artifacts/data/reconfirmation/task{task:02d}_query_256_all.index.json"
    )
    index = read_index(index_path)
    suffix = f"rollout_{rollout_id:06d}.pt"
    matches = [row for row in index["shards"] if str(row["path"]).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one representative shard for task={task}, rollout={rollout_id}.")
    path = resolve_in_project(matches[0]["path"], must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_shard(payload)
    return payload, sha256_file(path)


@torch.inference_mode()
def _rollout_model(
    task: int,
    method: str,
    shard: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, list[float], str]:
    checkpoint = _checkpoint(task, method)
    model, _metadata = load_checkpoint_model(checkpoint, device=device)
    trajectory = shard["trajectory"]
    resolution = int(trajectory.shape[-1])
    axis = torch.linspace(0.0, 1.0, resolution, dtype=trajectory.dtype, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((xx, yy))[None]
    lift = shard["boundary_lift"][None].to(device)
    mask = shard["constraint_mask"][None].to(device).bool()
    state = trajectory[0:1].to(device)
    states = [state[0, 0].cpu().numpy()]
    errors = [0.0]
    for step in range(50):
        prediction, _probe = model(torch.cat((state, grid), dim=1), lift, mask)
        target = trajectory[step + 1 : step + 2].to(device)
        errors.append(float((prediction - target).flatten().norm() / target.flatten().norm().clamp_min(1e-12)))
        states.append(prediction[0, 0].cpu().numpy())
        state = prediction
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.stack(states), errors, sha256_file(checkpoint)


def _representative_rollouts(
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = _load_json(
        "artifacts/figures/allen_cahn_reconfirmation_query256_fields.json"
    )
    results = []
    sources = []
    for record in frozen["records"]:
        task = int(record["task"])
        rollout_id = int(record["rollout_id"])
        shard, shard_sha = _representative_shard(task, rollout_id)
        predictions = {}
        curves = {}
        checkpoint_hashes = {}
        for method in METHODS:
            states, errors, checkpoint_sha = _rollout_model(
                task, method, shard, device
            )
            predictions[method] = states
            curves[method] = errors
            checkpoint_hashes[method] = checkpoint_sha
        results.append({
            "task": task,
            "family": str(record["family"]),
            "rollout_id": rollout_id,
            "initial_condition_id": int(record["initial_condition_id"]),
            "ground_truth": shard["trajectory"][:, 0].numpy(),
            "boundary_values": shard["boundary_values"][0].numpy(),
            "boundary_lift": shard["boundary_lift"][0].numpy(),
            "predictions": predictions,
            "curves": curves,
        })
        sources.append({
            "task": task, "rollout_id": rollout_id,
            "shard_sha256": shard_sha,
            "checkpoint_sha256": checkpoint_hashes,
            "selection_rule": record["selection_rule"],
        })
    return results, sources


def _perimeter_trace(field: np.ndarray) -> np.ndarray:
    return np.concatenate((
        field[0, :],
        field[1:, -1],
        field[-1, -2::-1],
        field[-2:0:-1, 0],
    ))


def _boundary_condition_figure(results: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(
        2, len(results), figsize=(17.2, 6.0), constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 0.72)},
    )
    image = None
    for column, result in enumerate(results):
        image = axes[0, column].imshow(
            result["boundary_lift"], origin="lower", cmap="coolwarm",
            vmin=-1.05, vmax=1.05,
        )
        axes[0, column].set_title(FAMILY_LABELS[result["family"]])
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        trace = _perimeter_trace(result["boundary_values"])
        s = np.linspace(0.0, 4.0, len(trace), endpoint=False)
        axes[1, column].plot(s, trace, color=FAMILY_COLORS[result["family"]], linewidth=1.8)
        axes[1, column].axhline(0.0, color="#777777", linewidth=0.7)
        for location in (1.0, 2.0, 3.0):
            axes[1, column].axvline(location, color="#BBBBBB", linewidth=0.6, linestyle=":")
        axes[1, column].set_xlim(0.0, 4.0)
        axes[1, column].set_ylim(-0.72, 0.72)
        axes[1, column].set_xticks((0.5, 1.5, 2.5, 3.5), ("bottom", "right", "top", "left"), rotation=28)
        axes[1, column].grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("Harmonic lift $E_hg$")
    axes[1, 0].set_ylabel("Dirichlet value $g(s)$")
    assert image is not None
    figure.colorbar(
        image, ax=axes[0, :], location="right", fraction=0.012, pad=0.012,
        label="Lift value",
    )
    figure.suptitle(
        "Pre-frozen representative non-homogeneous Dirichlet conditions",
        fontsize=15,
    )
    return figure


def _method_field_comparison(results: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(len(results), 7, figsize=(18.2, 15.5), constrained_layout=True)
    titles = (
        "Ground truth", "BL-RPNG-NLT", "Final-pretrain + NLT", "Lifted LR scratch",
        "|Error| BL-RPNG", "|Error| ordinary", "|Error| scratch",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    state_image = error_image = None
    for row, result in enumerate(results):
        target = result["ground_truth"][20]
        fields = [target] + [result["predictions"][m][20] for m in METHODS]
        errors = [np.abs(result["predictions"][m][20] - target) for m in METHODS]
        for column, field in enumerate(fields):
            state_image = axes[row, column].imshow(field, origin="lower", cmap="coolwarm", vmin=-1.05, vmax=1.05)
        for offset, error in enumerate(errors):
            error_image = axes[row, 4 + offset].imshow(error, origin="lower", cmap="magma", vmin=0.0, vmax=2.0)
        axes[row, 0].set_ylabel(
            FAMILY_LABELS[result["family"]]
            + f"\nBL relL2={100.0 * result['curves']['nlt'][20]:.2f}%"
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    assert state_image is not None and error_image is not None
    figure.colorbar(state_image, ax=axes[:, :4], location="bottom", fraction=0.018, pad=0.018, label="Physical field u")
    figure.colorbar(error_image, ax=axes[:, 4:], location="bottom", fraction=0.025, pad=0.018, label="Absolute error (shared scale)")
    figure.suptitle("Frozen representative rollouts at step 20 (256²)", fontsize=15)
    return figure


def _representative_error_growth(results: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(13.4, 7.9), constrained_layout=True, sharex=True)
    for axis, result in zip(axes.flat, results):
        steps = np.arange(1, 51)
        for method in METHODS:
            axis.plot(steps, 100.0 * np.asarray(result["curves"][method][1:]), color=METHOD_COLORS[method], linewidth=2.0, label=METHOD_LABELS[method])
        axis.axvline(20, color="#777777", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_title(FAMILY_LABELS[result["family"]])
        axis.grid(alpha=0.22, which="both")
        axis.set_xlabel("Autoregressive rollout step")
        axis.set_ylabel("Instantaneous relative L2 (%)")
    axes[0, 0].legend(frameon=False, fontsize=8.7)
    figure.suptitle("Error growth on pre-frozen representative 256² rollouts", fontsize=15)
    return figure


def _evolution_figure(result: dict[str, Any]) -> plt.Figure:
    figure, axes = plt.subplots(3, len(REPRESENTATIVE_TIMES), figsize=(18.0, 7.4), constrained_layout=True)
    target = result["ground_truth"]
    prediction = result["predictions"]["nlt"]
    state_image = error_image = None
    for column, step in enumerate(REPRESENTATIVE_TIMES):
        error = np.abs(prediction[step] - target[step])
        state_image = axes[0, column].imshow(target[step], origin="lower", cmap="coolwarm", vmin=-1.05, vmax=1.05)
        axes[1, column].imshow(prediction[step], origin="lower", cmap="coolwarm", vmin=-1.05, vmax=1.05)
        error_image = axes[2, column].imshow(error, origin="lower", cmap="magma", vmin=0.0, vmax=2.0)
        axes[0, column].set_title(f"step {step}\nt={0.2 * step:.1f}")
        if step > 0:
            axes[2, column].text(
                0.04, 0.94, f"relL2={100.0 * result['curves']['nlt'][step]:.2f}%",
                transform=axes[2, column].transAxes, va="top", color="white",
                fontsize=8.0, bbox={"facecolor": "black", "alpha": 0.42, "pad": 2, "edgecolor": "none"},
            )
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Ground truth")
    axes[1, 0].set_ylabel("BL-RPNG-NLT")
    axes[2, 0].set_ylabel("Absolute error")
    assert state_image is not None and error_image is not None
    figure.colorbar(state_image, ax=axes[:2, :], location="right", fraction=0.012, pad=0.012, label="Physical field u")
    figure.colorbar(error_image, ax=axes[2, :], location="right", fraction=0.018, pad=0.012, label="Absolute error (shared scale)")
    figure.suptitle(
        f"Boundary family: {FAMILY_LABELS[result['family']]} — frozen 256² rollout evolution",
        fontsize=15,
    )
    return figure


def main() -> None:
    _style()
    gate = _load_json(
        "artifacts/manifests/allen_cahn_reconfirmation_seed42_stage_gate.json"
    )
    if not gate.get("passed", False):
        raise RuntimeError("Frozen Allen-Cahn reconfirmation stage gate has not passed.")
    _tasks, families = _task_descriptors()
    rollout_rows, reports = _aggregate_reports(families)
    task_means = _task_means(reports)
    primary128 = _load_json(
        "artifacts/reports/allen_cahn_reconfirmation_seed42_query128_h20_m4.json"
    )
    primary256 = _load_json(
        "artifacts/reports/allen_cahn_reconfirmation_seed42_query256_h20_m4.json"
    )
    efficiency128 = _load_json(
        "artifacts/reports/allen_cahn_reconfirmation_seed42_query128_efficiency.json"
    )
    efficiency256 = _load_json(
        "artifacts/reports/allen_cahn_reconfirmation_seed42_query256_efficiency.json"
    )
    cost_points = _cost_points(reports)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    representatives, representative_sources = _representative_rollouts(device)

    outputs = []
    with workspace_audit("visualize_reconfirmation_paper_20260901") as audit:
        outputs.append(_save_png(
            _overview(primary128, primary256, efficiency128, efficiency256),
            "paper_seed42_performance_overview",
        ))
        outputs.append(_save_png(
            _paired_effects(task_means, families),
            "paper_seed42_paired_task_effects",
        ))
        outputs.append(_save_png(
            _rollout_distributions(rollout_rows),
            "paper_seed42_rollout_error_distributions",
        ))
        outputs.append(_save_png(
            _physics_diagnostics(reports, primary128, primary256),
            "paper_seed42_physics_diagnostics",
        ))
        outputs.append(_save_png(
            _accuracy_cost(cost_points),
            "paper_seed42_accuracy_cost",
        ))
        outputs.append(_save_png(
            _boundary_condition_figure(representatives),
            "paper_seed42_boundary_conditions",
        ))
        outputs.append(_save_png(
            _method_field_comparison(representatives),
            "paper_seed42_representative_method_comparison",
        ))
        outputs.append(_save_png(
            _representative_error_growth(representatives),
            "paper_seed42_representative_error_growth",
        ))
        for result in representatives:
            outputs.append(_save_png(
                _evolution_figure(result),
                f"paper_seed42_evolution_{result['family']}",
            ))

        csv_path = ensure_artifact_path(
            "artifacts/figures/paper_seed42_visualization_data.csv"
        )
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "record_type", "task", "family", "method", "unit", "step",
                "relative_l2", "final_relative_l2", "cost_seconds", "context",
            ))
            for method in METHODS:
                for row in rollout_rows[method]:
                    writer.writerow((
                        "query_rollout", row["task"], row["family"], method,
                        row["initial_condition_id"], "", row["relative_l2"],
                        row["final_relative_l2"], "", "frozen 256x256 h20 report",
                    ))
            for result in representatives:
                for method in METHODS:
                    for step, value in enumerate(result["curves"][method]):
                        writer.writerow((
                            "representative_curve", result["task"], result["family"],
                            method, result["rollout_id"], step, value, "", "",
                            "visualization-only deterministic rollout; no selection",
                        ))
            for row in cost_points:
                writer.writerow((
                    "cost_point", "", "", row["method"], row["name"], "",
                    row["relative_l2"], "", row["cost_seconds"], row["context"],
                ))
        source_paths = (
            "artifacts/reports/allen_cahn_reconfirmation_seed42_query128_h20_m4.json",
            "artifacts/reports/allen_cahn_reconfirmation_seed42_query128_efficiency.json",
            "artifacts/reports/allen_cahn_reconfirmation_seed42_query256_h20_m4.json",
            "artifacts/reports/allen_cahn_reconfirmation_seed42_query256_efficiency.json",
            "artifacts/reports/allen_cahn_reconfirmation_seed42_target_cost_query256.json",
            "artifacts/figures/allen_cahn_reconfirmation_query256_fields.json",
        )
        metadata = write_json(
            "artifacts/figures/paper_seed42_visualizations.json",
            {
                "schema": "blrpng_paper_visualizations_v1",
                "frozen_seed": 42,
                "selection_performed": False,
                "resolution": 256,
                "primary_horizon": 20,
                "evolution_horizon": 50,
                "representative_times": list(REPRESENTATIVE_TIMES),
                "representative_sources": representative_sources,
                "source_sha256": {
                    path: sha256_file(resolve_in_project(path, must_exist=True))
                    for path in source_paths
                },
                "csv": str(csv_path.relative_to(PROJECT_ROOT)),
                "csv_sha256": sha256_file(csv_path),
                "figures": outputs,
                "claim_limits": [
                    "seed42 only; seeds 0/1/2/3 remain pending",
                    "resource-capped HR point is contextual, unsaturated, and not paired to the 48 new-boundary tasks",
                    "representative visualization rollouts were pre-frozen and were not selected by HR error",
                    "post-freeze 50-step ordinary/scratch trajectories are visualization-only and did not affect checkpoint selection",
                ],
            },
        )
        audit.update({
            "metadata": str(metadata), "figures": len(outputs),
            "representatives": len(representatives),
        })
    print(json.dumps({
        "metadata": str(metadata),
        "csv": str(csv_path),
        "figures": outputs,
        "representatives": len(representatives),
    }, indent=2))


if __name__ == "__main__":
    main()
