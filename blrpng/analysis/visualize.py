from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from blrpng.paths import ensure_artifact_path, sha256_file, write_json


STYLE = {
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "svg.hashsalt": "blrpng-v2",
}


def _save(stem: str, fig: plt.Figure, data: dict[str, Any]) -> dict[str, Any]:
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = ensure_artifact_path(f"artifacts/figures/v2/{stem}.{extension}")
        metadata = {"Creator": "BL-RPNG", "CreationDate": None} if extension == "pdf" else {"Creator": "BL-RPNG"}
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        outputs[extension] = {"path": str(path), "sha256": sha256_file(path)}
    plt.close(fig)
    companion = write_json(f"artifacts/figures/v2/{stem}.json", data)
    return {"figure": stem, "outputs": outputs, "data": str(companion), "data_sha256": sha256_file(companion)}


def accuracy_cost(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["methods"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for row in rows:
            ax.scatter(row["cost_seconds"], row["relative_l2"], s=48, label=row["method"])
        ax.set_xscale("log"); ax.set_yscale("log")
        costs = [float(row["cost_seconds"]) for row in rows]
        if costs:
            ax.axvline(max(costs) / 10.0, color="#777777", ls="--", lw=1,
                       label="10× vs HR oracle cost")
        ax.set_xlabel("Target-task cost (s)"); ax.set_ylabel("256² 20-step relative L2")
        ax.legend(fontsize=7)
    return _save("accuracy_cost_pareto", fig, {"methods": rows})


def cost_break_even(summary: dict[str, Any]) -> dict[str, Any]:
    costs = summary["costs"]
    task_ids = sorted(costs, key=int)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
        x = np.arange(len(task_ids)); width = .36
        labels = ("assembly_seconds", "solve_seconds", "adaptation_seconds")
        colors = ("#8ecae6", "#219ebc", "#023047")
        for offset, key in ((-width / 2, "method_components"), (width / 2, "oracle_components")):
            bottom = np.zeros(len(task_ids))
            for label, color in zip(labels, colors):
                values = np.asarray([costs[t][key][label] for t in task_ids])
                axes[0].bar(x + offset, values, width, bottom=bottom, color=color,
                            label=label.replace("_seconds", "") if offset < 0 else None)
                bottom += values
        axes[0].text(.02, .98, "left=BL-RPNG, right=HR oracle", transform=axes[0].transAxes,
                     fontsize=7, va="top")
        axes[0].set_yscale("log"); axes[0].set_xticks(x, task_ids); axes[0].set_xlabel("Target task")
        axes[0].set_ylabel("Target cost (s)"); axes[0].legend(fontsize=7)
        break_even = [costs[t]["break_even_queries"] or 0 for t in task_ids]
        axes[1].bar(x, break_even); axes[1].set_xticks(x, task_ids)
        axes[1].set_xlabel("Target task"); axes[1].set_ylabel("Break-even queries")
    return _save("cost_break_even", fig, {"tasks": task_ids, "costs": costs})


def oracle_gap(summary: dict[str, Any]) -> dict[str, Any]:
    curves = summary["oracle_curves"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for name, curve in curves.items():
            budgets = sorted(map(int, curve))
            ax.plot(budgets, [curve[str(value)] if str(value) in curve else curve[value] for value in budgets], marker="o", label=name)
        ax.set_xscale("log", base=2); ax.set_xlabel("Support rollouts"); ax.set_ylabel("Relative L2")
        ax.legend(fontsize=7)
    return _save("oracle_gap_support_curve", fig, {"oracle_curves": curves})


def resolution_curve(summary: dict[str, Any]) -> dict[str, Any]:
    curves = summary["resolution_curves"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for name, curve in curves.items():
            resolutions = sorted(map(int, curve))
            ax.plot(resolutions, [curve[str(value)] if str(value) in curve else curve[value] for value in resolutions], marker="o", label=name)
        ax.set_xscale("log", base=2); ax.set_xlabel("Inference resolution"); ax.set_ylabel("Relative L2 / commutator")
        ax.legend(fontsize=7)
    return _save("resolution_commutator", fig, {"resolution_curves": curves})


def paired_heatmap(summary: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(summary["paired_heatmap"]["values"], dtype=float)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-np.max(np.abs(values)), vmax=np.max(np.abs(values)), aspect="auto")
        ax.set_xticks(range(values.shape[1]), summary["paired_heatmap"]["tasks"])
        ax.set_yticks(range(values.shape[0]), summary["paired_heatmap"]["methods"])
        baseline = summary["paired_heatmap"].get("baseline", "baseline")
        fig.colorbar(image, ax=ax, label=f"Relative improvement vs {baseline}")
    return _save("paired_improvement_heatmap", fig, summary["paired_heatmap"])


def gradient_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["gradient_history"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(5.0, 4.8), sharex=True)
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["gradient_cosine"] for row in rows], label="cos(final,path)")
        axes[0].axhline(0, color="black", lw=.8)
        axes[1].plot(epochs, [row["path_target_ratio"] for row in rows], label="target ratio")
        axes[1].plot(epochs, [row["path_gate_active"] for row in rows], label="active fraction")
        axes[0].legend(); axes[1].legend(); axes[1].set_xlabel("Epoch")
    return _save("gradient_gate_diagnostics", fig, {"gradient_history": rows})


def generate_core_figures(summary: dict[str, Any]) -> list[dict[str, Any]]:
    functions: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = [
        ("methods", accuracy_cost), ("oracle_curves", oracle_gap),
        ("resolution_curves", resolution_curve), ("paired_heatmap", paired_heatmap),
        ("gradient_history", gradient_diagnostics), ("costs", cost_break_even),
    ]
    return [function(summary) for key, function in functions if key in summary]


def _derived_multiphysics_fields(
    value: np.ndarray, equation: str
) -> tuple[list[np.ndarray], list[str]]:
    if equation == "hyperelastic" and value.shape[0] >= 2:
        ux, uy = value[0], value[1]
        h = 1.0 / max(value.shape[-1] - 1, 1)
        dux_dx, dux_dy = np.gradient(ux, h, h)
        duy_dx, duy_dy = np.gradient(uy, h, h)
        trace = dux_dx + duy_dy
        sxx = 2.0 * dux_dx + 2.0 * trace
        syy = 2.0 * duy_dy + 2.0 * trace
        sxy = dux_dy + duy_dx
        von_mises = np.sqrt((sxx - syy) ** 2 + 3.0 * sxy**2)
        return [np.sqrt(ux**2 + uy**2), von_mises], ["displacement magnitude", "von Mises stress"]
    if equation in {"navier_stokes", "incompressible_navier_stokes"} and value.shape[0] >= 3:
        u, v, pressure = value[:3]
        h = 1.0 / max(value.shape[-1] - 1, 1)
        du_dx, du_dy = np.gradient(u, h, h)
        dv_dx, dv_dy = np.gradient(v, h, h)
        return [
            np.sqrt(u**2 + v**2), pressure, dv_dx - du_dy, du_dx + dv_dy,
        ], ["speed", "pressure", "vorticity", "divergence"]
    return [], []


def multifield_comparison(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    boundary_mask: np.ndarray,
    field_names: list[str],
    equation: str,
    stem: str,
) -> dict[str, Any]:
    """Create a checkpoint-selection-free multiphysics comparison panel."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(boundary_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have identical [C,H,W] shape.")
    if len(field_names) != prediction.shape[0]:
        raise ValueError("field_names must contain one label per channel.")
    if mask.shape[0] == 1 and prediction.shape[0] > 1:
        mask = np.repeat(mask, prediction.shape[0], axis=0)
    if mask.shape != prediction.shape:
        raise ValueError("boundary mask is not channel-compatible.")
    pred_derived, derived_names = _derived_multiphysics_fields(prediction, equation)
    true_derived, _ = _derived_multiphysics_fields(target, equation)
    pred_fields = [*prediction, *pred_derived]
    true_fields = [*target, *true_derived]
    names = [*field_names, *derived_names]
    # Derived diagnostics use the union of constrained field masks.
    union_mask = mask.any(axis=0)
    derived_masks = [union_mask for _ in derived_names]
    masks = [*mask, *derived_masks]
    rows = len(names)
    diagnostics = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, 4, figsize=(10.8, max(2.3 * rows, 2.8)), squeeze=False,
                                 constrained_layout=True)
        for row, (name, pred, truth, field_mask) in enumerate(
            zip(names, pred_fields, true_fields, masks)
        ):
            error = np.abs(pred - truth)
            if not field_mask.any():
                field_mask = np.zeros_like(field_mask, dtype=bool)
                field_mask[(0, -1), :] = True
                field_mask[:, (0, -1)] = True
            distance = distance_transform_edt(~field_mask)
            near = np.where(distance <= max(4, prediction.shape[-1] // 32), error, np.nan)
            low, high = float(min(pred.min(), truth.min())), float(max(pred.max(), truth.max()))
            images = (
                axes[row, 0].imshow(pred, cmap="viridis", vmin=low, vmax=high),
                axes[row, 1].imshow(truth, cmap="viridis", vmin=low, vmax=high),
                axes[row, 2].imshow(error, cmap="magma"),
                axes[row, 3].imshow(near, cmap="magma"),
            )
            for column, title in enumerate(("prediction", "ground truth", "absolute error", "near boundary")):
                axes[row, column].set_title(f"{name}: {title}")
                axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            fig.colorbar(images[1], ax=axes[row, :2], shrink=.65)
            fig.colorbar(images[2], ax=axes[row, 2:], shrink=.65)
            diagnostics.append({
                "field": name,
                "relative_l2": float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-15)),
                "mae": float(error.mean()),
                "near_boundary_mae": float(np.nanmean(near)),
                "max_absolute_error": float(error.max()),
            })
    return _save(f"{stem}_multiphysics_panel", fig, {
        "equation": equation,
        "shape": list(prediction.shape),
        "diagnostics": diagnostics,
    })
