from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    write_json,
)
from blrpng.training.losses import per_sample_relative_l2


TASKS = (0, 4, 8, 12)
TASK_LABELS = {
    0: "High-frequency BC",
    4: "Amplitude OOD",
    8: "Cross-field correlation shift",
    12: "Reaction-parameter OOD",
}
METHOD_LABELS = {
    "final_nlt": "Ordinary pretrain + NLT",
    "path_nlt": "BL-RPNG-NLT",
    "final_nlt_multiscale": "Ordinary + multiscale",
    "path_nlt_multiscale": "BL-RPNG + multiscale",
    "scratch": "LR scratch (converged)",
}
COLORS = {
    "final_nlt": "#E69F00",
    "path_nlt": "#0072B2",
    "final_nlt_multiscale": "#CC79A7",
    "path_nlt_multiscale": "#009E73",
    "scratch": "#D55E00",
    "truth": "#222222",
}
FIELD_NAMES = ("u", "v")
TIME_POINTS = (0, 5, 10, 20, 35, 50)
OUT = "artifacts/figures/gray_scott_r2"


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(
    figure: plt.Figure,
    stem: str,
    *,
    data: dict[str, Any],
    sources: list[dict[str, str]],
) -> dict[str, str]:
    png = ensure_artifact_path(f"{OUT}/{stem}.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metadata = write_json(f"{OUT}/{stem}.json", {
        "schema": "blrpng_frozen_png_figure_v1",
        "selection_performed": False,
        "representative_rule": "first rollout in each frozen task index",
        "sources": sources,
        "data": data,
        "png": str(png.relative_to(PROJECT_ROOT)),
        "png_sha256": sha256_file(png),
    })
    return {
        "name": stem,
        "png": str(png.relative_to(PROJECT_ROOT)),
        "png_sha256": sha256_file(png),
        "metadata": str(metadata.relative_to(PROJECT_ROOT)),
        "metadata_sha256": sha256_file(metadata),
    }


def _method_paths() -> tuple[dict[int, dict[str, dict[str, str]]], list[dict[str, str]]]:
    selection_path = "artifacts/manifests/gray_scott_r2_target_selection.json"
    scratch_path = "artifacts/manifests/gray_scott_seed42_saturated_scratch_freeze.json"
    selection = _load_json(selection_path)
    scratch = _load_json(scratch_path)
    scratch_by_task = {int(row["task"]): row for row in scratch["tasks"]}
    outputs: dict[int, dict[str, dict[str, str]]] = {}
    for task in TASKS:
        outputs[task] = {
            method: {
                "path": selection["checkpoints"][str(task)][method]["path"],
                "sha256": selection["checkpoints"][str(task)][method]["sha256"],
            }
            for method in (
                "final_nlt", "path_nlt", "final_nlt_multiscale",
                "path_nlt_multiscale",
            )
        }
        outputs[task]["scratch"] = {
            "path": scratch_by_task[task]["selected_path"],
            "sha256": scratch_by_task[task]["selected_sha256"],
        }
        for method, spec in outputs[task].items():
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"Frozen checkpoint hash mismatch: task={task}, {method}")
    sources = [
        {"path": selection_path, "sha256": sha256_file(resolve_in_project(selection_path))},
        {"path": scratch_path, "sha256": sha256_file(resolve_in_project(scratch_path))},
    ]
    return outputs, sources


def _grid(resolution: int, batch: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy)).unsqueeze(0).expand(batch, -1, -1, -1)


def _region_relative(error: torch.Tensor, target: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    region = region.to(device=error.device, dtype=error.dtype).view(1, 1, *region.shape)
    numerator = torch.sqrt(torch.sum(error.square() * region, dim=(1, 2, 3)))
    denominator = torch.sqrt(torch.sum(target.square() * region, dim=(1, 2, 3))).clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def _rollout_model(
    model: torch.nn.Module,
    payloads: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    resolution = int(payloads[0]["trajectory"].shape[-1])
    batch = len(payloads)
    current = torch.stack([row["trajectory"][0] for row in payloads]).to(device)
    lift = torch.stack([row["boundary_lift"] for row in payloads]).to(device)
    mask = torch.stack([row["constraint_mask"] for row in payloads]).to(device)
    condition = torch.stack([row["condition_fields"] for row in payloads]).to(device)
    grid = _grid(resolution, batch, device)
    coordinates = torch.arange(resolution)
    pixel_distance = torch.minimum(
        torch.minimum(coordinates[:, None], coordinates.flip(0)[:, None]),
        torch.minimum(coordinates[None, :], coordinates.flip(0)[None, :]),
    ).float() / (resolution - 1)
    bins = (0.0, 0.02, 0.05, 0.10, 0.20, 0.51)
    step_error, field_error = [], []
    representative = {0: current[0].detach().cpu().numpy()}
    final_distance = None
    for step in range(50):
        target = torch.stack([row["trajectory"][step + 1] for row in payloads]).to(device)
        prediction, _ = model(torch.cat((current, grid, condition), dim=1), lift, mask)
        step_error.append(per_sample_relative_l2(prediction, target).cpu().numpy())
        per_field = []
        for field in range(2):
            per_field.append(per_sample_relative_l2(
                prediction[:, field : field + 1], target[:, field : field + 1]
            ).cpu().numpy())
        field_error.append(np.stack(per_field, axis=1))
        if step + 1 in TIME_POINTS:
            representative[step + 1] = prediction[0].detach().cpu().numpy()
        if step == 49:
            difference = prediction - target
            final_distance = np.stack([
                _region_relative(
                    difference, target,
                    (pixel_distance >= low) & (pixel_distance < high),
                ).cpu().numpy()
                for low, high in zip(bins[:-1], bins[1:])
            ], axis=1)
        current = prediction
    return {
        "step_error": np.stack(step_error, axis=1),
        "field_error": np.stack(field_error, axis=1),
        "distance_error": final_distance,
        "distance_bins": bins,
        "representative": representative,
    }


def _radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = field - field.mean()
    spectrum = np.abs(np.fft.rfft2(centered, norm="ortho")) ** 2
    kx = np.fft.fftfreq(field.shape[0]) * field.shape[0]
    ky = np.fft.rfftfreq(field.shape[1]) * field.shape[1]
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).astype(int)
    shell = np.bincount(radius.ravel(), weights=spectrum.ravel())
    count = np.bincount(radius.ravel()).clip(min=1)
    shell = shell / count
    shell = shell / max(shell.sum(), 1e-15)
    cutoff = min(40, len(shell))
    return np.arange(cutoff), shell[:cutoff]


def _summary_records() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    p128 = "artifacts/reports/gray_scott/r2_blind_query128.json"
    p256 = "artifacts/reports/gray_scott/r2_blind_query256.json"
    s128 = "artifacts/reports/gray_scott/r2_blind_query128_scratch.json"
    s256 = "artifacts/reports/gray_scott/r2_blind_query256_scratch.json"
    reports = {
        128: _load_json(p128), 256: _load_json(p256),
        "s128": _load_json(s128), "s256": _load_json(s256),
    }
    records = {
        128: reports[128]["records"] + reports["s128"]["records"],
        256: reports[256]["records"] + reports["s256"]["records"],
    }
    for resolution in (128, 256):
        for row in records[resolution]:
            if row["method"] == "lifted_lr_scratch_saturated":
                row["method"] = "scratch"
    sources = [
        {"path": path, "sha256": sha256_file(resolve_in_project(path))}
        for path in (p128, p256, s128, s256)
    ]
    return reports, records, sources


def _performance_figure(records: dict[int, list[dict[str, Any]]]) -> tuple[plt.Figure, dict[str, Any]]:
    methods = ("final_nlt", "path_nlt", "path_nlt_multiscale", "scratch")
    means: dict[str, float] = {}
    for resolution in (128, 256):
        for horizon in (20, 50):
            for method in methods:
                values = [
                    row["relative_l2"] for row in records[resolution]
                    if row["horizon"] == horizon and row["method"] == method
                ]
                means[f"r{resolution}_h{horizon}_{method}"] = float(np.mean(values))
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    x = np.arange(len(methods)); width = 0.36
    for offset, horizon, hatch in ((-0.5, 20, ""), (0.5, 50, "//")):
        axes[0, 0].bar(
            x + offset * width,
            [100 * means[f"r256_h{horizon}_{m}"] for m in methods],
            width, color=[COLORS[m] for m in methods], hatch=hatch,
            alpha=0.92, label=f"{horizon} steps",
        )
    axes[0, 0].set_xticks(x, [METHOD_LABELS[m].replace(" + ", "\n+") for m in methods])
    axes[0, 0].set_ylabel("Relative L2 error (%)")
    axes[0, 0].set_title("(a) Fresh 256² blind query")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.22)

    for method in methods:
        axes[0, 1].plot(
            [128, 256], [100 * means[f"r{r}_h50_{method}"] for r in (128, 256)],
            "o-", lw=2, color=COLORS[method], label=METHOD_LABELS[method],
        )
    axes[0, 1].set_xticks((128, 256), ("128²", "256²"))
    axes[0, 1].set_ylabel("50-step relative L2 (%)")
    axes[0, 1].set_title("(b) Resolution transfer")
    axes[0, 1].grid(alpha=0.22)
    axes[0, 1].legend(frameon=False, ncol=2)

    task_x = np.arange(len(TASKS)); bar_width = 0.25
    for offset, method in zip((-1, 0, 1), ("final_nlt", "path_nlt", "scratch")):
        vals = []
        for task in TASKS:
            vals.append(100 * np.mean([
                row["relative_l2"] for row in records[256]
                if row["horizon"] == 50 and row["method"] == method
                and row["target_task_id"] == task
            ]))
        axes[1, 0].bar(task_x + offset * bar_width, vals, bar_width,
                       color=COLORS[method], label=METHOD_LABELS[method])
    axes[1, 0].set_xticks(task_x, [TASK_LABELS[t].replace(" ", "\n", 1) for t in TASKS])
    axes[1, 0].set_ylabel("256² / 50-step error (%)")
    axes[1, 0].set_title("(c) Boundary-condition transfer")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.22)

    improvements = []
    for task in TASKS:
        values = {}
        for method in ("final_nlt", "path_nlt"):
            values[method] = np.mean([
                row["relative_l2"] for row in records[256]
                if row["horizon"] == 50 and row["method"] == method
                and row["target_task_id"] == task
            ])
        improvements.append(100 * (1 - values["path_nlt"] / values["final_nlt"]))
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in improvements]
    axes[1, 1].barh(np.arange(len(TASKS)), improvements, color=colors)
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_yticks(np.arange(len(TASKS)), [TASK_LABELS[t] for t in TASKS])
    axes[1, 1].set_xlabel("BL-RPNG improvement over ordinary (%)")
    axes[1, 1].set_title("(d) Heterogeneous path-pretraining effect")
    axes[1, 1].grid(axis="x", alpha=0.22)
    return figure, {"means": means, "task_h50_improvements_percent": dict(zip(map(str, TASKS), improvements))}


def _paired_scatter(records: dict[int, list[dict[str, Any]]]) -> tuple[plt.Figure, dict[str, Any]]:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    exported = {}
    for axis, horizon in zip(axes, (20, 50)):
        selected = [row for row in records[256] if row["horizon"] == horizon]
        indexed = defaultdict(dict)
        for row in selected:
            indexed[(row["target_task_id"], row["initial_condition_id"])][row["method"]] = row["relative_l2"]
        for task in TASKS:
            pairs = [values for (t, _), values in indexed.items() if t == task]
            x = [100 * row["final_nlt"] for row in pairs]
            y = [100 * row["path_nlt"] for row in pairs]
            axis.scatter(x, y, s=38, alpha=0.8, label=TASK_LABELS[task])
        limit = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([0, limit], [0, limit], "--", color="#555555", lw=1)
        axis.set_xlim(0, limit); axis.set_ylim(0, limit)
        axis.set_xlabel("Ordinary pretrain + NLT error (%)")
        axis.set_ylabel("BL-RPNG-NLT error (%)")
        axis.set_title(f"{horizon}-step paired rollouts")
        axis.grid(alpha=0.2)
        exported[str(horizon)] = {"paired_units": len(indexed)}
    axes[1].legend(frameon=False, fontsize=7)
    return figure, exported


def _load_query_payloads(task: int, resolution: int = 256) -> tuple[list[dict[str, Any]], str]:
    index_path = (
        "artifacts/data/gray_scott_r2_seed42/blind_query/"
        f"task{task:02d}_query_{resolution}.index.json"
    )
    index = read_index(index_path, verify_hashes=True)
    payloads = [
        torch.load(resolve_in_project(row["path"], must_exist=True),
                   map_location="cpu", weights_only=False)
        for row in index["shards"]
    ]
    return payloads, index_path


def _run_visual_inference(
    method_paths: dict[int, dict[str, dict[str, str]]],
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, str]]]:
    cache: dict[int, dict[str, Any]] = {}
    truth: dict[int, dict[str, Any]] = {}
    sources = []
    for task in TASKS:
        payloads, index_path = _load_query_payloads(task)
        sources.append({"path": index_path, "sha256": sha256_file(resolve_in_project(index_path))})
        truth[task] = {
            "representative": {time: payloads[0]["trajectory"][time].numpy() for time in TIME_POINTS},
            "boundary_values": payloads[0]["boundary_values"].numpy(),
            "boundary_lift": payloads[0]["boundary_lift"].numpy(),
            "mask": payloads[0]["constraint_mask"].numpy(),
            "initial_condition_id": int(payloads[0]["ids"]["initial_condition_id"]),
        }
        cache[task] = {}
        for method in (
            "final_nlt", "path_nlt", "final_nlt_multiscale",
            "path_nlt_multiscale", "scratch",
        ):
            model, _ = load_checkpoint_model(method_paths[task][method]["path"], device=device)
            cache[task][method] = _rollout_model(model, payloads, device=device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return cache, truth, sources


def _error_growth_figure(cache: dict[int, dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    methods = ("final_nlt", "path_nlt", "path_nlt_multiscale", "scratch")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    exported = {}
    for axis, task in zip(axes.flat, TASKS):
        exported[str(task)] = {}
        for method in methods:
            values = 100 * cache[task][method]["step_error"]
            mean = values.mean(axis=0)
            low, high = np.quantile(values, (0.25, 0.75), axis=0)
            steps = np.arange(1, 51)
            axis.plot(steps, mean, color=COLORS[method], lw=2, label=METHOD_LABELS[method])
            axis.fill_between(steps, low, high, color=COLORS[method], alpha=0.12)
            exported[str(task)][method] = mean.tolist()
        axis.set_title(TASK_LABELS[task])
        axis.set_xlabel("Rollout step"); axis.set_ylabel("Relative L2 (%)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    return figure, exported


def _boundary_figure(truth: dict[int, dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    figure, axes = plt.subplots(4, 4, figsize=(12.5, 12), constrained_layout=True)
    for row, task in enumerate(TASKS):
        arrays = [
            truth[task]["boundary_values"][0], truth[task]["boundary_lift"][0],
            truth[task]["boundary_values"][1], truth[task]["boundary_lift"][1],
        ]
        titles = ("$g_u$ (trace)", "$E g_u$", "$g_v$ (trace)", "$E g_v$")
        for col, (array, title) in enumerate(zip(arrays, titles)):
            image = axes[row, col].imshow(array, origin="lower", cmap="viridis")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(title)
            if col == 0: axes[row, col].set_ylabel(TASK_LABELS[task])
            figure.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.02)
    return figure, {"tasks": list(TASKS), "resolution": 256}


def _evolution_figure(
    task: int,
    cache: dict[int, dict[str, Any]],
    truth: dict[int, dict[str, Any]],
) -> tuple[plt.Figure, dict[str, Any]]:
    rows = (
        (0, "truth", "$u$ ground truth"),
        (0, "final_nlt", "$u$ ordinary"),
        (0, "path_nlt", "$u$ BL-RPNG"),
        (0, "path_error", "$|e_u|$ BL-RPNG"),
        (1, "truth", "$v$ ground truth"),
        (1, "final_nlt", "$v$ ordinary"),
        (1, "path_nlt", "$v$ BL-RPNG"),
        (1, "path_error", "$|e_v|$ BL-RPNG"),
    )
    figure, axes = plt.subplots(len(rows), len(TIME_POINTS), figsize=(16.5, 17.5), constrained_layout=True)
    field_limits = {}
    for field in (0, 1):
        arrays = [truth[task]["representative"][time][field] for time in TIME_POINTS]
        field_limits[field] = (min(float(a.min()) for a in arrays), max(float(a.max()) for a in arrays))
    errors = [
        np.abs(cache[task]["path_nlt"]["representative"][time] - truth[task]["representative"][time])
        for time in TIME_POINTS
    ]
    error_limits = {
        field: max(float(np.quantile(e[field], 0.995)) for e in errors) for field in (0, 1)
    }
    for row, (field, kind, label) in enumerate(rows):
        for col, time in enumerate(TIME_POINTS):
            if kind == "truth":
                array = truth[task]["representative"][time][field]
                cmap = "viridis"; vmin, vmax = field_limits[field]
            elif kind == "path_error":
                array = np.abs(
                    cache[task]["path_nlt"]["representative"][time][field]
                    - truth[task]["representative"][time][field]
                )
                cmap = "magma"; vmin, vmax = 0, error_limits[field]
            else:
                array = cache[task][kind]["representative"][time][field]
                cmap = "viridis"; vmin, vmax = field_limits[field]
            axes[row, col].imshow(array, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(f"t = {time}")
            if col == 0: axes[row, col].set_ylabel(label)
    figure.suptitle(
        f"{TASK_LABELS[task]} — fixed first blind rollout "
        f"(IC {truth[task]['initial_condition_id']})",
        fontsize=13,
    )
    return figure, {
        "task": task, "boundary_condition": TASK_LABELS[task],
        "initial_condition_id": truth[task]["initial_condition_id"],
        "times": list(TIME_POINTS), "resolution": 256,
        "methods": ["final_nlt", "path_nlt"],
    }


def _final_comparison_figure(
    cache: dict[int, dict[str, Any]], truth: dict[int, dict[str, Any]]
) -> tuple[plt.Figure, dict[str, Any]]:
    columns = (
        ("truth", "$v$ GT"), ("final_nlt", "$v$ ordinary"),
        ("path_nlt", "$v$ BL-RPNG"), ("scratch", "$v$ LR scratch"),
        ("error_final", "$|e_v|$ ordinary"), ("error_path", "$|e_v|$ BL-RPNG"),
    )
    figure, axes = plt.subplots(4, len(columns), figsize=(16, 10.7), constrained_layout=True)
    for row, task in enumerate(TASKS):
        gt = truth[task]["representative"][50][1]
        vmin, vmax = float(gt.min()), float(gt.max())
        error_max = max(
            float(np.quantile(np.abs(cache[task][method]["representative"][50][1] - gt), 0.995))
            for method in ("final_nlt", "path_nlt")
        )
        for col, (kind, title) in enumerate(columns):
            if kind == "truth": array, cmap, lo, hi = gt, "viridis", vmin, vmax
            elif kind == "error_final":
                array = np.abs(cache[task]["final_nlt"]["representative"][50][1] - gt)
                cmap, lo, hi = "magma", 0, error_max
            elif kind == "error_path":
                array = np.abs(cache[task]["path_nlt"]["representative"][50][1] - gt)
                cmap, lo, hi = "magma", 0, error_max
            else:
                array = cache[task][kind]["representative"][50][1]
                cmap, lo, hi = "viridis", vmin, vmax
            axes[row, col].imshow(array, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(title)
            if col == 0: axes[row, col].set_ylabel(TASK_LABELS[task])
    return figure, {"tasks": list(TASKS), "time": 50, "field": "v", "resolution": 256}


def _boundary_distance_figure(cache: dict[int, dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True, sharex=True)
    methods = ("final_nlt", "path_nlt", "scratch")
    exported = {}
    for axis, task in zip(axes.flat, TASKS):
        bins = cache[task]["path_nlt"]["distance_bins"]
        centers = [(low + high) / 2 for low, high in zip(bins[:-1], bins[1:])]
        exported[str(task)] = {}
        for method in methods:
            mean = 100 * cache[task][method]["distance_error"].mean(axis=0)
            axis.plot(centers, mean, "o-", lw=2, color=COLORS[method], label=METHOD_LABELS[method])
            exported[str(task)][method] = mean.tolist()
        axis.set_yscale("log")
        axis.set_title(TASK_LABELS[task]); axis.grid(alpha=0.2)
        axis.set_xlabel("Distance from boundary"); axis.set_ylabel("t=50 regional rel. L2 (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    return figure, {"bin_edges": list(cache[0]["path_nlt"]["distance_bins"]), "values": exported}


def _spectrum_figure(
    cache: dict[int, dict[str, Any]], truth: dict[int, dict[str, Any]]
) -> tuple[plt.Figure, dict[str, Any]]:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True, sharex=True, sharey=True)
    exported = {}
    for axis, task in zip(axes.flat, TASKS):
        exported[str(task)] = {}
        arrays = {
            "truth": truth[task]["representative"][50][1],
            "final_nlt": cache[task]["final_nlt"]["representative"][50][1],
            "path_nlt": cache[task]["path_nlt"]["representative"][50][1],
            "scratch": cache[task]["scratch"]["representative"][50][1],
        }
        for method, array in arrays.items():
            shell, energy = _radial_spectrum(array)
            axis.semilogy(shell[1:], energy[1:] + 1e-14, lw=2,
                          color=COLORS[method], label="Ground truth" if method == "truth" else METHOD_LABELS[method])
            exported[str(task)][method] = energy.tolist()
        axis.set_title(TASK_LABELS[task]); axis.grid(alpha=0.2)
        axis.set_xlabel("Radial Fourier shell"); axis.set_ylabel("Normalized shell energy")
    axes[0, 0].legend(frameon=False, fontsize=7)
    return figure, exported


@torch.no_grad()
def _newton_figure(
    method_paths: dict[int, dict[str, dict[str, str]]], device: torch.device
) -> tuple[plt.Figure, dict[str, Any], list[dict[str, str]]]:
    figure, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    exported = {}; sources = []
    for row, task in enumerate(TASKS):
        index_path = f"artifacts/data/gray_scott_seed42/task{task:02d}_support_64_m4.index.json"
        index = read_index(index_path, verify_hashes=True)
        payload = torch.load(resolve_in_project(index["shards"][0]["path"], must_exist=True),
                             map_location="cpu", weights_only=False)
        sources.append({"path": index_path, "sha256": sha256_file(resolve_in_project(index_path))})
        model, _ = load_checkpoint_model(method_paths[task]["path_nlt"]["path"], device=device)
        resolution = 64
        state = payload["trajectory"][0].unsqueeze(0).to(device)
        lift = payload["boundary_lift"].unsqueeze(0).to(device)
        mask = payload["constraint_mask"].unsqueeze(0).to(device)
        condition = payload["condition_fields"].unsqueeze(0).to(device)
        _, probes = model(torch.cat((state, _grid(resolution, 1, device), condition), 1), lift, mask)
        predicted = probes["layer1:hook0"][0, 1].cpu().numpy()
        target = payload["newton_correction"][0, 1].numpy()
        cosine = float(np.dot(predicted.ravel(), target.ravel()) /
                       max(np.linalg.norm(predicted) * np.linalg.norm(target), 1e-15))
        norm_ratio = float(np.linalg.norm(predicted) / max(np.linalg.norm(target), 1e-15))
        maximum = max(float(np.abs(target).max()), float(np.abs(predicted).max()), 1e-12)
        for col, (array, title) in enumerate((
            (target, "Newton $\\delta v$"), (predicted, "Probe $\\hat{\\delta v}$"),
            (np.abs(predicted - target), "Absolute error"),
        )):
            cmap = "magma" if col == 2 else "coolwarm"
            lo, hi = (0, maximum) if col == 2 else (-maximum, maximum)
            axes[row, col].imshow(array, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(title)
            if col == 0: axes[row, col].set_ylabel(f"{TASK_LABELS[task]}\ncos={cosine:.2f}, norm={norm_ratio:.2f}")
        exported[str(task)] = {"cosine": cosine, "norm_ratio": norm_ratio}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    return figure, exported, sources


def _cost_figure(
    method_paths: dict[int, dict[str, dict[str, str]]]
) -> tuple[plt.Figure, dict[str, Any], list[dict[str, str]]]:
    values = {}; sources = []
    selection = _load_json("artifacts/manifests/gray_scott_r2_target_selection.json")
    eval256 = _load_json("artifacts/reports/gray_scott/r2_blind_query256.json")
    for task in TASKS:
        support_index_path = f"artifacts/data/gray_scott_seed42/task{task:02d}_support_64_m4.index.json"
        query_index_path = (
            "artifacts/data/gray_scott_r2_seed42/blind_query/"
            f"task{task:02d}_query_256.index.json"
        )
        support = read_index(support_index_path, verify_hashes=True)
        query = read_index(query_index_path, verify_hashes=True)
        sources.extend([
            {"path": support_index_path, "sha256": sha256_file(resolve_in_project(support_index_path))},
            {"path": query_index_path, "sha256": sha256_file(resolve_in_project(query_index_path))},
        ])
        lr_solve = 0.0
        for row in support["shards"]:
            payload = torch.load(resolve_in_project(row["path"], must_exist=True),
                                 map_location="cpu", weights_only=False)
            lr_solve += sum(payload["meta"]["solver_timing"])
        hr_times = []
        for row in query["shards"]:
            payload = torch.load(resolve_in_project(row["path"], must_exist=True),
                                 map_location="cpu", weights_only=False)
            hr_times.append(sum(payload["meta"]["solver_timing"][:50]))
        adaptation = selection["checkpoints"][str(task)]["path_nlt"]["training_seconds"]
        reports = [
            row for row in eval256["model_reports"]
            if row["task"] == task and row["method"] == "path_nlt" and row["horizon"] == 50
        ]
        inference = reports[0]["evaluation_seconds"] / 8
        values[str(task)] = {
            "lr_support_solve_seconds": lr_solve,
            "nlt_adaptation_seconds": adaptation,
            "hr_numerical_solve_seconds_per_query": float(np.mean(hr_times)),
            "neural_inference_seconds_per_query": inference,
        }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    x = np.arange(len(TASKS)); width = 0.2
    keys = (
        "lr_support_solve_seconds", "nlt_adaptation_seconds",
        "hr_numerical_solve_seconds_per_query", "neural_inference_seconds_per_query",
    )
    labels = ("4× LR support solve", "NLT adaptation", "One HR numerical solve", "One neural rollout")
    colors = ("#56B4E9", "#0072B2", "#D55E00", "#009E73")
    for index, (key, label, color) in enumerate(zip(keys, labels, colors)):
        axes[0].bar(x + (index - 1.5) * width, [values[str(t)][key] for t in TASKS],
                    width, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [f"Task {t}" for t in TASKS])
    axes[0].set_ylabel("Measured wall time (s, log scale)")
    axes[0].set_title("(a) Target-task cost components")
    axes[0].legend(frameon=False, fontsize=7); axes[0].grid(axis="y", alpha=0.2)
    break_even = []
    marginal = []
    for task in TASKS:
        row = values[str(task)]
        fixed = row["lr_support_solve_seconds"] + row["nlt_adaptation_seconds"]
        difference = row["hr_numerical_solve_seconds_per_query"] - row["neural_inference_seconds_per_query"]
        break_even.append(fixed / max(difference, 1e-12))
        marginal.append(row["hr_numerical_solve_seconds_per_query"] /
                        max(row["neural_inference_seconds_per_query"], 1e-12))
    axes[1].bar(x - width / 2, break_even, width, color="#0072B2", label="Break-even queries")
    axes[1].bar(x + width / 2, marginal, width, color="#009E73", label="Marginal speedup")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [f"Task {t}" for t in TASKS])
    axes[1].set_ylabel("Queries or speedup (log scale)")
    axes[1].set_title("(b) Deployment break-even")
    axes[1].legend(frameon=False); axes[1].grid(axis="y", alpha=0.2)
    return figure, {"tasks": values, "break_even_queries": break_even, "marginal_speedup": marginal}, sources


def _write_curve_csv(cache: dict[int, dict[str, Any]]) -> Path:
    path = ensure_artifact_path(f"{OUT}/gray_scott_r2_rollout_curves.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task", "boundary_condition", "method", "step", "mean_rel_l2", "q25", "q75"))
        for task in TASKS:
            for method, result in cache[task].items():
                values = result["step_error"]
                for step in range(50):
                    writer.writerow((
                        task, TASK_LABELS[task], method, step + 1,
                        float(values[:, step].mean()),
                        float(np.quantile(values[:, step], 0.25)),
                        float(np.quantile(values[:, step], 0.75)),
                    ))
    return path


def main() -> None:
    _style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_paths, checkpoint_sources = _method_paths()
    _, records, report_sources = _summary_records()
    sources = checkpoint_sources + report_sources
    figures = []
    with workspace_audit("visualize_gray_scott_r2_paper") as audit:
        figure, data = _performance_figure(records)
        figures.append(_save(figure, "01_performance_overview", data=data, sources=sources))
        figure, data = _paired_scatter(records)
        figures.append(_save(figure, "02_paired_rollout_scatter", data=data, sources=sources))

        cache, truth, query_sources = _run_visual_inference(method_paths, device)
        all_sources = sources + query_sources
        curve_csv = _write_curve_csv(cache)
        figure, data = _error_growth_figure(cache)
        data["csv"] = str(curve_csv.relative_to(PROJECT_ROOT)); data["csv_sha256"] = sha256_file(curve_csv)
        figures.append(_save(figure, "03_rollout_error_growth", data=data, sources=all_sources))
        figure, data = _boundary_figure(truth)
        figures.append(_save(figure, "04_boundary_values_and_lifts", data=data, sources=query_sources))
        for index, task in enumerate(TASKS, start=5):
            figure, data = _evolution_figure(task, cache, truth)
            figures.append(_save(
                figure, f"{index:02d}_evolution_task{task:02d}", data=data,
                sources=all_sources,
            ))
        figure, data = _final_comparison_figure(cache, truth)
        figures.append(_save(figure, "09_final_state_method_comparison", data=data, sources=all_sources))
        figure, data = _boundary_distance_figure(cache)
        figures.append(_save(figure, "10_boundary_distance_error", data=data, sources=all_sources))
        figure, data = _spectrum_figure(cache, truth)
        figures.append(_save(figure, "11_fourier_shell_spectrum", data=data, sources=all_sources))
        figure, data, correction_sources = _newton_figure(method_paths, device)
        figures.append(_save(
            figure, "12_newton_correction_alignment", data=data,
            sources=checkpoint_sources + correction_sources,
        ))
        figure, data, cost_sources = _cost_figure(method_paths)
        figures.append(_save(
            figure, "13_target_task_cost", data=data,
            sources=checkpoint_sources + cost_sources,
        ))
        manifest = write_json(f"{OUT}/figure_manifest.json", {
            "schema": "blrpng_gray_scott_r2_png_figure_manifest_v1",
            "frozen": True,
            "png_only": True,
            "selection_performed": False,
            "representative_rule": "first rollout in each frozen query index",
            "seed": 42,
            "resolution": 256,
            "figures": figures,
            "curve_csv": str(curve_csv.relative_to(PROJECT_ROOT)),
            "curve_csv_sha256": sha256_file(curve_csv),
        })
        audit.update({
            "manifest": str(manifest), "figures": len(figures),
            "outside_outputs": False,
        })
    print(json.dumps({
        "manifest": str(manifest), "figure_count": len(figures),
        "figures": figures,
    }, indent=2))


if __name__ == "__main__":
    main()
