from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json


EXPERIMENT = Path("artifacts/experiments/poisson_negative_control")
FIGURE_ROOT = Path("artifacts/figures/poisson_negative_control")
COLORS = {
    "ordinary_nlt": "#0072B2",
    "path_nlt": "#D55E00",
    "lr_scratch_converged": "#666666",
    "boundary_augmented_lr_scratch": "#CC79A7",
}


def _load(path: str | Path) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))


def _task_means(method: str, resolution: int) -> np.ndarray:
    if method in {"ordinary_nlt", "path_nlt"}:
        payload = _load(EXPERIMENT / f"seed42_{method}_frozen_query.json")
        rows = [row for row in payload["evaluations"] if row["resolution"] == resolution]
        return np.asarray([
            row["metrics"]["primary_rollout_relative_l2"] for row in rows
        ])
    values = []
    for task in range(30):
        name = (
            f"seed42_lr_scratch_task{task:02d}.json"
            if method == "lr_scratch_converged"
            else f"seed42_{method}_task{task:02d}.json"
        )
        values.append(_load(EXPERIMENT / name)["query"][str(resolution)]["primary_rollout_relative_l2"])
    return np.asarray(values)


def _comparison_figure(summary: dict) -> tuple[Path, dict]:
    methods = [
        "ordinary_nlt", "path_nlt", "lr_scratch_converged",
        "boundary_augmented_lr_scratch",
    ]
    labels = ["Final-pretrain→NLT", "Path-pretrain→NLT", "Lifted LR scratch", "Boundary+mask scratch"]
    task_values = {method: _task_means(method, 256) for method in methods}
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)

    x = np.arange(len(methods))
    width = 0.34
    means128 = [summary["resolutions"]["128"]["means"][method] for method in methods]
    means256 = [summary["resolutions"]["256"]["means"][method] for method in methods]
    axes[0].bar(x - width / 2, means128, width, label="128²", color="#56B4E9")
    axes[0].bar(x + width / 2, means256, width, label="256²", color="#E69F00")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean relative L2 (log scale)")
    axes[0].set_title("Poisson negative control: accuracy after four 64² support solves")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25, which="both")
    for index, value in enumerate(means256):
        axes[0].text(index + width / 2, value * 1.14, f"{100*value:.2f}%", ha="center", fontsize=9)

    tasks = np.arange(30)
    for method, label in zip(methods, labels):
        axes[1].plot(
            tasks, 100 * task_values[method], marker="o", markersize=3,
            linewidth=1.5, color=COLORS[method], label=label,
        )
    axes[1].axvline(9.5, color="black", linewidth=0.8, alpha=0.4)
    axes[1].axvline(19.5, color="black", linewidth=0.8, alpha=0.4)
    axes[1].text(4.5, 0.98, "Fourier OOD", transform=axes[1].get_xaxis_transform(), ha="center")
    axes[1].text(14.5, 0.98, "Localized bump", transform=axes[1].get_xaxis_transform(), ha="center")
    axes[1].text(24.5, 0.98, "Piecewise-smooth", transform=axes[1].get_xaxis_transform(), ha="center")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Frozen target boundary task")
    axes[1].set_ylabel("256² relative L2 (%)")
    axes[1].grid(alpha=0.2, which="both")
    axes[1].legend(ncol=2, frameon=False)

    improvement = 100 * (task_values["ordinary_nlt"] - task_values["path_nlt"]) / task_values["ordinary_nlt"]
    family_colors = np.asarray(["#009E73"] * 10 + ["#E69F00"] * 10 + ["#56B4E9"] * 10)
    axes[2].bar(tasks, improvement, color=family_colors)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].axhline(5, color="black", linestyle="--", linewidth=1, label="±5% materiality band")
    axes[2].axhline(-5, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Frozen target boundary task")
    axes[2].set_ylabel("Path improvement over ordinary (%)")
    axes[2].set_title("No systematic linear-PDE benefit from Newton-path pretraining")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False)

    output = ensure_artifact_path(FIGURE_ROOT / "poisson_negative_control_comparison.png")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    data = {
        "methods": methods,
        "labels": labels,
        "means_128": dict(zip(methods, means128)),
        "means_256": dict(zip(methods, means256)),
        "task_means_256": {key: value.tolist() for key, value in task_values.items()},
        "path_improvement_percent_by_task": improvement.tolist(),
    }
    write_json(FIGURE_ROOT / "poisson_negative_control_comparison.json", data)
    return output, data


@torch.no_grad()
def _field_figure() -> tuple[Path, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = (0, 10, 20)
    families = ("Fourier OOD", "Localized bump", "Piecewise-smooth")
    fig, axes = plt.subplots(len(tasks), 7, figsize=(21, 9), constrained_layout=True)
    metadata = []
    for row_index, (task, family) in enumerate(zip(tasks, families)):
        shard_index = _load(
            f"artifacts/data/poisson_negative_control/task{task:02d}_query_256_all.index.json"
        )
        shard = torch.load(
            resolve_in_project(shard_index["shards"][0]["path"], must_exist=True),
            map_location="cpu", weights_only=False,
        )
        trajectory = shard["trajectory"].to(device)
        lift = shard["boundary_lift"].unsqueeze(0).to(device)
        mask = shard["constraint_mask"].unsqueeze(0).to(device)
        forcing = shard["condition_fields"].unsqueeze(0).to(device)
        axis = torch.linspace(0, 1, 256, device=device)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        model_input = torch.cat((trajectory[0:1], torch.stack((xx, yy)).unsqueeze(0), forcing), dim=1)
        predictions = {}
        for method in ("ordinary_nlt", "path_nlt"):
            checkpoint = (
                f"artifacts/checkpoints/poisson/"
                f"poisson_task{task:02d}_m4_s42_{method}.adapter.pt"
            )
            model, _ = load_checkpoint_model(checkpoint, device=device)
            predictions[method] = model(model_input, lift, mask)[0][0, 0].cpu().numpy()
            del model
        target = trajectory[1, 0].cpu().numpy()
        forcing_np = forcing[0, 0].cpu().numpy()
        lift_np = lift[0, 0].cpu().numpy()
        ordinary_error = np.abs(predictions["ordinary_nlt"] - target)
        path_error = np.abs(predictions["path_nlt"] - target)
        field_values = [
            forcing_np, lift_np, target, predictions["ordinary_nlt"], ordinary_error,
            predictions["path_nlt"], path_error,
        ]
        titles = ["Forcing f", "Harmonic lift Eg", "Ground truth", "Ordinary NLT", "|error|", "Path NLT", "|error|"]
        common_min = min(target.min(), predictions["ordinary_nlt"].min(), predictions["path_nlt"].min())
        common_max = max(target.max(), predictions["ordinary_nlt"].max(), predictions["path_nlt"].max())
        error_max = max(ordinary_error.max(), path_error.max())
        for column, (value, title) in enumerate(zip(field_values, titles)):
            if column in (2, 3, 5):
                image = axes[row_index, column].imshow(value.T, origin="lower", cmap="coolwarm", vmin=common_min, vmax=common_max)
            elif column in (4, 6):
                image = axes[row_index, column].imshow(value.T, origin="lower", cmap="magma", vmin=0, vmax=error_max)
            else:
                image = axes[row_index, column].imshow(value.T, origin="lower", cmap="coolwarm")
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
            if row_index == 0:
                axes[row_index, column].set_title(title)
            if column == 0:
                axes[row_index, column].set_ylabel(f"{family}\nTask {task}")
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.02)
        metadata.append({
            "task": task, "family": family,
            "ordinary_relative_l2": float(np.linalg.norm(predictions["ordinary_nlt"] - target) / np.linalg.norm(target)),
            "path_relative_l2": float(np.linalg.norm(predictions["path_nlt"] - target) / np.linalg.norm(target)),
            "pair_id": int(shard["ids"]["pair_id"]),
        })
    output = ensure_artifact_path(FIGURE_ROOT / "poisson_representative_fields_256.png")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    data = {"resolution": 256, "cases": metadata}
    write_json(FIGURE_ROOT / "poisson_representative_fields_256.json", data)
    return output, data


def main() -> None:
    summary = _load(EXPERIMENT / "seed42_summary.json")
    comparison, _ = _comparison_figure(summary)
    fields, _ = _field_figure()
    manifest = {
        "schema": "blrpng_poisson_figure_manifest_v1",
        "frozen_summary": str(EXPERIMENT / "seed42_summary.json"),
        "figures": [
            {"path": str(comparison), "sha256": sha256_file(comparison)},
            {"path": str(fields), "sha256": sha256_file(fields)},
        ],
    }
    output = write_json(FIGURE_ROOT / "figure_manifest.json", manifest)
    print(json.dumps({"manifest": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
