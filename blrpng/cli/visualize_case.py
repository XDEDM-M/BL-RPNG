from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from blrpng.analysis.visualize import STYLE, _save
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import resolve_in_project, write_json
from blrpng.training.physics import allen_cahn_defect


def _grid(resolution: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(0, 1, resolution, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy)).unsqueeze(0)


def _radial_shell(error: np.ndarray, truth: np.ndarray) -> tuple[list[int], list[float]]:
    height, width = error.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2).astype(int)
    ef = np.abs(np.fft.fftshift(np.fft.fft2(error))) ** 2
    tf = np.abs(np.fft.fftshift(np.fft.fft2(truth))) ** 2
    shells, values = [], []
    for shell in range(1, min(height, width) // 2):
        selected = radius == shell
        if selected.any():
            shells.append(shell)
            values.append(float(np.sqrt(ef[selected].mean() / max(tf[selected].mean(), 1e-20))))
    return shells, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen representative-case paper diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--resolution", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stem", default="allen_cahn_v2_representative")
    args = parser.parse_args()
    split = "query_512" if args.resolution == 512 else "query_hr"
    index = read_index(f"artifacts/data/tasks/task{args.task:02d}_{split}_{args.resolution}_all.index.json",
                       verify_hashes=True)
    shard = torch.load(resolve_in_project(index["shards"][0]["path"], must_exist=True),
                       map_location="cpu", weights_only=False)
    horizon = min(args.horizon, shard["trajectory"].shape[0] - 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_model(args.checkpoint, device=device)
    lift = shard["boundary_lift"].unsqueeze(0).to(device)
    mask = shard["constraint_mask"].unsqueeze(0).to(device)
    current = shard["trajectory"][0].unsqueeze(0).to(device)
    predictions = [current]
    probes = None
    with workspace_audit("visualize_case"):
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.cat((current, _grid(args.resolution, device)), dim=1)
                current, probes = model(x, lift, mask)
                predictions.append(current)
        prediction = predictions[-1][0, 0].cpu().numpy()
        truth = shard["trajectory"][horizon, 0].numpy()
        absolute = np.abs(prediction - truth)
        boundary = shard["constraint_mask"][0].numpy().astype(bool)
        distance = distance_transform_edt(~boundary)
        near = np.where(distance <= max(4, args.resolution // 32), absolute, np.nan)
        common = (float(min(prediction.min(), truth.min())), float(max(prediction.max(), truth.max())))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), constrained_layout=True)
            images = [
                axes[0].imshow(prediction, cmap="viridis", vmin=common[0], vmax=common[1]),
                axes[1].imshow(truth, cmap="viridis", vmin=common[0], vmax=common[1]),
                axes[2].imshow(absolute, cmap="magma"), axes[3].imshow(near, cmap="magma"),
            ]
            for ax, title in zip(axes, ("Prediction", "Ground truth", "Absolute error", "Near-boundary error")):
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(images[1], ax=axes[:2], shrink=.72); fig.colorbar(images[2], ax=axes[2:], shrink=.72)
        snapshot = _save(f"{args.stem}_field", fig, {
            "task": args.task, "resolution": args.resolution, "horizon": horizon,
            "relative_l2": float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-15)),
            "near_boundary_mae": float(np.nanmean(near)), "max_absolute_error": float(absolute.max()),
        })

        step_errors, pred_residual, truth_residual = [], [], []
        for step in range(1, horizon + 1):
            predicted = predictions[step]
            target = shard["trajectory"][step].unsqueeze(0).to(device)
            step_errors.append(float(torch.linalg.vector_norm(predicted - target) /
                                     torch.linalg.vector_norm(target).clamp_min(1e-12)))
            pred_residual.append(float(allen_cahn_defect(predictions[step - 1], predicted,
                {"dt": .2, "epsilon": .03, "gamma": 5.0}).square().mean().sqrt()))
            previous_truth = shard["trajectory"][step - 1].unsqueeze(0).to(device)
            truth_residual.append(float(allen_cahn_defect(previous_truth, target,
                {"dt": .2, "epsilon": .03, "gamma": 5.0}).square().mean().sqrt()))
        shells, shell_error = _radial_shell(prediction - truth, truth)
        max_bin = min(int(distance.max()), 32)
        distance_error = [float(absolute[(distance >= d) & (distance < d + 1)].mean())
                          for d in range(max_bin)]
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), constrained_layout=True)
            axes[0].plot(range(1, horizon + 1), step_errors); axes[0].set_title("Rollout error")
            axes[1].plot(shells, shell_error); axes[1].set_yscale("log"); axes[1].set_title("Fourier shell error")
            axes[2].plot(range(max_bin), distance_error); axes[2].set_title("Error vs boundary distance")
            axes[3].plot(range(1, horizon + 1), pred_residual, label="prediction")
            axes[3].plot(range(1, horizon + 1), truth_residual, label="truth"); axes[3].set_yscale("log")
            axes[3].set_title("PDE defect"); axes[3].legend(fontsize=7)
        curves = _save(f"{args.stem}_diagnostics", fig, {
            "step": list(range(1, horizon + 1)), "rollout_relative_l2": step_errors,
            "fourier_shell": shells, "fourier_shell_relative_error": shell_error,
            "boundary_distance_pixels": list(range(max_bin)), "boundary_distance_mae": distance_error,
            "prediction_defect_rms": pred_residual, "ground_truth_defect_rms": truth_residual,
        })

        x0 = torch.cat((shard["trajectory"][0].unsqueeze(0).to(device), _grid(args.resolution, device)), dim=1)
        with torch.no_grad():
            _, first_probes = model(x0, lift, mask)
        predicted_correction = first_probes["layer1:hook0"][0, 0].cpu().numpy()
        true_correction = shard["newton_correction"][0, 0].numpy()
        correction_error = np.abs(predicted_correction - true_correction)
        inner = float(np.sum(predicted_correction * true_correction))
        cosine = inner / max(float(np.linalg.norm(predicted_correction) * np.linalg.norm(true_correction)), 1e-15)
        log_magnitude = float(np.log((np.linalg.norm(predicted_correction) + 1e-12) /
                                     (np.linalg.norm(true_correction) + 1e-12)))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.8), constrained_layout=True)
            for ax, value, title in zip(axes, (true_correction, predicted_correction, correction_error),
                                        ("Newton correction", "Probe correction", "Absolute difference")):
                image = ax.imshow(value, cmap="coolwarm" if "difference" not in title.lower() else "magma")
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(image, ax=ax, shrink=.7)
        correction = _save(f"{args.stem}_newton_correction", fig, {
            "cosine": cosine, "log_magnitude_ratio": log_magnitude,
            "relative_l2": float(np.linalg.norm(predicted_correction - true_correction) /
                                 max(np.linalg.norm(true_correction), 1e-15)),
        })
        report = write_json(f"artifacts/reports/{args.stem}_visualization.json",
                            {"snapshot": snapshot, "diagnostics": curves, "correction": correction})
    print(json.dumps({"report": str(report), "relative_l2": snapshot}, indent=2))


if __name__ == "__main__":
    main()
