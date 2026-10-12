from __future__ import annotations

from collections import defaultdict
from typing import Any
import time

import torch
from torch.nn import functional as F

from blrpng.data.schema import DatasetBundle
from blrpng.models.fno import DirichletOperator
from blrpng.training.losses import per_sample_relative_l2


def _region_masks(mask: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    near = mask.float()
    kernel = 2 * width + 1
    near = F.max_pool2d(near, kernel_size=kernel, stride=1, padding=width).bool()
    return near, ~near


def _relative_region(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    difference = ((prediction - target) * mask).flatten(1).norm(dim=1)
    denominator = (target * mask).flatten(1).norm(dim=1).clamp_min(1e-12)
    return difference / denominator


def allen_cahn_residual(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    dt: float,
    epsilon: float,
    gamma: float,
) -> torch.Tensor:
    height, width = current.shape[-2:]
    hx = 1.0 / (height - 1)
    hy = 1.0 / (width - 1)
    center = current[:, :, 1:-1, 1:-1]
    laplace = (
        (current[:, :, 2:, 1:-1] - 2.0 * center + current[:, :, :-2, 1:-1])
        / hx**2
        + (current[:, :, 1:-1, 2:] - 2.0 * center + current[:, :, 1:-1, :-2])
        / hy**2
    )
    return (center - previous[:, :, 1:-1, 1:-1]) / dt - epsilon**2 * laplace + gamma * (
        center**3 - center
    )


def fourier_shell_relative_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shells: int = 8,
) -> list[float]:
    difference = torch.fft.rfft2(prediction - target, norm="ortho").abs().square()
    reference = torch.fft.rfft2(target, norm="ortho").abs().square()
    height, width_half = difference.shape[-2:]
    ky = torch.fft.fftfreq(height, device=prediction.device).abs()[:, None]
    kx = torch.fft.rfftfreq((width_half - 1) * 2, device=prediction.device)[None, :]
    radius = torch.sqrt(kx.square() + ky.square())
    edges = torch.linspace(0.0, float(radius.max()), shells + 1, device=prediction.device)
    result = []
    for index in range(shells):
        selected = (radius >= edges[index]) & (
            radius <= edges[index + 1] if index == shells - 1 else radius < edges[index + 1]
        )
        numerator = difference[..., selected].sum()
        denominator = reference[..., selected].sum().clamp_min(1e-20)
        result.append(float(torch.sqrt(numerator / denominator)))
    return result


@torch.no_grad()
def evaluate_bundle(
    model: DirichletOperator,
    bundle: DatasetBundle,
    *,
    device: torch.device,
    horizon: int,
    near_boundary_width: int,
    equation: dict[str, float],
    coarse_bundle: DatasetBundle | None = None,
    coarse_initial_resolution: int | None = None,
) -> dict[str, Any]:
    model.eval()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tensors = bundle.tensors
    groups: dict[int, list[int]] = defaultdict(list)
    for index, rollout_id in enumerate(tensors["rollout_id"].tolist()):
        groups[int(rollout_id)].append(index)

    rollout_records = []
    all_predictions = []
    all_targets = []
    all_previous = []
    boundary_errors = []
    near_errors = []
    interior_errors = []
    one_step_errors = []
    coarse_floor = []
    interpolated_initial_rollouts = []
    coarse_by_pair: dict[tuple[int, int], int] = {}
    if coarse_bundle is not None:
        for index in range(len(coarse_bundle)):
            key = (
                int(coarse_bundle.tensors["pair_id"][index]),
                int(coarse_bundle.tensors["local_step_id"][index]),
            )
            coarse_by_pair[key] = index

    for rollout_id, indices in groups.items():
        indices = sorted(indices, key=lambda idx: int(tensors["local_step_id"][idx]))[:horizon]
        current = tensors["x"][indices[0] : indices[0] + 1].to(device)
        step_values = []
        for index in indices:
            lift = tensors["boundary_lift"][index : index + 1].to(device)
            mask = tensors["boundary_mask"][index : index + 1].to(device).bool()
            target = tensors["y"][index : index + 1].to(device)
            # Teacher-forced one-step.
            teacher_prediction, _ = model(
                tensors["x"][index : index + 1].to(device), lift, mask
            )
            one_step_errors.append(float(per_sample_relative_l2(teacher_prediction, target)[0]))
            prediction, _ = model(current, lift, mask)
            relative = float(per_sample_relative_l2(prediction, target)[0])
            step_values.append(relative)
            boundary_errors.append(float((prediction[mask] - target[mask]).abs().max()))
            near, interior = _region_masks(mask, near_boundary_width)
            near_errors.append(float(_relative_region(prediction, target, near)[0]))
            interior_errors.append(float(_relative_region(prediction, target, interior)[0]))
            all_predictions.append(prediction)
            all_targets.append(target)
            all_previous.append(current[:, : model.config.out_channels])

            if coarse_bundle is not None:
                pair_key = (int(tensors["pair_id"][index]), int(tensors["local_step_id"][index]))
                if pair_key in coarse_by_pair:
                    coarse_index = coarse_by_pair[pair_key]
                    interpolated = F.interpolate(
                        coarse_bundle.tensors["y"][coarse_index : coarse_index + 1].to(device),
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                    interpolated = torch.where(mask, target, interpolated)
                    coarse_floor.append(float(per_sample_relative_l2(interpolated, target)[0]))
            current = torch.cat(
                (
                    prediction,
                    tensors["x"][
                        index : index + 1,
                        model.config.out_channels :,
                    ].to(device),
                ),
                dim=1,
            )
        first = indices[0]
        rollout_records.append(
            {
                "rollout_id": rollout_id,
                "pair_id": int(tensors["pair_id"][first]),
                "boundary_id": int(tensors["boundary_id"][first]),
                "initial_condition_id": int(tensors["initial_condition_id"][first]),
                "relative_l2": float(sum(step_values) / len(step_values)),
                "final_relative_l2": step_values[-1],
            }
        )

        if coarse_bundle is not None or coarse_initial_resolution is not None:
            first_key = (
                int(tensors["pair_id"][indices[0]]),
                int(tensors["local_step_id"][indices[0]]),
            )
            if coarse_bundle is not None and first_key in coarse_by_pair:
                coarse_first = coarse_by_pair[first_key]
                coarse_state = coarse_bundle.tensors["x"][
                    coarse_first : coarse_first + 1, : model.config.out_channels
                ].to(device)
            else:
                fine_state = tensors["x"][
                    indices[0] : indices[0] + 1, : model.config.out_channels
                ].to(device)
                coarse_state = F.interpolate(
                    fine_state,
                    size=(int(coarse_initial_resolution), int(coarse_initial_resolution)),
                    mode="bilinear",
                    align_corners=True,
                )
            interpolated_state = F.interpolate(
                    coarse_state,
                    size=tensors["y"].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            first_mask = tensors["boundary_mask"][
                indices[0] : indices[0] + 1
            ].to(device).bool()
            first_boundary = tensors["boundary_values"][
                indices[0] : indices[0] + 1
            ].to(device)
            interpolated_state = torch.where(
                first_mask, first_boundary, interpolated_state
            )
            interpolated_current = torch.cat(
                (
                    interpolated_state,
                    tensors["x"][
                        indices[0] : indices[0] + 1, model.config.out_channels :
                    ].to(device),
                ),
                dim=1,
            )
            interpolated_errors = []
            for index in indices:
                lift = tensors["boundary_lift"][index : index + 1].to(device)
                mask = tensors["boundary_mask"][index : index + 1].to(device).bool()
                target = tensors["y"][index : index + 1].to(device)
                interpolated_prediction, _ = model(
                    interpolated_current, lift, mask
                )
                interpolated_errors.append(
                    float(per_sample_relative_l2(interpolated_prediction, target)[0])
                )
                interpolated_current = torch.cat(
                    (
                        interpolated_prediction,
                        tensors["x"][
                            index : index + 1, model.config.out_channels :
                        ].to(device),
                    ),
                    dim=1,
                )
            interpolated_initial_rollouts.append(
                float(sum(interpolated_errors) / len(interpolated_errors))
            )

    prediction = torch.cat(all_predictions)
    target = torch.cat(all_targets)
    previous = torch.cat(all_previous)
    residual = allen_cahn_residual(previous, prediction, **equation)
    target_residual = allen_cahn_residual(previous, target, **equation)
    residual_relative = float(
        (residual - target_residual).flatten(1).norm(dim=1).mean()
        / target_residual.flatten(1).norm(dim=1).mean().clamp_min(1e-12)
    )
    elapsed = time.perf_counter() - started
    return {
        "primary_rollout_rel_l2": float(
            sum(row["relative_l2"] for row in rollout_records) / len(rollout_records)
        ),
        "rollout_mse": float(torch.mean((prediction - target) ** 2)),
        "one_step_rel_l2": float(sum(one_step_errors) / len(one_step_errors)),
        "near_boundary_rel_l2": float(sum(near_errors) / len(near_errors)),
        "interior_rel_l2": float(sum(interior_errors) / len(interior_errors)),
        "boundary_error_max": max(boundary_errors),
        "pde_residual_relative_error": residual_relative,
        "fourier_shell_relative_error": fourier_shell_relative_error(prediction, target),
        "coarse_solver_interpolation_floor": (
            float(sum(coarse_floor) / len(coarse_floor)) if coarse_floor else None
        ),
        "coarse_initial_interpolated_rollout_rel_l2": (
            float(
                sum(interpolated_initial_rollouts)
                / len(interpolated_initial_rollouts)
            )
            if interpolated_initial_rollouts
            else None
        ),
        "rollouts": rollout_records,
        "evaluation_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
