from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def per_sample_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    difference = (prediction - target).flatten(1)
    denominator = target.flatten(1).norm(dim=1).clamp_min(epsilon)
    return difference.norm(dim=1) / denominator


def per_sample_field_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor, rms_floor: float = 1e-3,
) -> torch.Tensor:
    """Return [batch, field] relative errors; equal field weighting is explicit."""
    if prediction.shape != target.shape or target.ndim != 4:
        raise ValueError('Field-relative loss requires matching [B,C,H,W] tensors.')
    numerator = torch.linalg.vector_norm(prediction - target, dim=(-2, -1))
    denominator = torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(
        rms_floor * (target.shape[-1] * target.shape[-2]) ** 0.5
    )
    return numerator / denominator


def correction_direction_magnitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    log_magnitude_weight: float = 0.1,
    epsilon: float = 1e-8,
    field_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Grid-spacing-aware direction and log-magnitude correction loss."""
    if prediction.shape != target.shape:
        raise ValueError(f"Correction shapes differ: {prediction.shape} != {target.shape}")
    batch, _, height, width = prediction.shape
    cell_weight = 1.0 / max((height - 1) * (width - 1), 1)
    if field_scales is not None:
        scales = field_scales.to(prediction)
        if scales.ndim == 1:
            scales = scales.view(1, -1, 1, 1)
        elif scales.ndim == 2:
            scales = scales[:, :, None, None]
        else:
            raise ValueError("field_scales must have shape [C] or [B,C].")
        if scales.shape[1] != prediction.shape[1]:
            raise ValueError("field_scales must contain one value per correction field.")
        prediction = prediction / scales
        target = target / scales
    pred_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    pred_norm = (cell_weight * pred_flat.square().sum(dim=1)).sqrt()
    target_norm = (cell_weight * target_flat.square().sum(dim=1)).sqrt()
    inner = cell_weight * (pred_flat * target_flat).sum(dim=1)
    cosine = inner / ((pred_norm + epsilon) * (target_norm + epsilon))
    direction = 1.0 - cosine.clamp(-1.0, 1.0)
    log_ratio = torch.log((pred_norm + epsilon) / (target_norm + epsilon))
    magnitude = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction="none")
    valid = valid_mask.reshape(batch).bool() & (target_norm > epsilon)
    if not torch.any(valid):
        zero = prediction.sum() * 0.0
        return zero, {"direction": 0.0, "log_magnitude": 0.0, "valid": 0.0}
    loss = (direction + log_magnitude_weight * magnitude)[valid].mean()
    return loss, {
        "direction": float(direction[valid].mean().detach()),
        "log_magnitude": float(magnitude[valid].mean().detach()),
        "valid": float(valid.float().mean()),
    }


def _gradients_and_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[tuple[torch.Tensor | None, ...], torch.Tensor]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    terms = [
        gradient.detach().abs().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not terms:
        return gradients, loss.detach().new_zeros(())
    return gradients, torch.stack(terms).sum().sqrt()


def _gradient_cosine(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
    first_norm: torch.Tensor,
    second_norm: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    terms = [
        torch.sum(torch.conj(left.detach()) * right.detach()).real
        for left, right in zip(first, second)
        if left is not None and right is not None
    ]
    if not terms:
        return first_norm.new_zeros(())
    dot = torch.stack(terms).sum()
    return dot / (first_norm * second_norm).clamp_min(epsilon)


@dataclass(frozen=True)
class BalancedLoss:
    total: torch.Tensor
    final: torch.Tensor
    path: torch.Tensor
    path_weight: float
    gradient_ratio: float
    gradient_cosine: float
    final_gradient_norm: float
    path_gradient_norm: float
    shared_parameter_count: int
    path_details: dict[str, float]


def balanced_blrpng_loss(
    *,
    final_loss: torch.Tensor,
    path_loss: torch.Tensor,
    path_parameters: list[torch.nn.Parameter],
    target_ratio: float,
    epsilon: float = 1e-12,
    path_details: dict[str, float] | None = None,
    conflict_gate: bool = False,
) -> BalancedLoss:
    final_gradients, final_norm = _gradients_and_norm(final_loss, path_parameters)
    path_gradients, path_norm = _gradients_and_norm(path_loss, path_parameters)
    cosine = _gradient_cosine(
        final_gradients,
        path_gradients,
        final_norm,
        path_norm,
        epsilon,
    )
    if (
        float(path_norm) <= epsilon
        or float(final_norm) <= epsilon
        or (conflict_gate and float(cosine) < 0.0)
    ):
        weight = final_loss.detach().new_zeros(())
        achieved = 0.0
    else:
        weight = (target_ratio * final_norm / path_norm.clamp_min(epsilon)).detach()
        achieved = float((weight * path_norm / final_norm.clamp_min(epsilon)).detach())
    return BalancedLoss(
        total=final_loss + weight * path_loss,
        final=final_loss,
        path=path_loss,
        path_weight=float(weight),
        gradient_ratio=achieved,
        gradient_cosine=float(cosine),
        final_gradient_norm=float(final_norm),
        path_gradient_norm=float(path_norm),
        shared_parameter_count=sum(parameter.numel() for parameter in path_parameters),
        path_details=path_details or {},
    )
