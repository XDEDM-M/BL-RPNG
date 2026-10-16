from __future__ import annotations

from collections.abc import Callable
import torch


def supervised_rollout_loss(model, batch: dict[str, torch.Tensor], first_prediction: torch.Tensor,
                            *, horizon: int,
                            loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Mean true LR-window loss with recursive predictions and full BPTT.

    RolloutWindowDataset rejects time-dependent auxiliary inputs and geometry.
    The first prediction is supplied by the caller; no teacher forcing or
    detachment is applied to later predicted states.
    """
    if horizon < 1:
        raise ValueError("rollout supervision horizon must be positive")
    first_loss = loss_function(first_prediction, batch["y"])
    if horizon == 1:
        return first_loss
    future = batch.get("future_targets")
    if future is None or future.ndim != 5 or future.shape[1] < horizon:
        raise ValueError("multi-step supervision requires [B,H,C,X,Y] future_targets")
    if not torch.equal(future[:, 0], batch["y"]):
        raise ValueError("first future target must exactly equal the one-step label")
    state = first_prediction
    auxiliary = batch["x"][:, state.shape[1]:]
    losses = [first_loss]
    for step in range(1, horizon):
        state, _ = model(torch.cat((state, auxiliary), dim=1),
                         batch["boundary_lift"], batch["boundary_mask"])
        losses.append(loss_function(state, future[:, step]))
    return torch.stack(losses).mean()
