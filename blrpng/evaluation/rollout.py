from __future__ import annotations

from collections import defaultdict

import torch

from blrpng.data.schema import DatasetBundle
from blrpng.data.shards import read_index
from blrpng.models.fno import DirichletOperator
from blrpng.paths import resolve_in_project
from blrpng.training.losses import per_sample_relative_l2, per_sample_field_relative_l2


@torch.no_grad()
def projected_rollout(
    model: DirichletOperator,
    bundle: DatasetBundle,
    *,
    device: torch.device,
    horizon: int | None = None,
    rollout_batch_size: int = 16,
) -> dict[str, float | list[float]]:
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive.")
    model.eval()
    tensors = bundle.tensors
    rollout_ids = tensors["rollout_id"]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, rollout_id in enumerate(rollout_ids.tolist()):
        groups[int(rollout_id)].append(index)
    rollout_errors: list[float] = []
    step_errors: list[float] = []
    mse_values: list[float] = []
    sequences = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda idx: int(tensors["local_step_id"][idx]))
        sequences.append(ordered if horizon is None else ordered[:horizon])
    for batch_start in range(0, len(sequences), rollout_batch_size):
        batch_sequences = sequences[batch_start : batch_start + rollout_batch_size]
        lengths = {len(indices) for indices in batch_sequences}
        if len(lengths) != 1:
            raise ValueError("Batched rollout requires equal horizons within a dataset.")
        steps = next(iter(lengths))
        first_indices = torch.tensor(
            [indices[0] for indices in batch_sequences], dtype=torch.long
        )
        current = tensors["x"].index_select(0, first_indices).to(device)
        batch_errors: list[list[float]] = [[] for _ in batch_sequences]
        batch_mses: list[list[float]] = [[] for _ in batch_sequences]
        for local_step in range(steps):
            cpu_indices = torch.tensor(
                [indices[local_step] for indices in batch_sequences], dtype=torch.long
            )
            lift = tensors["boundary_lift"].index_select(0, cpu_indices).to(device)
            mask = tensors["boundary_mask"].index_select(0, cpu_indices).to(device)
            target = tensors["y"].index_select(0, cpu_indices).to(device)
            prediction, _ = model(current, lift, mask)
            errors = per_sample_relative_l2(prediction, target)
            mses = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
            for local_rollout in range(len(batch_sequences)):
                batch_errors[local_rollout].append(float(errors[local_rollout]))
                batch_mses[local_rollout].append(float(mses[local_rollout]))
            fields = model.config.out_channels
            auxiliary = tensors["x"].index_select(0, cpu_indices)[:, fields:].to(device)
            current = torch.cat((prediction, auxiliary), dim=1)
        for errors, mses in zip(batch_errors, batch_mses):
            rollout_errors.append(sum(errors) / max(len(errors), 1))
            step_errors.extend(errors)
            mse_values.extend(mses)
    return {
        "rollout_rel_l2": float(sum(rollout_errors) / max(len(rollout_errors), 1)),
        "rollout_mse": float(sum(mse_values) / max(len(mse_values), 1)),
        "per_rollout_rel_l2": rollout_errors,
        "per_step_rel_l2": step_errors,
    }


@torch.no_grad()
def projected_rollout_index(
    model: DirichletOperator,
    index_path: str,
    *,
    device: torch.device,
    horizon: int | None = None,
    rollout_batch_size: int = 16,
    field_balanced: bool = False,
) -> dict[str, float | list[float] | list[int]]:
    """Roll out v4 atomic shards without materializing repeated transitions."""
    index = read_index(index_path)
    rows = index["shards"]
    rollout_errors, step_errors, mse_values, initial_ids = [], [], [], []
    model.eval()
    for start in range(0, len(rows), rollout_batch_size):
        payloads = [torch.load(resolve_in_project(row["path"], must_exist=True),
                               map_location="cpu", weights_only=False)
                    for row in rows[start:start + rollout_batch_size]]
        lengths = {int(payload["trajectory"].shape[0] - 1) for payload in payloads}
        resolutions = {int(payload["trajectory"].shape[-1]) for payload in payloads}
        if len(lengths) != 1 or len(resolutions) != 1:
            raise ValueError("Batched shard rollout requires equal horizons and resolutions.")
        steps = min(next(iter(lengths)), horizon) if horizon is not None else next(iter(lengths))
        resolution = next(iter(resolutions))
        axis = torch.linspace(0, 1, resolution, device=device)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((xx, yy)).unsqueeze(0).expand(len(payloads), -1, -1, -1)
        condition_payloads = [
            payload.get("condition_fields", payload["trajectory"].new_empty(
                (0, resolution, resolution)
            ))
            for payload in payloads
        ]
        current = torch.stack([payload["trajectory"][0] for payload in payloads]).to(device)
        lift = torch.stack([payload["boundary_lift"] for payload in payloads]).to(device)
        mask = torch.stack([payload["constraint_mask"] for payload in payloads]).to(device)
        batch_errors = [[] for _ in payloads]
        for step in range(steps):
            conditions = torch.stack([
                value[step] if value.ndim == 4 else value
                for value in condition_payloads
            ]).to(device)
            target = torch.stack([payload["trajectory"][step + 1] for payload in payloads]).to(device)
            prediction, _ = model(torch.cat((current, grid, conditions), dim=1), lift, mask)
            errors = (per_sample_field_relative_l2(prediction, target).mean(dim=1)
                      if field_balanced else per_sample_relative_l2(prediction, target))
            mses = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
            for local in range(len(payloads)):
                value = float(errors[local])
                batch_errors[local].append(value); step_errors.append(value)
                mse_values.append(float(mses[local]))
            current = prediction
        rollout_errors.extend(sum(values) / max(len(values), 1) for values in batch_errors)
        initial_ids.extend(int(payload["ids"]["initial_condition_id"]) for payload in payloads)
    return {
        "rollout_rel_l2": float(sum(rollout_errors) / max(len(rollout_errors), 1)),
        "rollout_mse": float(sum(mse_values) / max(len(mse_values), 1)),
        "per_rollout_rel_l2": rollout_errors, "per_step_rel_l2": step_errors,
        "initial_condition_ids": initial_ids,
    }
