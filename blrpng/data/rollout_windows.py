from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Dataset

from blrpng.data.shards import RolloutShardDataset


class RolloutWindowDataset(Dataset):
    """Expose same-rollout future targets without changing the v4 shard loader.

    The wrapped dataset remains the source of every transition-level field.  This
    wrapper only removes starting steps that cannot supply ``horizon`` targets and
    adds ``future_targets`` with shape ``[H, C, height, width]``.  At horizon one,
    the wrapped dataset's length, ordering, and original field values are exactly
    preserved.

    Multi-step windows currently support static boundary geometry and static
    condition fields only.  A future engine must add per-step auxiliary inputs
    before time-dependent conditions or boundaries can be trained safely.
    """

    def __init__(self, dataset: RolloutShardDataset, *, horizon: int):
        if not isinstance(dataset, RolloutShardDataset):
            raise TypeError("RolloutWindowDataset requires a RolloutShardDataset.")
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
            raise ValueError("rollout window horizon must be a positive integer.")

        self.dataset = dataset
        self.horizon = int(horizon)
        # Preserve the attributes used by the engine and RolloutBatchSampler so
        # integration can remain a small wrapper around the existing dataset.
        self.index = dataset.index
        self.index_path = dataset.index_path
        self.cache_rollouts = dataset.cache_rollouts

        self._base_index_by_sample: dict[tuple[int, int], int] = {}
        self.samples: list[tuple[int, int]] = []
        for base_index, (shard_index, step) in enumerate(dataset.samples):
            key = (int(shard_index), int(step))
            if key in self._base_index_by_sample:
                raise ValueError(f"duplicate rollout transition in wrapped dataset: {key}")
            self._base_index_by_sample[key] = base_index
            available_steps = int(dataset.index["shards"][shard_index]["steps"])
            if step + self.horizon <= available_steps:
                self.samples.append(key)

        if not self.samples:
            raise ValueError(
                f"no rollout contains a valid horizon-{self.horizon} training window"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _reject_unsupported_time_dependence(payload: dict[str, Any]) -> None:
        conditions = payload.get("condition_fields")
        if isinstance(conditions, torch.Tensor) and conditions.ndim == 4:
            raise NotImplementedError(
                "multi-step rollout windows do not yet support time-dependent "
                "condition_fields; future per-step auxiliary inputs are required"
            )
        for key in ("boundary_values", "boundary_lift", "constraint_mask"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor) and value.ndim != 3:
                raise NotImplementedError(
                    "multi-step rollout windows require static boundary fields; "
                    f"{key} has shape {tuple(value.shape)}"
                )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, step = self.samples[index]
        base_index = self._base_index_by_sample[(shard_index, step)]
        payload = self.dataset._load(shard_index)
        if self.horizon > 1:
            self._reject_unsupported_time_dependence(payload)

        sample = dict(self.dataset[base_index])
        future = payload["trajectory"][step + 1 : step + self.horizon + 1]
        if int(future.shape[0]) != self.horizon:
            raise RuntimeError(
                "rollout window crossed a shard boundary or encountered truncated data"
            )
        sample["future_targets"] = future.contiguous()
        return sample
