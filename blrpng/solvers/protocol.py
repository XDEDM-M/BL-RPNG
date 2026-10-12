from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from blrpng.data.shards import SHARD_SCHEMA, validate_shard


@dataclass(frozen=True)
class SolverRollout:
    trajectory: torch.Tensor
    boundary_values: torch.Tensor
    boundary_lift: torch.Tensor
    constraint_mask: torch.Tensor
    newton_correction: torch.Tensor
    residual_progress: torch.Tensor
    valid_correction_mask: torch.Tensor
    condition_fields: torch.Tensor
    ids: dict[str, int]
    meta: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        value = {
            "schema": SHARD_SCHEMA,
            "trajectory": self.trajectory,
            "boundary_values": self.boundary_values,
            "boundary_lift": self.boundary_lift,
            "constraint_mask": self.constraint_mask,
            "newton_correction": self.newton_correction,
            "residual_progress": self.residual_progress,
            "valid_correction_mask": self.valid_correction_mask,
            "condition_fields": self.condition_fields,
            "ids": self.ids,
            "meta": self.meta,
        }
        validate_shard(value)
        return value
