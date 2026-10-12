"""Label-free lifted cross-resolution consistency for Gray--Scott adapters."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.solvers.gray_scott import _boundary_fields


def refined_geometry(payload, resolution, device):
    """Use the prescribed analytic boundary and coefficients, never fine solutions."""
    meta = payload['meta']
    axis = np.linspace(0, 1, resolution)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    lo, hi = meta['frequency_range']
    values = np.stack(_boundary_fields(x, y, seed=meta['boundary_seed'],
                    correlation=meta['correlation'], amplitude=meta['amplitude'],
                    frequency_low=lo, frequency_high=hi))
    mask = rectangular_boundary_mask(resolution, resolution)
    lifts = np.stack([harmonic_lift(np.where(mask, v, 0)) for v in values])
    lift = torch.tensor(lifts, dtype=torch.float32, device=device)[None]
    mask_t = torch.tensor(np.broadcast_to(mask, values.shape).copy(), device=device)[None]
    conditions = F.interpolate(payload['condition_fields'][None].to(device),
                               size=(resolution, resolution), mode='bilinear', align_corners=True)
    grid = torch.tensor(np.stack((x,y)), dtype=torch.float32, device=device)[None]
    return lift, mask_t, conditions, grid


def prolong_lifted(state, coarse_lift, fine_geo):
    lift, mask, _, _ = fine_geo
    residual = F.interpolate(state - coarse_lift, size=lift.shape[-2:], mode='bilinear', align_corners=True)
    return residual.masked_fill(mask, 0) + lift


def field_commutator(fine, coarse, reference, mask):
    """Symmetric gradients; denominator uses LR labels, not a learnable norm."""
    restricted = F.interpolate(fine, size=coarse.shape[-2:], mode='bilinear', align_corners=True)
    difference = (restricted - coarse).masked_fill(mask, 0)
    numerator = torch.linalg.vector_norm(difference, dim=(-2,-1))
    denominator = torch.linalg.vector_norm(reference, dim=(-2,-1)).clamp_min(
        1e-3 * (reference.shape[-2] * reference.shape[-1])**.5)
    return (numerator / denominator).mean()
