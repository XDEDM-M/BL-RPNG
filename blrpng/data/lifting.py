from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def rectangular_boundary_mask(height: int, width: int) -> np.ndarray:
    if min(height, width) < 3:
        raise ValueError("A harmonic lifting grid needs at least 3 points per axis.")
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


@lru_cache(maxsize=16)
def _laplace_factor(height: int, width: int):
    inner_h, inner_w = height - 2, width - 2
    tx = sp.diags(
        [-np.ones(inner_w - 1), 2.0 * np.ones(inner_w), -np.ones(inner_w - 1)],
        offsets=[-1, 0, 1],
        format="csc",
    )
    ty = sp.diags(
        [-np.ones(inner_h - 1), 2.0 * np.ones(inner_h), -np.ones(inner_h - 1)],
        offsets=[-1, 0, 1],
        format="csc",
    )
    matrix = sp.kron(sp.eye(inner_h, format="csc"), tx) + sp.kron(
        ty, sp.eye(inner_w, format="csc")
    )
    return spla.factorized(matrix.tocsc())


def harmonic_lift(boundary_field: np.ndarray) -> np.ndarray:
    values = np.asarray(boundary_field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [H,W] boundary field, got {values.shape}")
    height, width = values.shape
    lift = np.zeros_like(values)
    mask = rectangular_boundary_mask(height, width)
    lift[mask] = values[mask]

    rhs = np.zeros((height - 2, width - 2), dtype=np.float64)
    rhs[0, :] += lift[0, 1:-1]
    rhs[-1, :] += lift[-1, 1:-1]
    rhs[:, 0] += lift[1:-1, 0]
    rhs[:, -1] += lift[1:-1, -1]
    lift[1:-1, 1:-1] = _laplace_factor(height, width)(rhs.reshape(-1)).reshape(
        height - 2, width - 2
    )
    return lift


def discrete_laplace_interior(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field)
    return (
        values[1:-1, 2:]
        + values[1:-1, :-2]
        + values[2:, 1:-1]
        + values[:-2, 1:-1]
        - 4.0 * values[1:-1, 1:-1]
    )

