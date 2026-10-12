"""Diagnostic no-slip-preserving Cartesian divergence projection.

This is an analysis-only postprocessor, not a substitute for validating the
underlying finite-element solve or its Newton path.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsmr


def grid_divergence(velocity: np.ndarray) -> np.ndarray:
    if velocity.ndim != 3 or velocity.shape[0] != 2 or velocity.shape[1] != velocity.shape[2]:
        raise ValueError("Expected velocity [2,N,N]")
    h = 1.0 / (velocity.shape[-1] - 1)
    u, v = velocity
    return (
        (u[2:, 1:-1] - u[:-2, 1:-1])
        + (v[1:-1, 2:] - v[1:-1, :-2])
    ) / (2.0 * h)


@lru_cache(maxsize=3)
def _interior_divergence_matrix(n: int) -> csr_matrix:
    if n < 5:
        raise ValueError("Resolution must be at least 5")
    side = n - 2
    pixels = side * side
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    factor = (n - 1) / 2.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = (i - 1) * side + (j - 1)
            for neighbor, sign in ((i + 1, 1.0), (i - 1, -1.0)):
                if 1 <= neighbor < n - 1:
                    rows.append(row)
                    cols.append((neighbor - 1) * side + (j - 1))
                    data.append(sign * factor)
            for neighbor, sign in ((j + 1, 1.0), (j - 1, -1.0)):
                if 1 <= neighbor < n - 1:
                    rows.append(row)
                    cols.append(pixels + (i - 1) * side + (neighbor - 1))
                    data.append(sign * factor)
    return coo_matrix((data, (rows, cols)), shape=(pixels, 2 * pixels)).tocsr()


def project_no_slip_divergence(
    velocity: np.ndarray, *, tolerance: float = 1e-8, max_iterations: int = 2000
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Find the least-L2 interior correction for central-difference continuity."""
    original = np.asarray(velocity, dtype=np.float64)
    divergence_before = grid_divergence(original)
    n = original.shape[-1]
    matrix = _interior_divergence_matrix(n)
    solution = lsmr(
        matrix,
        -divergence_before.reshape(-1),
        atol=tolerance,
        btol=tolerance,
        maxiter=max_iterations,
    )
    correction = solution[0].reshape(2, n - 2, n - 2)
    projected = original.copy()
    projected[:, 1:-1, 1:-1] += correction
    divergence_after = grid_divergence(projected)
    metadata = {
        "resolution": n,
        "iterations": int(solution[2]),
        "lsmr_stop_code": int(solution[1]),
        "divergence_rms_before": float(np.sqrt(np.mean(divergence_before**2))),
        "divergence_rms_after": float(np.sqrt(np.mean(divergence_after**2))),
        "relative_velocity_change": float(np.linalg.norm(correction) / max(np.linalg.norm(original), 1e-30)),
        "boundary_change_max": float(np.max(np.abs(projected[:, (0, -1), :] - original[:, (0, -1), :]))),
    }
    return projected, metadata
