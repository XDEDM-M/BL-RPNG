from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.solvers.protocol import SolverRollout


def _negative_laplacian(resolution: int) -> sp.csr_matrix:
    inner = resolution - 2
    h = 1.0 / (resolution - 1)
    line = sp.diags(
        (-np.ones(inner - 1), 2.0 * np.ones(inner), -np.ones(inner - 1)),
        offsets=(-1, 0, 1),
    )
    return ((sp.kron(sp.eye(inner), line) + sp.kron(line, sp.eye(inner))) / h**2).tocsr()


def poisson_boundary(
    resolution: int,
    seed: int,
    family: str = "fourier",
    *,
    regime: str = "source",
) -> np.ndarray:
    """Sample a resolution-consistent Dirichlet field.

    ``source`` contains the boundary distribution used by pretraining.  The
    ``target`` variants deliberately leave that distribution while preserving
    the same PDE and forcing distribution.  Keeping the boundary and forcing
    RNG streams separate is essential: all support/query members of one target
    task share exactly one boundary but have independent forcing fields.
    """
    if regime not in {"source", "target"}:
        raise ValueError(f"Unknown Poisson boundary regime: {regime}")
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    if family == "fourier":
        frequency_pool = np.arange(1, 5) if regime == "source" else np.arange(5, 9)
        kx, ky = rng.choice(frequency_pool, size=2, replace=True)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        boundary = 0.35 * (
            np.sin(np.pi * kx * xx + phase) * np.cos(np.pi * ky * yy)
            + 0.35 * np.cos(np.pi * (kx * xx + ky * yy) - phase)
        )
    elif family == "bump":
        center = rng.uniform(0.2, 0.8, size=2)
        sharpness = 32.0 if regime == "source" else 150.0
        edge = int(rng.integers(0, 4))
        distances = (
            (xx - center[0]) ** 2 + yy**2,
            (xx - center[0]) ** 2 + (1.0 - yy) ** 2,
            xx**2 + (yy - center[1]) ** 2,
            (1.0 - xx) ** 2 + (yy - center[1]) ** 2,
        )
        boundary = 0.5 * np.exp(-sharpness * distances[edge])
        boundary += 0.12 * (xx - yy)
    elif family == "piecewise_smooth":
        steepness = 8.0 if regime == "source" else 28.0
        offset = rng.uniform(0.42, 0.68)
        slope = rng.uniform(-0.65, 0.65)
        boundary = 0.3 * np.tanh(steepness * (xx + slope * yy - offset))
    else:
        raise ValueError(f"Unknown Poisson boundary family: {family}")
    return boundary


def poisson_forcing(resolution: int, seed: int) -> np.ndarray:
    """Sample the common source/target forcing distribution."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    fx, fy = rng.integers(1, 5, size=2)
    amplitude = rng.uniform(0.75, 1.25)
    center = rng.uniform(0.25, 0.75, size=2)
    forcing = amplitude * np.sin(np.pi * fx * xx) * np.sin(np.pi * fy * yy)
    forcing += 0.4 * np.exp(
        -35.0 * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    )
    forcing += 0.15 * np.cos(np.pi * (fx + 1) * xx) * np.sin(np.pi * (fy + 1) * yy)
    return forcing


def poisson_fields(
    resolution: int,
    seed: int,
    family: str = "fourier",
    *,
    regime: str = "source",
) -> tuple[np.ndarray, np.ndarray]:
    boundary = poisson_boundary(resolution, seed, family, regime=regime)
    forcing = poisson_forcing(resolution, seed + 1_000_003)
    return boundary, forcing


def solve_poisson(
    boundary_field: np.ndarray, forcing: np.ndarray
) -> tuple[np.ndarray, float, float]:
    if boundary_field.shape != forcing.shape or boundary_field.ndim != 2:
        raise ValueError("Poisson boundary/forcing must have one identical square grid.")
    resolution = boundary_field.shape[0]
    mask = rectangular_boundary_mask(resolution, resolution)
    boundary = np.zeros_like(boundary_field, dtype=np.float64)
    boundary[mask] = boundary_field[mask]
    h = 1.0 / (resolution - 1)
    started = time.perf_counter()
    matrix = _negative_laplacian(resolution)
    assembly_seconds = time.perf_counter() - started
    rhs = forcing[1:-1, 1:-1].copy()
    rhs[0, :] += boundary[0, 1:-1] / h**2
    rhs[-1, :] += boundary[-1, 1:-1] / h**2
    rhs[:, 0] += boundary[1:-1, 0] / h**2
    rhs[:, -1] += boundary[1:-1, -1] / h**2
    started = time.perf_counter()
    interior = spla.spsolve(matrix, rhs.reshape(-1))
    solve_seconds = time.perf_counter() - started
    solution = boundary.copy()
    solution[1:-1, 1:-1] = interior.reshape(resolution - 2, resolution - 2)
    return solution, assembly_seconds, solve_seconds


def generate_poisson_rollout(
    *, resolution: int, seed: int, case_id: int = 0,
    family: str = "fourier", fidelity: int = 0, role: str = "support",
    boundary_seed: int | None = None, forcing_seed: int | None = None,
    regime: str = "source", target_task_id: int | None = None,
    boundary_id: int | None = None, forcing_id: int | None = None,
    pair_id: int | None = None,
) -> SolverRollout:
    boundary_seed = int(seed if boundary_seed is None else boundary_seed)
    forcing_seed = int(seed + 1_000_003 if forcing_seed is None else forcing_seed)
    boundary = poisson_boundary(resolution, boundary_seed, family, regime=regime)
    forcing = poisson_forcing(resolution, forcing_seed)
    solution, assembly_seconds, solve_seconds = solve_poisson(boundary, forcing)
    mask_np = rectangular_boundary_mask(resolution, resolution)
    boundary_values = np.where(mask_np, boundary, 0.0)
    lift = harmonic_lift(boundary_values)
    correction = solution - lift
    correction[mask_np] = 0.0
    residual = _negative_laplacian(resolution) @ solution[1:-1, 1:-1].reshape(-1)
    h = 1.0 / (resolution - 1)
    rhs = forcing[1:-1, 1:-1].copy()
    rhs[0] += boundary_values[0, 1:-1] / h**2
    rhs[-1] += boundary_values[-1, 1:-1] / h**2
    rhs[:, 0] += boundary_values[1:-1, 0] / h**2
    rhs[:, -1] += boundary_values[1:-1, -1] / h**2
    residual_norm = float(np.linalg.norm(residual - rhs.reshape(-1)))
    return SolverRollout(
        trajectory=torch.from_numpy(np.stack((lift, solution))[:, None].astype(np.float32)),
        boundary_values=torch.from_numpy(boundary_values[None].astype(np.float32)),
        boundary_lift=torch.from_numpy(lift[None].astype(np.float32)),
        constraint_mask=torch.from_numpy(mask_np[None]),
        newton_correction=torch.from_numpy(correction[None, None].astype(np.float32)),
        residual_progress=torch.ones(1, 1, dtype=torch.float32),
        valid_correction_mask=torch.ones(1, 1, dtype=torch.bool),
        condition_fields=torch.from_numpy(forcing[None].astype(np.float32)),
        ids={
            "case_id": case_id,
            "pair_id": case_id if pair_id is None else int(pair_id),
            "boundary_id": case_id if boundary_id is None else int(boundary_id),
            "initial_condition_id": case_id if forcing_id is None else int(forcing_id),
            "rollout_id": case_id,
            "target_task_id": case_id if target_task_id is None else int(target_task_id),
        },
        meta={
            "equation": "poisson", "role": role, "resolution": resolution,
            "fidelity": fidelity, "field_names": ["solution"], "field_scales": [1.0],
            "solver_converged": residual_norm < 1e-7,
            "nonlinear_iterations": [1], "linear_iterations": [1],
            "solver_timing": [solve_seconds], "assembly_time": assembly_seconds,
            "residual_norm": residual_norm, "boundary_family": family,
            "boundary_regime": regime, "boundary_seed": boundary_seed,
            "forcing_seed": forcing_seed,
            "energy_joules": None,
        },
    )
