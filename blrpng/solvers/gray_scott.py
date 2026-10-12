from __future__ import annotations

import time

import numpy as np
import torch

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.solvers.progress import select_progress_update
from blrpng.solvers.protocol import SolverRollout
from blrpng.solvers.sfepy_gray_scott_worker import _newton_step, _state_to_fields


def _boundary_fields(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    seed: int,
    correlation: float,
    amplitude: float,
    frequency_low: int = 1,
    frequency_high: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if frequency_low < 1 or frequency_high < frequency_low:
        raise ValueError("Invalid Gray--Scott boundary frequency range.")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("Gray--Scott boundary correlation must be in [-1,1].")
    rng = np.random.default_rng(seed)
    frequencies = rng.integers(frequency_low, frequency_high + 1, size=4)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    first = np.sin(np.pi * frequencies[0] * xx + phases[0]) * np.cos(
        np.pi * frequencies[1] * yy + phases[1]
    )
    independent = np.cos(np.pi * frequencies[2] * xx + phases[2]) * np.sin(
        np.pi * frequencies[3] * yy + phases[3]
    )
    second = correlation * first + np.sqrt(max(1.0 - correlation**2, 0.0)) * independent
    first /= max(np.max(np.abs(first)), 1e-12)
    second /= max(np.max(np.abs(second)), 1e-12)
    return 0.82 + amplitude * first, 0.08 + 0.4 * amplitude * second


def generate_gray_scott_rollout(
    *,
    resolution: int,
    steps: int,
    seed: int,
    case_id: int = 0,
    correlation: float = 0.5,
    amplitude: float = 0.1,
    dt: float = 1.0,
    diffusion_u: float = 2e-5,
    diffusion_v: float = 1e-5,
    feed: float = 0.04,
    kill: float = 0.06,
    fidelity: int = 0,
    role: str = "smoke",
    progress_threshold: float = 0.25,
    boundary_seed: int | None = None,
    initial_seed: int | None = None,
    boundary_regime: str = "source",
    frequency_low: int | None = None,
    frequency_high: int | None = None,
    target_task_id: int | None = None,
    boundary_id: int | None = None,
    initial_condition_id: int | None = None,
    pair_id: int | None = None,
) -> SolverRollout:
    boundary_seed = int(seed if boundary_seed is None else boundary_seed)
    initial_seed = int(seed + 17 if initial_seed is None else initial_seed)
    if boundary_regime not in {
        "source", "high_frequency", "amplitude_ood", "correlation_shift",
        "parameter_ood",
    }:
        raise ValueError(f"Unknown Gray--Scott boundary regime: {boundary_regime}")
    if frequency_low is None or frequency_high is None:
        if boundary_regime in {"high_frequency", "parameter_ood"}:
            frequency_low, frequency_high = 5, 8
        else:
            frequency_low, frequency_high = 1, 4
    axis = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    boundary_u, boundary_v = _boundary_fields(
        xx, yy, seed=boundary_seed, correlation=correlation, amplitude=amplitude,
        frequency_low=int(frequency_low), frequency_high=int(frequency_high),
    )
    mask = rectangular_boundary_mask(resolution, resolution)
    boundary_values = np.stack(
        (np.where(mask, boundary_u, 0.0), np.where(mask, boundary_v, 0.0))
    )
    lift = np.stack((harmonic_lift(boundary_values[0]), harmonic_lift(boundary_values[1])))
    rng = np.random.default_rng(initial_seed)
    cx, cy = rng.uniform(0.35, 0.65, size=2)
    sharpness = rng.uniform(55.0, 110.0)
    bump = np.exp(-sharpness * ((xx - cx) ** 2 + (yy - cy) ** 2))
    phase = rng.uniform(0.0, 2.0 * np.pi)
    perturbation = 0.015 * np.sin(2.0 * np.pi * xx + phase) * np.sin(
        2.0 * np.pi * yy - phase
    )
    previous_u = 1.0 - rng.uniform(0.32, 0.48) * bump + perturbation
    previous_v = rng.uniform(0.20, 0.32) * bump - 0.5 * perturbation
    previous_u[mask], previous_v[mask] = boundary_u[mask], boundary_v[mask]
    trajectory = [np.stack((previous_u, previous_v))]
    corrections, progresses, valid, iterations, timings, final_residuals = [], [], [], [], [], []
    converged_all = True
    for _ in range(steps):
        started = time.perf_counter()
        final_u, final_v, states, residuals, converged = _newton_step(
            previous_u, previous_v, boundary_u, boundary_v,
            dt=dt, diffusion_u=diffusion_u, diffusion_v=diffusion_v,
            feed=feed, kill=kill, max_iters=20, tolerance=1e-8,
        )
        timings.append(time.perf_counter() - started)
        final_residuals.append(float(residuals[-1]))
        pre, correction, progress, is_valid = select_progress_update(
            states, residuals, progress_threshold
        )
        pre_u, pre_v = _state_to_fields(pre, boundary_u, boundary_v)
        post_u, post_v = _state_to_fields(pre + correction, boundary_u, boundary_v)
        selected = np.stack((post_u - pre_u, post_v - pre_v))
        selected[:, mask] = 0.0
        corrections.append(selected)
        progresses.append([progress])
        valid.append([is_valid])
        iterations.append(len(states) - 1)
        trajectory.append(np.stack((final_u, final_v)))
        previous_u, previous_v = final_u, final_v
        converged_all = converged_all and converged
    condition = np.stack(
        (
            np.full_like(xx, feed),
            np.full_like(xx, kill),
            np.full_like(xx, correlation),
        )
    )
    return SolverRollout(
        trajectory=torch.from_numpy(np.stack(trajectory).astype(np.float32)),
        boundary_values=torch.from_numpy(boundary_values.astype(np.float32)),
        boundary_lift=torch.from_numpy(lift.astype(np.float32)),
        constraint_mask=torch.from_numpy(np.broadcast_to(mask, boundary_values.shape).copy()),
        newton_correction=torch.from_numpy(np.stack(corrections).astype(np.float32)),
        residual_progress=torch.tensor(progresses, dtype=torch.float32),
        valid_correction_mask=torch.tensor(valid, dtype=torch.bool),
        condition_fields=torch.from_numpy(condition.astype(np.float32)),
        ids={
            "case_id": case_id,
            "pair_id": case_id if pair_id is None else int(pair_id),
            "boundary_id": case_id if boundary_id is None else int(boundary_id),
            "initial_condition_id": (
                case_id if initial_condition_id is None else int(initial_condition_id)
            ),
            "rollout_id": case_id,
            "target_task_id": case_id if target_task_id is None else int(target_task_id),
        },
        meta={
            "equation": "gray_scott", "role": role, "resolution": resolution,
            "fidelity": fidelity, "field_names": ["u", "v"],
            "field_scales": [1.0, 0.25], "solver_converged": converged_all,
            "nonlinear_iterations": iterations, "linear_iterations": [1] * steps,
            "solver_timing": timings, "assembly_time": 0.0,
            "correlation": correlation, "feed": feed, "kill": kill,
            "amplitude": amplitude, "boundary_regime": boundary_regime,
            "frequency_range": [int(frequency_low), int(frequency_high)],
            "boundary_seed": boundary_seed, "initial_seed": initial_seed,
            "final_residual_norms": final_residuals,
            "energy_joules": None,
        },
    )
