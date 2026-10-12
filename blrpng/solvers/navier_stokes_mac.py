from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import torch

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.solvers.progress import select_progress_update
from blrpng.solvers.protocol import SolverRollout


def _lid_profile(x: np.ndarray, amplitude: float, bump: float) -> np.ndarray:
    smooth = 16.0 * x**2 * (1.0 - x) ** 2
    local = np.exp(-80.0 * (x - 0.63) ** 2)
    return amplitude * smooth * (1.0 + bump * local)


class MACSystem:
    """Implicit BDF1 incompressible Navier--Stokes system on a MAC grid."""

    def __init__(self, cells: int, *, dt: float, reynolds: float, lid_amplitude: float, lid_bump: float):
        if cells < 4:
            raise ValueError("MAC grid needs at least four cells per axis.")
        self.n = int(cells)
        self.h = 1.0 / self.n
        self.dt = float(dt)
        self.viscosity = 1.0 / float(reynolds)
        self.lid = _lid_profile(np.linspace(0.0, 1.0, self.n + 1), lid_amplitude, lid_bump)
        self.u_size = (self.n - 1) * self.n
        self.v_size = self.n * (self.n - 1)
        self.p_size = self.n * self.n

    @property
    def size(self) -> int:
        return self.u_size + self.v_size + self.p_size

    def unpack(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        u = np.zeros((n + 1, n), dtype=np.float64)
        v = np.zeros((n, n + 1), dtype=np.float64)
        u[1:n] = vector[: self.u_size].reshape(n - 1, n)
        start = self.u_size
        v[:, 1:n] = vector[start : start + self.v_size].reshape(n, n - 1)
        p = vector[start + self.v_size :].reshape(n, n)
        return u, v, p

    def pack(self, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.concatenate((u[1:self.n].reshape(-1), v[:, 1:self.n].reshape(-1), p.reshape(-1)))

    def residual(self, vector: np.ndarray, previous: np.ndarray) -> np.ndarray:
        n, h, nu = self.n, self.h, self.viscosity
        u, v, p = self.unpack(vector)
        pu, pv, _ = self.unpack(previous)
        ru = np.zeros((n - 1, n), dtype=np.float64)
        rv = np.zeros((n, n - 1), dtype=np.float64)
        for i in range(1, n):
            for j in range(n):
                below = u[i, j - 1] if j > 0 else -u[i, j]
                top_value = 0.5 * (self.lid[i] + self.lid[i + 1])
                above = u[i, j + 1] if j < n - 1 else 2.0 * top_value - u[i, j]
                laplace = (
                    u[i + 1, j] + u[i - 1, j] + above + below - 4.0 * u[i, j]
                ) / h**2
                du_dx = (u[i + 1, j] - u[i - 1, j]) / (2.0 * h)
                du_dy = (above - below) / (2.0 * h)
                v_here = 0.25 * (
                    v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1]
                )
                ru[i - 1, j] = (
                    (u[i, j] - pu[i, j]) / self.dt
                    + u[i, j] * du_dx + v_here * du_dy
                    + (p[i, j] - p[i - 1, j]) / h - nu * laplace
                )
        for i in range(n):
            for j in range(1, n):
                left = v[i - 1, j] if i > 0 else -v[i, j]
                right = v[i + 1, j] if i < n - 1 else -v[i, j]
                laplace = (
                    left + right + v[i, j - 1] + v[i, j + 1] - 4.0 * v[i, j]
                ) / h**2
                dv_dx = (right - left) / (2.0 * h)
                dv_dy = (v[i, j + 1] - v[i, j - 1]) / (2.0 * h)
                u_here = 0.25 * (
                    u[i, j - 1] + u[i + 1, j - 1] + u[i, j] + u[i + 1, j]
                )
                rv[i, j - 1] = (
                    (v[i, j] - pv[i, j]) / self.dt
                    + u_here * dv_dx + v[i, j] * dv_dy
                    + (p[i, j] - p[i, j - 1]) / h - nu * laplace
                )
        divergence = (
            (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
        )
        divergence = divergence.reshape(-1)
        divergence[0] = float(np.mean(p))
        return np.concatenate((ru.reshape(-1), rv.reshape(-1), divergence))

    def divergence(self, vector: np.ndarray) -> np.ndarray:
        u, v, _ = self.unpack(vector)
        return (u[1:] - u[:-1]) / self.h + (v[:, 1:] - v[:, :-1]) / self.h


def _to_collocated(system: MACSystem, vector: np.ndarray) -> np.ndarray:
    n = system.n
    resolution = n + 1
    u, v, p = system.unpack(vector)
    output_u = np.zeros((resolution, resolution), dtype=np.float64)
    output_v = np.zeros_like(output_u)
    output_u[:, 1:n] = 0.5 * (u[:, :-1] + u[:, 1:])
    output_u[:, -1] = system.lid
    output_v[1:n, :] = 0.5 * (v[:-1, :] + v[1:, :])
    centers = (np.arange(n) + 0.5) / n
    nodes = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    pressure = RegularGridInterpolator(
        (centers, centers), p, bounds_error=False, fill_value=None
    )(np.column_stack((xx.reshape(-1), yy.reshape(-1)))).reshape(resolution, resolution)
    pressure -= pressure.mean()
    return np.stack((output_u, output_v, pressure))


def _newton_krylov_step(
    system: MACSystem,
    previous: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_newton: int = 15,
) -> tuple[np.ndarray, list[np.ndarray], list[float], list[int], bool, float]:
    state = previous.copy()
    states = [state.copy()]
    residuals: list[float] = []
    krylov_iterations: list[int] = []
    started = time.perf_counter()
    converged = False
    for _ in range(max_newton):
        residual = system.residual(state, previous)
        norm = float(np.linalg.norm(residual))
        residuals.append(norm)
        if norm <= tolerance:
            converged = True
            break
        epsilon = 1e-7 * (1.0 + np.linalg.norm(state))

        def matvec(direction: np.ndarray) -> np.ndarray:
            scale = epsilon / max(np.linalg.norm(direction), 1e-12)
            return (system.residual(state + scale * direction, previous) - residual) / scale

        operator = spla.LinearOperator((system.size, system.size), matvec=matvec, dtype=np.float64)
        counter = [0]

        def callback(_: np.ndarray) -> None:
            counter[0] += 1

        update, info = spla.gmres(
            operator, -residual, rtol=1e-6, atol=0.0, restart=50,
            maxiter=200, callback=callback, callback_type="pr_norm",
        )
        krylov_iterations.append(counter[0])
        if info != 0 or not np.all(np.isfinite(update)):
            break
        step = 1.0
        accepted = False
        while step >= 1e-4:
            proposal = state + step * update
            if np.linalg.norm(system.residual(proposal, previous)) < norm:
                state = proposal
                states.append(state.copy())
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    final = system.residual(state, previous)
    residuals.append(float(np.linalg.norm(final)))
    converged = converged or residuals[-1] <= tolerance
    return state, states, residuals, krylov_iterations, converged, time.perf_counter() - started


def generate_navier_stokes_rollout(
    *, resolution: int, steps: int, seed: int, case_id: int = 0,
    reynolds: float = 100.0, dt: float = 0.01, lid_amplitude: float = 1.0,
    lid_bump: float = 0.2, fidelity: int = 0, role: str = "smoke",
    progress_threshold: float = 0.25,
) -> SolverRollout:
    system = MACSystem(
        resolution - 1, dt=dt, reynolds=reynolds,
        lid_amplitude=lid_amplitude, lid_bump=lid_bump,
    )
    state = np.zeros(system.size, dtype=np.float64)
    trajectory = [_to_collocated(system, state)]
    corrections, progresses, valid = [], [], []
    nonlinear_iterations, linear_iterations, timings = [], [], []
    converged_all = True
    divergence_max = 0.0
    for _ in range(steps):
        final, states, residuals, krylov, converged, elapsed = _newton_krylov_step(system, state)
        pre, correction, progress, is_valid = select_progress_update(
            states, residuals, progress_threshold
        )
        correction_field = _to_collocated(system, pre + correction) - _to_collocated(system, pre)
        correction_field[:2, 0, :] = 0.0
        correction_field[:2, -1, :] = 0.0
        correction_field[:2, :, 0] = 0.0
        correction_field[:2, :, -1] = 0.0
        correction_field[2] -= correction_field[2].mean()
        corrections.append(correction_field)
        progresses.append([progress]); valid.append([is_valid])
        nonlinear_iterations.append(len(states) - 1)
        linear_iterations.append(sum(krylov))
        timings.append(elapsed)
        state = final
        trajectory.append(_to_collocated(system, state))
        divergence_max = max(divergence_max, float(np.max(np.abs(system.divergence(state)))))
        converged_all = converged_all and converged
    mask = np.zeros((3, resolution, resolution), dtype=np.bool_)
    velocity_boundary = rectangular_boundary_mask(resolution, resolution)
    mask[:2] = velocity_boundary
    boundary = np.zeros((3, resolution, resolution), dtype=np.float64)
    boundary[0, :, -1] = system.lid
    lift = np.zeros_like(boundary)
    lift[0] = harmonic_lift(boundary[0])
    lift[1] = harmonic_lift(boundary[1])
    condition = np.full((1, resolution, resolution), 1.0 / reynolds, dtype=np.float64)
    return SolverRollout(
        trajectory=torch.from_numpy(np.stack(trajectory).astype(np.float32)),
        boundary_values=torch.from_numpy(boundary.astype(np.float32)),
        boundary_lift=torch.from_numpy(lift.astype(np.float32)),
        constraint_mask=torch.from_numpy(mask),
        newton_correction=torch.from_numpy(np.stack(corrections).astype(np.float32)),
        residual_progress=torch.tensor(progresses, dtype=torch.float32),
        valid_correction_mask=torch.tensor(valid, dtype=torch.bool),
        condition_fields=torch.from_numpy(condition.astype(np.float32)),
        ids={
            "case_id": case_id, "pair_id": case_id, "boundary_id": case_id,
            "initial_condition_id": case_id, "rollout_id": case_id,
            "target_task_id": case_id,
        },
        meta={
            "equation": "navier_stokes", "discretization": "MAC",
            "time_integrator": "BDF1", "nonlinear_solver": "Newton-GMRES",
            "role": role, "resolution": resolution, "fidelity": fidelity,
            "field_names": ["u", "v", "p"], "field_scales": [1.0, 1.0, 0.1],
            "solver_converged": converged_all,
            "nonlinear_iterations": nonlinear_iterations,
            "linear_iterations": linear_iterations, "solver_timing": timings,
            "assembly_time": 0.0, "reynolds": reynolds, "dt": dt,
            "divergence_max": divergence_max,
            "pressure_gauge_error": float(abs(trajectory[-1][2].mean())),
            "energy_joules": None,
        },
    )
