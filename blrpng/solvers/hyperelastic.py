from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.solvers.progress import select_progress_update
from blrpng.solvers.protocol import SolverRollout


_NODE_SIGNS = np.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1)), dtype=np.float64)
_GAUSS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


def _shape_gradients(xi: float, eta: float, h: float) -> np.ndarray:
    sx, sy = _NODE_SIGNS[:, 0], _NODE_SIGNS[:, 1]
    dxi = 0.25 * sx * (1.0 + sy * eta)
    deta = 0.25 * sy * (1.0 + sx * xi)
    return np.stack((2.0 * dxi / h, 2.0 * deta / h), axis=1)


def _constitutive(f: np.ndarray, mu: float, lam: float) -> tuple[np.ndarray, np.ndarray, float]:
    determinant = float(np.linalg.det(f))
    if determinant <= 1e-8:
        raise FloatingPointError("Neo-Hookean deformation has non-positive Jacobian.")
    inverse_t = np.linalg.inv(f).T
    log_j = np.log(determinant)
    p = mu * (f - inverse_t) + lam * log_j * inverse_t
    tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    tangent[i, j, k, ell] = (
                        mu * float(i == k and j == ell)
                        + (mu - lam * log_j) * inverse_t[i, ell] * inverse_t[k, j]
                        + lam * inverse_t[i, j] * inverse_t[k, ell]
                    )
    energy = 0.5 * mu * (np.sum(f * f) - 2.0 - 2.0 * log_j) + 0.5 * lam * log_j**2
    return p, tangent, float(energy)


def _assemble(
    displacement: np.ndarray,
    mu_nodes: np.ndarray,
    lambda_nodes: np.ndarray,
    free_map: np.ndarray,
) -> tuple[np.ndarray, sp.csr_matrix, float, float]:
    n = displacement.shape[1]
    h = 1.0 / (n - 1)
    free_count = int(free_map.max()) + 1
    residual = np.zeros(free_count, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    energy = 0.0
    min_j = float("inf")
    for ix in range(n - 1):
        for iy in range(n - 1):
            nodes = ((ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1))
            u_e = np.asarray([[displacement[0, i, j], displacement[1, i, j]] for i, j in nodes])
            mu = float(np.mean([mu_nodes[i, j] for i, j in nodes]))
            lam = float(np.mean([lambda_nodes[i, j] for i, j in nodes]))
            for xi in _GAUSS:
                for eta in _GAUSS:
                    gradients = _shape_gradients(xi, eta, h)
                    f = np.eye(2) + u_e.T @ gradients
                    p, tangent, density = _constitutive(f, mu, lam)
                    determinant = float(np.linalg.det(f))
                    min_j = min(min_j, determinant)
                    weight = h * h / 4.0
                    energy += density * weight
                    for a, (ai, aj) in enumerate(nodes):
                        for component in range(2):
                            row = int(free_map[component, ai, aj])
                            if row < 0:
                                continue
                            residual[row] += np.einsum("j,j->", p[component], gradients[a]) * weight
                            for b, (bi, bj) in enumerate(nodes):
                                for other in range(2):
                                    col = int(free_map[other, bi, bj])
                                    if col < 0:
                                        continue
                                    value = np.einsum(
                                        "j,jl,l->",
                                        gradients[a], tangent[component, :, other, :], gradients[b],
                                    ) * weight
                                    rows.append(row); cols.append(col); values.append(float(value))
    tangent_matrix = sp.coo_matrix(
        (values, (rows, cols)), shape=(free_count, free_count)
    ).tocsr()
    return residual, tangent_matrix, energy, min_j


def _boundary_displacement(resolution: int, load: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    phase = rng.uniform(0.0, 2.0 * np.pi)
    ux = load * (0.045 * xx + 0.008 * xx * np.sin(2.0 * np.pi * yy + phase))
    uy = load * (0.025 * yy + 0.006 * yy * np.sin(2.0 * np.pi * xx - phase))
    return np.stack((ux, uy))


def _material_fields(resolution: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 101)
    axis = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    center = rng.uniform(0.3, 0.7, size=2)
    inclusion = np.exp(-45.0 * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    mu = 1.0 + 0.8 * inclusion
    lam = 2.0 + 1.2 * inclusion
    return mu, lam


def solve_hyperelastic(
    boundary: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 20,
) -> tuple[np.ndarray, list[np.ndarray], list[float], list[int], float, float, bool]:
    fields, n, _ = boundary.shape
    if fields != 2:
        raise ValueError("Hyperelastic displacement must contain ux and uy.")
    mask = rectangular_boundary_mask(n, n)
    lift = np.stack((harmonic_lift(boundary[0]), harmonic_lift(boundary[1])))
    state = lift.copy()
    state[:, mask] = boundary[:, mask]
    free_map = -np.ones((2, n, n), dtype=np.int64)
    free_nodes = np.argwhere(~mask)
    for node_index, (i, j) in enumerate(free_nodes):
        free_map[0, i, j] = 2 * node_index
        free_map[1, i, j] = 2 * node_index + 1

    def free_vector(value: np.ndarray) -> np.ndarray:
        result = np.empty(2 * len(free_nodes), dtype=np.float64)
        for index, (i, j) in enumerate(free_nodes):
            result[2 * index : 2 * index + 2] = value[:, i, j]
        return result

    def set_free(value: np.ndarray, vector: np.ndarray) -> np.ndarray:
        result = value.copy()
        for index, (i, j) in enumerate(free_nodes):
            result[:, i, j] = vector[2 * index : 2 * index + 2]
        return result

    states = [free_vector(state)]
    residuals: list[float] = []
    linear_iterations: list[int] = []
    started = time.perf_counter()
    converged = False
    energy = 0.0
    min_j = 1.0
    for _ in range(max_iterations):
        residual, tangent, energy, min_j = _assemble(state, mu, lam, free_map)
        norm = float(np.linalg.norm(residual))
        residuals.append(norm)
        if norm <= tolerance:
            converged = True
            break
        update = spla.spsolve(tangent, -residual)
        linear_iterations.append(1)
        current_vector = free_vector(state)
        step = 1.0
        accepted = False
        while step >= 1e-4:
            proposal = set_free(state, current_vector + step * update)
            try:
                proposal_residual, _, _, _ = _assemble(proposal, mu, lam, free_map)
            except FloatingPointError:
                step *= 0.5
                continue
            if np.linalg.norm(proposal_residual) < norm:
                state = proposal
                states.append(free_vector(state))
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    final_residual, _, energy, min_j = _assemble(state, mu, lam, free_map)
    residuals.append(float(np.linalg.norm(final_residual)))
    converged = converged or residuals[-1] <= tolerance
    return state, states, residuals, linear_iterations, time.perf_counter() - started, energy, converged


def generate_hyperelastic_rollout(
    *, resolution: int, seed: int, case_id: int = 0, load: float = 1.0,
    fidelity: int = 0, role: str = "smoke", progress_threshold: float = 0.25,
) -> SolverRollout:
    boundary_full = _boundary_displacement(resolution, load, seed)
    mask = rectangular_boundary_mask(resolution, resolution)
    boundary = np.where(mask[None], boundary_full, 0.0)
    mu, lam = _material_fields(resolution, seed)
    lift = np.stack((harmonic_lift(boundary[0]), harmonic_lift(boundary[1])))
    solution, states, residuals, linear_iterations, solve_seconds, energy, converged = solve_hyperelastic(
        boundary, mu, lam
    )
    pre, correction, progress, valid = select_progress_update(
        states, residuals, progress_threshold
    )
    free_nodes = np.argwhere(~mask)
    correction_field = np.zeros_like(solution)
    for index, (i, j) in enumerate(free_nodes):
        correction_field[:, i, j] = correction[2 * index : 2 * index + 2]
    return SolverRollout(
        trajectory=torch.from_numpy(np.stack((lift, solution)).astype(np.float32)),
        boundary_values=torch.from_numpy(boundary.astype(np.float32)),
        boundary_lift=torch.from_numpy(lift.astype(np.float32)),
        constraint_mask=torch.from_numpy(np.broadcast_to(mask, boundary.shape).copy()),
        newton_correction=torch.from_numpy(correction_field[None].astype(np.float32)),
        residual_progress=torch.tensor([[progress]], dtype=torch.float32),
        valid_correction_mask=torch.tensor([[valid]], dtype=torch.bool),
        condition_fields=torch.from_numpy(np.stack((mu, lam)).astype(np.float32)),
        ids={
            "case_id": case_id, "pair_id": case_id, "boundary_id": case_id,
            "initial_condition_id": case_id, "rollout_id": case_id,
            "target_task_id": case_id,
        },
        meta={
            "equation": "hyperelastic", "role": role, "resolution": resolution,
            "fidelity": fidelity, "field_names": ["ux", "uy"],
            "field_scales": [0.05, 0.03], "solver_converged": converged,
            "nonlinear_iterations": [len(states) - 1],
            "linear_iterations": [sum(linear_iterations)],
            "solver_timing": [solve_seconds], "assembly_time": 0.0,
            "strain_energy": energy, "load": load, "energy_joules": None,
        },
    )
