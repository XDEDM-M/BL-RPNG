"""Coupled Gray--Scott confirmation worker for the SfePy environment.

The worker uses SciPy's sparse Newton algebra on a structured P1-equivalent
grid.  It runs in ``sfepy_env`` (and records that environment) so the
orchestration and isolation protocol is identical to the Allen--Cahn worker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.data.manifest import evaluate_boundary, read_manifest
from blrpng.paths import ensure_artifact_path, resolve_in_project, write_json
from blrpng.solvers.progress import select_progress_update


def _laplace_matrix(n: int, spacing: float) -> sp.csr_matrix:
    inner = n - 2
    one = np.ones(inner)
    line = sp.diags((one[:-1], -2.0 * one, one[:-1]), (-1, 0, 1))
    return (
        sp.kron(sp.eye(inner), line) + sp.kron(line, sp.eye(inner))
    ).tocsr() / spacing**2


def _boundary_fields(xx: np.ndarray, yy: np.ndarray, case: dict) -> tuple[np.ndarray, np.ndarray]:
    trace = evaluate_boundary(
        xx,
        yy,
        case["boundary_coefficients"],
        case["boundary_frequencies"],
    )
    amplitude = max(float(case["boundary_amplitude"]), 1e-12)
    normalized = np.clip(trace / amplitude, -1.0, 1.0)
    return 0.82 + 0.10 * normalized, 0.08 + 0.04 * normalized


def _boundary_forcing(field: np.ndarray, diffusivity: float, spacing: float) -> np.ndarray:
    rhs = np.zeros_like(field[1:-1, 1:-1])
    rhs[0] += field[0, 1:-1]
    rhs[-1] += field[-1, 1:-1]
    rhs[:, 0] += field[1:-1, 0]
    rhs[:, -1] += field[1:-1, -1]
    return -diffusivity * rhs.reshape(-1) / spacing**2


def _newton_step(
    previous_u: np.ndarray,
    previous_v: np.ndarray,
    boundary_u: np.ndarray,
    boundary_v: np.ndarray,
    *,
    dt: float,
    diffusion_u: float,
    diffusion_v: float,
    feed: float,
    kill: float,
    max_iters: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float], bool]:
    n = previous_u.shape[0]
    spacing = 1.0 / (n - 1)
    laplace = _laplace_matrix(n, spacing)
    identity = sp.eye((n - 2) ** 2, format="csr")
    previous_u_inner = previous_u[1:-1, 1:-1].reshape(-1)
    previous_v_inner = previous_v[1:-1, 1:-1].reshape(-1)
    forcing_u = _boundary_forcing(boundary_u, diffusion_u, spacing)
    forcing_v = _boundary_forcing(boundary_v, diffusion_v, spacing)
    state = np.concatenate((previous_u_inner, previous_v_inner))
    states = [state.copy()]
    residuals: list[float] = []

    def residual_and_jacobian(value: np.ndarray):
        count = previous_u_inner.size
        u, v = value[:count], value[count:]
        residual_u = (
            (u - previous_u_inner) / dt
            - diffusion_u * (laplace @ u)
            + u * v**2
            - feed * (1.0 - u)
            + forcing_u
        )
        residual_v = (
            (v - previous_v_inner) / dt
            - diffusion_v * (laplace @ v)
            - u * v**2
            + (feed + kill) * v
            + forcing_v
        )
        juu = identity / dt - diffusion_u * laplace + sp.diags(v**2 + feed)
        juv = sp.diags(2.0 * u * v)
        jvu = sp.diags(-(v**2))
        jvv = (
            identity / dt
            - diffusion_v * laplace
            + sp.diags(-2.0 * u * v + feed + kill)
        )
        return np.concatenate((residual_u, residual_v)), sp.bmat(
            ((juu, juv), (jvu, jvv)), format="csr"
        )

    converged = False
    for _ in range(max_iters):
        residual, jacobian = residual_and_jacobian(state)
        residual_norm = float(np.linalg.norm(residual))
        residuals.append(residual_norm)
        if residual_norm <= tolerance:
            converged = True
            break
        update = spla.spsolve(jacobian, -residual)
        step = 1.0
        while step >= 1e-4:
            proposal = state + step * update
            proposal_residual, _ = residual_and_jacobian(proposal)
            if np.linalg.norm(proposal_residual) < residual_norm:
                state = proposal
                states.append(state.copy())
                break
            step *= 0.5
        else:
            break
    final_residual, _ = residual_and_jacobian(state)
    residuals.append(float(np.linalg.norm(final_residual)))
    converged = converged or residuals[-1] <= tolerance
    count = previous_u_inner.size
    final_u = boundary_u.copy()
    final_v = boundary_v.copy()
    final_u[1:-1, 1:-1] = state[:count].reshape(n - 2, n - 2)
    final_v[1:-1, 1:-1] = state[count:].reshape(n - 2, n - 2)
    return final_u, final_v, states, residuals, converged


def _state_to_fields(state: np.ndarray, boundary_u: np.ndarray, boundary_v: np.ndarray):
    n = boundary_u.shape[0]
    count = (n - 2) ** 2
    u, v = boundary_u.copy(), boundary_v.copy()
    u[1:-1, 1:-1] = state[:count].reshape(n - 2, n - 2)
    v[1:-1, 1:-1] = state[count:].reshape(n - 2, n - 2)
    return u, v


def _resample(field: np.ndarray, resolution: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, field.shape[0])
    target = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(target, target, indexing="ij")
    interpolator = RegularGridInterpolator((source, source), field)
    return interpolator(np.column_stack((xx.reshape(-1), yy.reshape(-1)))).reshape(
        resolution, resolution
    )


def generate(args: argparse.Namespace):
    manifest = read_manifest(resolve_in_project(args.manifest))
    cases = [case for case in manifest["cases"] if case["split"] == args.split]
    mesh_axis = np.linspace(0.0, 1.0, args.mesh_resolution)
    mesh_x, mesh_y = np.meshgrid(mesh_axis, mesh_axis, indexing="ij")
    output_axis = np.linspace(0.0, 1.0, args.output_resolution)
    output_x, output_y = np.meshgrid(output_axis, output_axis, indexing="ij")
    output_mask = rectangular_boundary_mask(args.output_resolution, args.output_resolution)
    records: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "x", "y", "trajectory", "boundary_values", "boundary_mask", "boundary_lift",
            "newton_correction", "residual_progress", "valid_correction_mask", "case_id",
            "pair_id", "boundary_id", "initial_condition_id", "rollout_id",
            "local_step_id", "fidelity", "solver_converged", "num_newton_iters",
        )
    }
    for rollout_id, case in enumerate(cases):
        boundary_u, boundary_v = _boundary_fields(mesh_x, mesh_y, case)
        phase = float(case["initial_phase"])
        bump = np.exp(-80.0 * ((mesh_x - 0.5) ** 2 + (mesh_y - 0.5) ** 2))
        previous_u = boundary_u + (1.0 - boundary_u) * (1.0 - bump)
        previous_v = boundary_v + (0.25 + 0.03 * np.cos(phase)) * bump
        previous_u[[0, -1], :] = boundary_u[[0, -1], :]
        previous_u[:, [0, -1]] = boundary_u[:, [0, -1]]
        previous_v[[0, -1], :] = boundary_v[[0, -1], :]
        previous_v[:, [0, -1]] = boundary_v[:, [0, -1]]
        output_boundary_u, output_boundary_v = _boundary_fields(output_x, output_y, case)
        boundary_values = np.stack(
            (
                np.where(output_mask, output_boundary_u, 0.0),
                np.where(output_mask, output_boundary_v, 0.0),
            )
        )
        lift = np.stack(
            (harmonic_lift(boundary_values[0]), harmonic_lift(boundary_values[1]))
        )
        for step in range(args.steps):
            final_u, final_v, states, residuals, converged = _newton_step(
                previous_u,
                previous_v,
                boundary_u,
                boundary_v,
                dt=args.dt,
                diffusion_u=args.diffusion_u,
                diffusion_v=args.diffusion_v,
                feed=args.feed,
                kill=args.kill,
                max_iters=args.max_newton_iters,
                tolerance=args.newton_tolerance,
            )
            pre, correction, progress, valid = select_progress_update(
                states, residuals, args.progress_threshold
            )
            pre_u, pre_v = _state_to_fields(pre, boundary_u, boundary_v)
            post_u, post_v = _state_to_fields(pre + correction, boundary_u, boundary_v)
            correction_output = np.stack(
                (_resample(post_u - pre_u, args.output_resolution),
                 _resample(post_v - pre_v, args.output_resolution))
            )
            correction_output[:, output_mask] = 0.0
            previous_output = np.stack(
                (_resample(previous_u, args.output_resolution),
                 _resample(previous_v, args.output_resolution))
            )
            final_output = np.stack(
                (_resample(final_u, args.output_resolution),
                 _resample(final_v, args.output_resolution))
            )
            previous_output[:, output_mask] = boundary_values[:, output_mask]
            final_output[:, output_mask] = boundary_values[:, output_mask]
            records["x"].append(
                np.concatenate(
                    (previous_output, output_x[None], output_y[None]), axis=0
                ).astype(np.float32)
            )
            for key in ("y", "trajectory"):
                records[key].append(final_output.astype(np.float32))
            records["boundary_values"].append(boundary_values.astype(np.float32))
            records["boundary_mask"].append(
                np.broadcast_to(output_mask, (2, *output_mask.shape)).copy()
            )
            records["boundary_lift"].append(lift.astype(np.float32))
            records["newton_correction"].append(correction_output.astype(np.float32))
            records["residual_progress"].append(np.asarray([progress], np.float32))
            records["valid_correction_mask"].append(np.asarray([valid], bool))
            for key in ("case_id", "pair_id", "boundary_id", "initial_condition_id"):
                records[key].append(np.asarray(case[key], np.int64))
            records["rollout_id"].append(np.asarray(rollout_id, np.int64))
            records["local_step_id"].append(np.asarray(step, np.int64))
            records["fidelity"].append(np.asarray(args.fidelity, np.int8))
            records["solver_converged"].append(np.asarray(converged, bool))
            records["num_newton_iters"].append(np.asarray(len(states) - 1, np.int16))
            previous_u, previous_v = final_u, final_v
    arrays = {key: np.stack(values) for key, values in records.items()}
    meta = {
        "schema_version": 3,
        "equation": "gray_scott",
        "split": args.split,
        "manifest_sha256": manifest["sha256"],
        "output_resolution": args.output_resolution,
        "mesh_resolution": args.mesh_resolution,
        "steps": args.steps,
        "num_rollouts": len(cases),
        "num_samples": int(arrays["x"].shape[0]),
        "all_converged": bool(np.all(arrays["solver_converged"])),
        "worker_python": sys.executable,
        "newton_progress_threshold": args.progress_threshold,
        "correction_semantics": "cumulative_from_initial_newton_state",
    }
    return arrays, meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--output-resolution", type=int, required=True)
    parser.add_argument("--mesh-resolution", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--fidelity", type=int, required=True)
    parser.add_argument("--progress-threshold", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--diffusion-u", type=float, default=2e-5)
    parser.add_argument("--diffusion-v", type=float, default=1e-5)
    parser.add_argument("--feed", type=float, default=0.04)
    parser.add_argument("--kill", type=float, default=0.06)
    parser.add_argument("--max-newton-iters", type=int, default=15)
    parser.add_argument("--newton-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    arrays, meta = generate(args)
    output_npz = ensure_artifact_path(args.output_npz)
    np.savez_compressed(output_npz, **arrays)
    write_json(args.output_meta, meta)


if __name__ == "__main__":
    main()
