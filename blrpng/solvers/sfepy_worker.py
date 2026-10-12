"""SfePy-only Allen--Cahn trajectory generator.

This module is launched with ``sfepy_env`` by the parent orchestrator.  It
intentionally writes NumPy/JSON only; PyTorch serialization stays in the
training environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sfepy import data_dir
from sfepy.base.base import IndexedStruct
from sfepy.discrete import Equation, Equations, FieldVariable, Function, Integral, Material, Problem
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term

from blrpng.data.lifting import harmonic_lift, rectangular_boundary_mask
from blrpng.data.blind_boundaries import evaluate_case_boundary
from blrpng.data.manifest import read_manifest
from blrpng.paths import ensure_artifact_path, resolve_in_project, write_json
from blrpng.solvers.progress import select_progress_update


def _build_problem(mesh_resolution: int, dt: float, epsilon: float, gamma: float):
    mesh = gen_block_mesh(
        [1.0, 1.0],
        [mesh_resolution, mesh_resolution],
        [0.5, 0.5],
        name=f"unit_square_{mesh_resolution}",
        verbose=False,
    )
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")
    boundary = domain.create_region("Boundary", "vertices of surface", "facet")
    field = Field.from_args("field", np.float64, "scalar", omega, approx_order=1)

    u = FieldVariable("u", "unknown", field)
    test = FieldVariable("test", "test", field, primary_var_name="u")
    u_prev = FieldVariable("u_prev", "parameter", field, primary_var_name="(set-to-None)")

    boundary_state: dict[str, Any] = {}

    def boundary_fun(ts, coors, bc=None, problem=None, **kwargs):
        del ts, bc, problem, kwargs
        case = boundary_state["case"]
        return evaluate_case_boundary(coors[:, 0], coors[:, 1], case)

    bc_fun = Function("boundary_fun", boundary_fun)
    essential = EssentialBC("dirichlet", boundary, {"u.0": bc_fun})
    integral = Integral("integral", order=3)

    def reaction(values):
        return gamma * (values**3 - values)

    def d_reaction(values):
        return gamma * (3.0 * values**2 - 1.0)

    inv_dt = Material("inv_dt", val=1.0 / float(dt))
    diffusion = Material("diffusion", val=float(epsilon) ** 2)
    term = (
        Term.new("dw_dot(inv_dt.val, test, u)", integral, omega, inv_dt=inv_dt, test=test, u=u)
        - Term.new(
            "dw_dot(inv_dt.val, test, u_prev)",
            integral,
            omega,
            inv_dt=inv_dt,
            test=test,
            u_prev=u_prev,
        )
        + Term.new(
            "dw_laplace(diffusion.val, test, u)",
            integral,
            omega,
            diffusion=diffusion,
            test=test,
            u=u,
        )
        + Term.new(
            "dw_volume_nvf(reaction, d_reaction, test, u)",
            integral,
            omega,
            reaction=reaction,
            d_reaction=d_reaction,
            test=test,
            u=u,
        )
    )
    equations = Equations([Equation("allen_cahn", term)])
    problem = Problem("allen_cahn", equations=equations)
    problem.set_bcs(ebcs=Conditions([essential]))
    return problem, u, boundary_state


def _set_full_state(problem: Problem, values: np.ndarray) -> FieldVariable:
    variables = problem.equations.variables
    variables.init_state(vec=np.asarray(values, dtype=np.float64).copy())
    return variables["u"]


def _set_reduced_state(problem: Problem, reduced: np.ndarray) -> FieldVariable:
    variables = problem.equations.variables
    variables.set_state(np.asarray(reduced, dtype=np.float64), problem.active_only, apply_ebc=True)
    return variables["u"]


def _interpolate(variable: FieldVariable, points: np.ndarray) -> np.ndarray:
    evaluated = variable.evaluate_at(
        points,
        ret_status=True,
        close_limit=1e-8,
        verbose=False,
    )
    if len(evaluated) == 3:
        values, _cells, status = evaluated
    else:
        values, status = evaluated
    values = np.asarray(values).reshape(-1)
    bad = np.asarray(status).reshape(-1) > 0
    if np.any(bad) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"SfePy interpolation failed for {int(bad.sum())} points")
    return values


def _initial_field(x: np.ndarray, y: np.ndarray, case: dict[str, Any]) -> np.ndarray:
    coeffs = np.asarray(case["initial_coefficients"], dtype=np.float64)
    phase = float(case.get("initial_phase", 0.0))
    interior = np.zeros_like(x, dtype=np.float64)
    modes = ((1, 1), (1, 2), (2, 1), (2, 2))
    for coefficient, (kx, ky) in zip(coeffs, modes):
        interior += coefficient * np.sin(np.pi * kx * x + phase) * np.sin(
            np.pi * ky * y - 0.5 * phase
        )
    scale = np.max(np.abs(interior))
    if scale > 0:
        interior *= 0.55 / scale
    boundary = evaluate_case_boundary(x, y, case)
    # A smooth transfinite-like blend gives a useful initial state while exact
    # Dirichlet values are imposed explicitly by SfePy below.
    edge_weight = np.maximum.reduce((1.0 - x, x, 1.0 - y, y))
    return interior + edge_weight**8 * boundary


def generate(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = read_manifest(resolve_in_project(args.manifest))
    cases = [case for case in manifest["cases"] if case["split"] == args.split]
    if args.case_start < 0 or args.case_start > len(cases):
        raise ValueError(
            f"case-start={args.case_start} is outside split size {len(cases)}"
        )
    cases = cases[args.case_start :]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError(f"manifest has no cases for split={args.split!r}")

    assembly_started = time.perf_counter()
    problem, u_var, boundary_state = _build_problem(
        args.mesh_resolution,
        args.dt,
        args.epsilon,
        args.gamma,
    )
    assembly_seconds = time.perf_counter() - assembly_started
    grid_axis = np.linspace(0.0, 1.0, args.output_resolution)
    xx, yy = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid_points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    bmask = rectangular_boundary_mask(args.output_resolution, args.output_resolution)

    records: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "x",
            "y",
            "trajectory",
            "boundary_values",
            "boundary_mask",
            "boundary_lift",
            "newton_correction",
            "newton_correction_p075",
            "residual_progress",
            "residual_progress_p075",
            "valid_correction_mask",
            "valid_correction_mask_p075",
            "case_id",
            "pair_id",
            "boundary_id",
            "initial_condition_id",
            "target_task_id",
            "rollout_id",
            "local_step_id",
            "fidelity",
            "solver_converged",
            "num_newton_iters",
            "solver_seconds",
            "assembly_seconds",
        )
    }
    convergence_failures: list[str] = []

    for local_rollout_idx, case in enumerate(cases):
        rollout_idx = args.case_start + local_rollout_idx
        boundary_state["case"] = case
        problem.time_update()
        vertex_coordinates = u_var.field.get_coor()
        initial = _initial_field(
            vertex_coordinates[:, 0],
            vertex_coordinates[:, 1],
            case,
        )
        prev_full = np.asarray(initial, dtype=np.float64).copy()

        boundary_grid = evaluate_case_boundary(xx, yy, case)
        boundary_values = np.where(bmask, boundary_grid, 0.0)
        lift = harmonic_lift(boundary_values)

        for local_step in range(args.steps):
            # SfePy may mutate state0/var_data arrays in-place. Sample the
            # physical input before solve() and pass independent copies so x_t
            # can never be overwritten by u_{t+1}.
            previous_variable = _set_full_state(problem, prev_full.copy())
            previous_grid = _interpolate(previous_variable, grid_points).reshape(xx.shape)
            iter_states: list[np.ndarray] = []
            iter_residuals: list[float] = []

            def iter_hook(context, solver, vec_x, it, err, err0):
                del context, it, err, err0
                vec = np.asarray(vec_x, dtype=np.float64).copy()
                iter_states.append(vec)
                residual = float(np.linalg.norm(np.asarray(solver.fun(vec)), ord=2))
                iter_residuals.append(residual)

            status: dict[str, Any] = {}
            nonlinear = Newton(
                {
                    "i_max": args.max_newton_iters,
                    "eps_a": args.newton_atol,
                    "eps_r": args.newton_rtol,
                    "ls_on": 10.0,
                    "report_status": True,
                },
                lin_solver=ScipyDirect({}),
                status=status,
            )
            problem.nls_iter_hook = iter_hook
            problem.set_solver(nonlinear)
            solve_started = time.perf_counter()
            solved = problem.solve(
                state0=prev_full.copy(),
                var_data={"u_prev": prev_full.copy()},
                save_results=False,
                verbose=False,
            )
            solve_seconds = time.perf_counter() - solve_started
            final_reduced = np.asarray(
                solved.get_state(problem.active_only, force=True), dtype=np.float64
            ).copy()
            if not iter_states or np.linalg.norm(iter_states[-1] - final_reduced) > 1e-13:
                iter_states.append(final_reduced.copy())
                iter_residuals.append(
                    float(np.linalg.norm(np.asarray(problem.get_nls().fun(final_reduced)), ord=2))
                )

            final_residual = iter_residuals[-1]
            residual_threshold = max(
                args.newton_atol,
                args.newton_rtol * max(iter_residuals[0] if iter_residuals else 1.0, 1.0),
            )
            converged = bool(
                np.isfinite(final_residual)
                and final_residual <= residual_threshold
                and int(status.get("condition", 1)) == 0
            )
            if not converged:
                convergence_failures.append(f"{case['case_id']}:{local_step}")

            pre_reduced, correction_reduced, progress, valid = select_progress_update(
                iter_states,
                iter_residuals,
                args.progress_threshold,
            )
            pre_reduced_p075, correction_reduced_p075, progress_p075, valid_p075 = (
                select_progress_update(iter_states, iter_residuals, 0.75)
            )
            pre_variable = _set_reduced_state(problem, pre_reduced)
            pre_grid = _interpolate(pre_variable, grid_points).reshape(xx.shape)
            post_variable = _set_reduced_state(problem, pre_reduced + correction_reduced)
            post_grid = _interpolate(post_variable, grid_points).reshape(xx.shape)
            correction_grid = post_grid - pre_grid
            pre_variable_p075 = _set_reduced_state(problem, pre_reduced_p075)
            pre_grid_p075 = _interpolate(pre_variable_p075, grid_points).reshape(xx.shape)
            post_variable_p075 = _set_reduced_state(
                problem, pre_reduced_p075 + correction_reduced_p075
            )
            correction_grid_p075 = (
                _interpolate(post_variable_p075, grid_points).reshape(xx.shape)
                - pre_grid_p075
            )
            final_variable = _set_reduced_state(problem, final_reduced)
            final_grid = _interpolate(final_variable, grid_points).reshape(xx.shape)
            correction_grid[bmask] = 0.0
            correction_grid_p075[bmask] = 0.0
            final_grid[bmask] = boundary_grid[bmask]
            previous_grid[bmask] = boundary_grid[bmask]

            records["x"].append(
                np.stack((previous_grid, xx, yy), axis=0).astype(np.float32)
            )
            records["y"].append(final_grid[None].astype(np.float32))
            records["trajectory"].append(final_grid[None].astype(np.float32))
            records["boundary_values"].append(boundary_values[None].astype(np.float32))
            records["boundary_mask"].append(bmask[None])
            records["boundary_lift"].append(lift[None].astype(np.float32))
            records["newton_correction"].append(correction_grid[None].astype(np.float32))
            records["newton_correction_p075"].append(
                correction_grid_p075[None].astype(np.float32)
            )
            records["residual_progress"].append(np.asarray([progress], dtype=np.float32))
            records["residual_progress_p075"].append(
                np.asarray([progress_p075], dtype=np.float32)
            )
            records["valid_correction_mask"].append(np.asarray([valid], dtype=bool))
            records["valid_correction_mask_p075"].append(
                np.asarray([valid_p075], dtype=bool)
            )
            records["case_id"].append(np.asarray(case["case_id"]))
            records["pair_id"].append(np.asarray(case["pair_id"]))
            records["boundary_id"].append(np.asarray(case["boundary_id"]))
            records["initial_condition_id"].append(np.asarray(case["initial_condition_id"]))
            records["target_task_id"].append(
                np.asarray(case.get("target_task_id", case["boundary_id"]), dtype=np.int64)
            )
            records["rollout_id"].append(np.asarray(rollout_idx, dtype=np.int64))
            records["local_step_id"].append(np.asarray(local_step, dtype=np.int64))
            records["fidelity"].append(np.asarray(args.fidelity, dtype=np.int8))
            records["solver_converged"].append(np.asarray(converged, dtype=bool))
            records["num_newton_iters"].append(
                np.asarray(max(len(iter_states) - 1, 0), dtype=np.int16)
            )
            records["solver_seconds"].append(np.asarray(solve_seconds, dtype=np.float64))
            records["assembly_seconds"].append(
                np.asarray(assembly_seconds / max(len(cases) * args.steps, 1), dtype=np.float64)
            )
            # Reconstruct the complete nodal state for the next implicit step.
            final_variable = _set_reduced_state(problem, final_reduced)
            prev_full = np.asarray(final_variable(), dtype=np.float64).reshape(-1).copy()

    arrays = {key: np.stack(value) for key, value in records.items()}
    meta = {
        "schema_version": 3,
        "equation": "allen_cahn",
        "split": args.split,
        "manifest_sha256": manifest["sha256"],
        "output_resolution": args.output_resolution,
        "mesh_resolution": args.mesh_resolution,
        "steps": args.steps,
        "num_rollouts": len(cases),
        "num_samples": int(arrays["x"].shape[0]),
        "case_start": args.case_start,
        "dt": args.dt,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "progress_threshold": args.progress_threshold,
        "correction_semantics": "cumulative_from_initial_newton_state",
        "convergence_failures": convergence_failures,
        "all_converged": not convergence_failures,
        "sfepy_data_dir": data_dir,
        "boundary_families": sorted(
            {str(case.get("boundary_family", "legacy_fourier")) for case in cases}
        ),
    }
    return arrays, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--output-resolution", type=int, required=True)
    parser.add_argument("--mesh-resolution", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--fidelity", type=int, choices=(0, 1), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--progress-threshold", type=float, default=0.25)
    parser.add_argument("--max-newton-iters", type=int, default=20)
    parser.add_argument("--newton-atol", type=float, default=1e-10)
    parser.add_argument("--newton-rtol", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_npz = ensure_artifact_path(args.output_npz)
    output_meta = ensure_artifact_path(args.output_meta)
    arrays, meta = generate(args)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    write_json(output_meta, meta)


if __name__ == "__main__":
    main()
