"""SfePy-only transient incompressible Navier--Stokes rollout worker.

The worker is launched with ``/opt/miniconda3/envs/sfepy_env/bin/python``.
It intentionally emits NumPy/JSON only; PyTorch serialization is performed by
the parent process in the ``fnotest`` environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, spilu
from sfepy.base.base import output
from sfepy.discrete import (
    Equation,
    Equations,
    FieldVariable,
    Function,
    Integral,
    Material,
    Problem,
)
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete.integrals import Integrals
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
from sfepy.terms import Term

from blrpng.paths import ensure_artifact_path, resolve_in_project, write_json
from blrpng.solvers.progress import select_progress_update


def lid_profile(x: np.ndarray, boundary: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    envelope = 16.0 * x**2 * (1.0 - x) ** 2
    modulation = np.ones_like(x)
    for coefficient, frequency, phase in zip(
        boundary["fourier_coefficients"],
        boundary["fourier_frequencies"],
        boundary["fourier_phases"],
    ):
        modulation += float(coefficient) * np.sin(
            2.0 * np.pi * float(frequency) * x + float(phase)
        )
    sigma = max(float(boundary.get("bump_sigma", 0.15)), 1e-6)
    modulation += float(boundary.get("bump_amplitude", 0.0)) * np.exp(
        -((x - float(boundary.get("bump_center", 0.5))) / sigma) ** 2
    )
    return float(boundary["amplitude"]) * envelope * modulation


def forcing_field(
    coordinates: np.ndarray,
    forcing: dict[str, Any],
    time_value: float = 0.0,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not bool(forcing.get("active", False)):
        return np.zeros((coordinates.shape[0], 2), dtype=np.float64)
    x, y = coordinates[:, 0], coordinates[:, 1]
    amplitude = float(forcing["amplitude"])
    if str(forcing.get("temporal_mode", "static")) == "sinusoidal":
        amplitude *= float(forcing.get("temporal_bias", 1.0)) + float(
            forcing.get("temporal_modulation", 0.0)
        ) * np.sin(
            float(forcing["temporal_omega"]) * float(time_value)
            + float(forcing.get("temporal_phase", 0.0))
        )
    kx, ky = int(forcing["kx"]), int(forcing["ky"])
    phase = float(forcing["phase"])
    phase += float(forcing.get("temporal_spatial_phase_amplitude", 0.0)) * np.sin(
        float(forcing.get("temporal_spatial_phase_omega", 0.0)) * float(time_value)
        + float(forcing.get("temporal_spatial_phase_offset", 0.0))
    )
    sx = np.sin(np.pi * kx * x + phase)
    cx = np.cos(np.pi * kx * x + phase)
    sy = np.sin(np.pi * ky * y)
    cy = np.cos(np.pi * ky * y)
    # f = curl(psi) is divergence free by construction.
    return np.column_stack(
        (
            amplitude * np.pi * ky * sx * cy,
            -amplitude * np.pi * kx * cx * sy,
        )
    )


def interior_velocity(
    coordinates: np.ndarray,
    seed: int,
    amplitude: float,
    relative_scale: float = 0.08,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    x, y = coordinates[:, 0], coordinates[:, 1]
    ex, ey = x**2 * (1.0 - x) ** 2, y**2 * (1.0 - y) ** 2
    dex = 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x)
    dey = 2.0 * y * (1.0 - y) * (1.0 - 2.0 * y)
    rng = np.random.default_rng(int(seed))
    stream = np.zeros_like(x)
    stream_x = np.zeros_like(x)
    stream_y = np.zeros_like(x)
    for kx, ky in ((1, 1), (1, 2), (2, 1), (2, 2)):
        coefficient = float(rng.normal())
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        sx = np.sin(np.pi * kx * x + phase)
        cx = np.cos(np.pi * kx * x + phase)
        sy = np.sin(np.pi * ky * y - 0.5 * phase)
        cy = np.cos(np.pi * ky * y - 0.5 * phase)
        stream += coefficient * sx * sy
        stream_x += coefficient * np.pi * kx * cx * sy
        stream_y += coefficient * np.pi * ky * sx * cy
    u = ex * (dey * stream + ey * stream_y)
    v = -ey * (dex * stream + ex * stream_x)
    velocity = np.column_stack((u, v))
    maximum = float(np.max(np.linalg.norm(velocity, axis=1)))
    if maximum > 0.0:
        velocity *= float(relative_scale) * float(amplitude) / maximum
    return velocity


def _regions(domain: FEDomain) -> dict[str, Any]:
    regions = {"Omega": domain.create_region("Omega", "all")}
    regions["Left"] = domain.create_region("Left", "vertices in (x < 1e-12)", "facet")
    regions["Right"] = domain.create_region("Right", "vertices in (x > 0.999999)", "facet")
    regions["Bottom"] = domain.create_region("Bottom", "vertices in (y < 1e-12)", "facet")
    regions["Top"] = domain.create_region("Top", "vertices in (y > 0.999999)", "facet")
    regions["Walls"] = domain.create_region(
        "Walls", "r.Left +v r.Right +v r.Bottom", "facet"
    )
    regions["PressureAnchor"] = domain.create_region(
        "PressureAnchor", "vertex 0", "vertex"
    )
    return regions


def _linear_solver(kind: str):
    if kind == "direct":
        return ScipyDirect({"use_presolve": True})
    if kind != "gmres_ilu":
        raise ValueError(f"Unknown linear solver: {kind}")

    def setup_preconditioner(matrix, context):
        del context
        ilu = spilu(matrix.tocsc(), drop_tol=1e-4, fill_factor=12.0)
        return LinearOperator(matrix.shape, matvec=ilu.solve, dtype=matrix.dtype)

    return ScipyIterative(
        {
            "method": "gmres",
            "i_max": 500,
            "eps_a": 1e-11,
            "eps_r": 1e-9,
            "restart": 80,
            "setup_precond": setup_preconditioner,
        }
    )


def _build_problem(
    mesh_resolution: int,
    *,
    boundary: dict[str, Any],
    viscosity: float,
    dt: float | None,
    include_convection: bool,
    include_forcing: bool,
    linear_solver: str,
    mass_current_coefficient: float = 1.0,
    grad_div_gamma: float = 0.0,
) -> tuple[Problem, Field, Field]:
    if not 0.0 <= grad_div_gamma <= 1000.0:
        raise ValueError("grad_div_gamma must be within [0, 1000]")
    mesh = gen_block_mesh(
        [1.0, 1.0],
        [int(mesh_resolution), int(mesh_resolution)],
        [0.5, 0.5],
        name=f"ns_unit_square_{mesh_resolution}",
        verbose=False,
    )
    domain = FEDomain("domain", mesh)
    regions = _regions(domain)
    velocity_field = Field.from_args(
        "velocity", np.float64, "vector", regions["Omega"], approx_order=2
    )
    pressure_field = Field.from_args(
        "pressure", np.float64, "scalar", regions["Omega"], approx_order=1
    )
    u = FieldVariable("u", "unknown", velocity_field, order=0)
    v = FieldVariable("v", "test", velocity_field, primary_var_name="u")
    p = FieldVariable("p", "unknown", pressure_field, order=1)
    q = FieldVariable("q", "test", pressure_field, primary_var_name="p")
    integral = Integral("integral", order=4)
    viscous = Material("viscous", val=float(viscosity))
    term = (
        Term.new(
            "dw_div_grad(viscous.val, v, u)",
            integral,
            regions["Omega"],
            viscous=viscous,
            v=v,
            u=u,
        )
        - Term.new("dw_stokes(v, p)", integral, regions["Omega"], v=v, p=p)
        + Term.new("dw_stokes(u, q)", integral, regions["Omega"], u=u, q=q)
    )
    if grad_div_gamma > 0.0:
        grad_div = Material("grad_div", val=float(grad_div_gamma))
        term = term + Term.new(
            "dw_st_grad_div(grad_div.val, v, u)",
            integral,
            regions["Omega"],
            grad_div=grad_div,
            v=v,
            u=u,
        )
    if dt is not None:
        previous = FieldVariable(
            "u_history", "parameter", velocity_field, primary_var_name="(set-to-None)"
        )
        current_mass = Material(
            "current_mass", val=float(mass_current_coefficient) / float(dt)
        )
        history_mass = Material("history_mass", val=1.0 / float(dt))
        term = (
            Term.new(
                "dw_dot(current_mass.val, v, u)",
                integral,
                regions["Omega"],
                current_mass=current_mass,
                v=v,
                u=u,
            )
            - Term.new(
                "dw_dot(history_mass.val, v, u_history)",
                integral,
                regions["Omega"],
                history_mass=history_mass,
                v=v,
                u_history=previous,
            )
            + term
        )
    if include_convection:
        term = term + Term.new(
            "dw_convect(v, u)", integral, regions["Omega"], v=v, u=u
        )
    if include_forcing:
        force = FieldVariable(
            "force", "parameter", velocity_field, primary_var_name="(set-to-None)"
        )
        term = term - Term.new(
            "dw_dot(v, force)", integral, regions["Omega"], v=v, force=force
        )
    equations = Equations([Equation("incompressible_navier_stokes", term)])
    problem = Problem("incompressible_navier_stokes", equations=equations)

    def top_velocity(ts, coordinates, bc=None, problem=None, **kwargs):
        del ts, bc, problem, kwargs
        return lid_profile(coordinates[:, 0], boundary)

    conditions = Conditions(
        [
            EssentialBC("stationary_walls", regions["Walls"], {"u.all": 0.0}),
            EssentialBC(
                "moving_lid",
                regions["Top"],
                {"u.0": Function("top_velocity", top_velocity), "u.1": 0.0},
            ),
            EssentialBC(
                "pressure_gauge", regions["PressureAnchor"], {"p.0": 0.0}
            ),
        ]
    )
    problem.set_bcs(ebcs=conditions)
    problem.time_update()
    problem.set_solver(
        Newton(
            {
                "i_max": 1 if not include_convection else 20,
                "eps_a": 1e-10,
                "eps_r": 1e-8,
                "ls_on": 0.99999,
                "ls_red": 0.1,
                "ls_min": 1e-6,
                "report_status": True,
            },
            lin_solver=_linear_solver(linear_solver),
        )
    )
    return problem, velocity_field, pressure_field


def _mixed_state(problem: Problem, velocity: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    variables = problem.equations.variables
    if variables.vec is None:
        variables.init_state()
    variables.set_state_parts(
        {
            "u": np.asarray(velocity, dtype=np.float64).reshape(-1),
            "p": np.asarray(pressure, dtype=np.float64).reshape(-1),
        }
    )
    return np.asarray(variables.get_state(), dtype=np.float64).copy()


def _set_reduced_state(problem: Problem, reduced: np.ndarray):
    variables = problem.equations.variables
    variables.set_state(
        np.asarray(reduced, dtype=np.float64), problem.active_only, apply_ebc=True
    )
    return variables["u"], variables["p"]


def _evaluate(variable: FieldVariable, points: np.ndarray) -> np.ndarray:
    result = variable.evaluate_at(points, ret_status=True, close_limit=1e-8, verbose=False)
    if len(result) == 3:
        values, _cells, status = result
    else:
        values, status = result
    status = np.asarray(status).reshape(-1)
    if np.any(status > 0) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"SfePy interpolation failed at {int(np.sum(status > 0))} points")
    return np.asarray(values)


def _grid_fields(
    problem: Problem,
    reduced: np.ndarray,
    points: np.ndarray,
    shape: tuple[int, int],
    boundary: dict[str, Any],
) -> np.ndarray:
    velocity, pressure = _set_reduced_state(problem, reduced)
    velocity_values = _evaluate(velocity, points).reshape(*shape, 2)
    pressure_values = _evaluate(pressure, points).reshape(*shape)
    pressure_values -= float(np.mean(pressure_values))
    fields = np.stack(
        (velocity_values[..., 0], velocity_values[..., 1], pressure_values)
    )
    lid = lid_profile(np.linspace(0.0, 1.0, shape[0]), boundary)
    fields[0, 0, :] = 0.0
    fields[0, -1, :] = 0.0
    fields[0, :, 0] = 0.0
    fields[0, :, -1] = lid
    fields[1, 0, :] = 0.0
    fields[1, -1, :] = 0.0
    fields[1, :, 0] = 0.0
    fields[1, :, -1] = 0.0
    return fields


def _diagnostics(fields: np.ndarray) -> tuple[float, float, float, float]:
    resolution = fields.shape[-1]
    spacing = 1.0 / float(resolution - 1)
    u, v = fields[0], fields[1]
    dudx = np.gradient(u, spacing, axis=0, edge_order=2)
    dvdy = np.gradient(v, spacing, axis=1, edge_order=2)
    divergence = dudx + dvdy
    divergence_rms = float(np.sqrt(np.mean(divergence[1:-1, 1:-1] ** 2)))
    vorticity = np.gradient(v, spacing, axis=0, edge_order=2) - np.gradient(
        u, spacing, axis=1, edge_order=2
    )
    energy = float(0.5 * np.mean(u**2 + v**2))
    enstrophy = float(0.5 * np.mean(vorticity**2))
    net_flux = float(
        spacing
        * (
            np.trapezoid(u[-1, :])
            - np.trapezoid(u[0, :])
            + np.trapezoid(v[:, -1])
            - np.trapezoid(v[:, 0])
        )
    )
    return divergence_rms, net_flux, energy, enstrophy


def accepted_path_information(
    accepted_fields: list[np.ndarray], accepted_residuals: list[float],
    field_slice: slice = slice(0, 2),
) -> dict[str, Any]:
    """Measure Newton-path information relative to its matched endpoint.

    Quantities use ``field_slice`` and compare every accepted state with the
    endpoint reached from the *same* natural predictor.  Pressure is already
    projected to zero mean by ``_grid_fields``.  The initial and converged
    fields must be the first and last entries, respectively.
    """
    if len(accepted_fields) != len(accepted_residuals) or len(accepted_fields) < 2:
        raise ValueError("An accepted Newton path needs aligned initial/final states.")
    initial = np.asarray(accepted_fields[0][field_slice], dtype=np.float64).reshape(-1)
    endpoint = np.asarray(accepted_fields[-1][field_slice], dtype=np.float64).reshape(-1) - initial
    endpoint_norm = max(float(np.linalg.norm(endpoint)), 1e-30)
    residual0 = max(float(accepted_residuals[0]), 1e-30)
    residual_final = max(float(accepted_residuals[-1]), 1e-30)
    log_span = max(math.log(residual0) - math.log(residual_final), 1e-12)
    updates = []
    increments = []
    previous = initial
    for index, (field, residual) in enumerate(
        zip(accepted_fields[1:], accepted_residuals[1:]), start=1
    ):
        current = np.asarray(field[field_slice], dtype=np.float64).reshape(-1)
        cumulative = current - initial
        increment = current - previous
        increments.append(increment)
        norm = float(np.linalg.norm(cumulative))
        dot = float(np.dot(cumulative, endpoint))
        cosine = dot / max(norm * endpoint_norm, 1e-30)
        coefficient = dot / (endpoint_norm**2)
        orthogonal = cumulative - coefficient * endpoint
        updates.append(
            {
                "accepted_index": index,
                "residual_progress": float(np.clip(
                    (math.log(residual0) - math.log(max(float(residual), 1e-30)))
                    / log_span,
                    0.0,
                    1.0,
                )),
                "endpoint_discrepancy": float(np.linalg.norm(cumulative - endpoint))
                / endpoint_norm,
                "norm_ratio": norm / endpoint_norm,
                "endpoint_cosine": float(np.clip(cosine, -1.0, 1.0)),
                "orthogonal_fraction": float(np.linalg.norm(orthogonal)) / endpoint_norm,
            }
        )
        previous = current
    turn_cosines = []
    for left, right in zip(increments, increments[1:]):
        cosine = float(np.dot(left, right)) / max(
            float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-30
        )
        turn_cosines.append(float(np.clip(cosine, -1.0, 1.0)))
    path_length = sum(float(np.linalg.norm(value)) for value in increments)
    endpoint_state_norm = max(
        float(np.linalg.norm(np.asarray(accepted_fields[-1][field_slice], dtype=np.float64))),
        1e-30,
    )
    nonfinal = updates[:-1] if len(updates) > 1 else updates
    return {
        "updates": updates,
        "turn_cosines": turn_cosines,
        "path_length_excess": path_length / endpoint_norm - 1.0,
        "maximum_orthogonal_fraction": max(
            (value["orthogonal_fraction"] for value in nonfinal), default=0.0
        ),
        "minimum_nonfinal_endpoint_cosine": min(
            (value["endpoint_cosine"] for value in nonfinal), default=1.0
        ),
        "endpoint_velocity_fraction_of_final_state": endpoint_norm / endpoint_state_norm,
    }


def _solve_stokes_lift(
    mesh_resolution: int,
    output_resolution: int,
    boundary: dict[str, Any],
    linear_solver: str,
    grad_div_gamma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    problem, velocity_field, pressure_field = _build_problem(
        mesh_resolution,
        boundary=boundary,
        viscosity=1.0,
        dt=None,
        include_convection=False,
        include_forcing=False,
        linear_solver=linear_solver,
        grad_div_gamma=grad_div_gamma,
    )
    started = time.perf_counter()
    solved = problem.solve(save_results=False, verbose=False)
    elapsed = time.perf_counter() - started
    velocity = np.asarray(solved["u"](), dtype=np.float64).reshape(-1, 2)
    pressure = np.asarray(solved["p"](), dtype=np.float64).reshape(-1)
    reduced = np.asarray(solved.get_state(problem.active_only, force=True), dtype=np.float64)
    axis = np.linspace(0.0, 1.0, output_resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    fields = _grid_fields(problem, reduced, points, xx.shape, boundary)
    lift = np.stack((fields[0], fields[1], np.zeros_like(fields[2])))
    if velocity.shape[0] != velocity_field.n_nod or pressure.shape[0] != pressure_field.n_nod:
        raise RuntimeError("Unexpected Taylor--Hood state sizes in Stokes lift.")
    return velocity, pressure, lift, elapsed


def generate_rollout(
    case: dict[str, Any],
    *,
    output_resolution: int,
    mesh_resolution: int,
    steps: int,
    dt: float,
    progress_threshold: float,
    path_selection_mode: str = "progress",
    linear_solver: str,
    max_newton_iterations: int,
    newton_atol: float,
    newton_rtol: float,
    time_integrator: str = "BDF1",
    progress_path: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if time_integrator == "STEADY_CONTINUATION":
        return generate_steady_continuation(
            case,
            output_resolution=output_resolution,
            mesh_resolution=mesh_resolution,
            steps=steps,
            dt=dt,
            progress_threshold=progress_threshold,
            path_selection_mode=path_selection_mode,
            linear_solver=linear_solver,
            max_newton_iterations=max_newton_iterations,
            newton_atol=newton_atol,
            newton_rtol=newton_rtol,
            progress_path=progress_path,
        )
    if time_integrator not in {"BDF1", "BDF2", "BDF3"}:
        raise ValueError(f"Unsupported time integrator: {time_integrator}")
    if path_selection_mode not in {"progress", "first_accepted"}:
        raise ValueError(f"Unsupported path selection mode: {path_selection_mode}")
    boundary = case["boundary"]
    amplitude = float(boundary["amplitude"])
    reynolds = float(case["reynolds"])
    viscosity = amplitude / reynolds
    grad_div_gamma = float(case.get("grad_div_gamma", 0.0))
    predictor_blend = float(case.get("newton_predictor_blend", 1.0))
    if not 0.0 <= predictor_blend <= 1.0:
        raise ValueError("newton_predictor_blend must be within [0, 1]")
    lift_velocity, _lift_pressure, lift, lift_seconds = _solve_stokes_lift(
        mesh_resolution, output_resolution, boundary, linear_solver, grad_div_gamma
    )
    assembly_started = time.perf_counter()
    problem_bdf1, velocity_field, pressure_field = _build_problem(
        mesh_resolution,
        boundary=boundary,
        viscosity=viscosity,
        dt=dt,
        include_convection=True,
        include_forcing=True,
        linear_solver=linear_solver,
        mass_current_coefficient=1.0,
        grad_div_gamma=grad_div_gamma,
    )
    problems = [problem_bdf1]
    problem_bdf2 = None
    if time_integrator in {"BDF2", "BDF3"}:
        problem_bdf2, _velocity_field_bdf2, _pressure_field_bdf2 = _build_problem(
            mesh_resolution,
            boundary=boundary,
            viscosity=viscosity,
            dt=dt,
            include_convection=True,
            include_forcing=True,
            linear_solver=linear_solver,
            mass_current_coefficient=1.5,
            grad_div_gamma=grad_div_gamma,
        )
        problems.append(problem_bdf2)
    problem_bdf3 = None
    if time_integrator == "BDF3":
        problem_bdf3, _velocity_field_bdf3, _pressure_field_bdf3 = _build_problem(
            mesh_resolution,
            boundary=boundary,
            viscosity=viscosity,
            dt=dt,
            include_convection=True,
            include_forcing=True,
            linear_solver=linear_solver,
            mass_current_coefficient=11.0 / 6.0,
            grad_div_gamma=grad_div_gamma,
        )
        problems.append(problem_bdf3)
    for candidate in problems:
        candidate.get_nls().conf.i_max = int(max_newton_iterations)
        candidate.get_nls().conf.eps_a = float(newton_atol)
        candidate.get_nls().conf.eps_r = float(newton_rtol)
    assembly_seconds = time.perf_counter() - assembly_started

    velocity_coordinates = velocity_field.get_coor()
    pressure_coordinates = pressure_field.get_coor()
    velocity = lift_velocity + interior_velocity(
        velocity_coordinates,
        int(case["initial_seed"]),
        amplitude,
        float(case.get("initial_perturbation_scale", 0.08)),
    )
    pressure = np.zeros(pressure_coordinates.shape[0], dtype=np.float64)
    axis = np.linspace(0.0, 1.0, output_resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    temporal_forcing = str(case["forcing"].get("temporal_mode", "static")) != "static"
    condition_sequence = []
    for condition_step in range(int(steps)):
        force_grid = forcing_field(
            points, case["forcing"], (condition_step + 1) * float(dt)
        ).reshape(output_resolution, output_resolution, 2)
        condition_sequence.append(
            np.stack(
                (
                    np.full_like(xx, viscosity),
                    np.full_like(xx, dt),
                    force_grid[..., 0],
                    force_grid[..., 1],
                )
            )
        )
    condition_fields = (
        np.stack(condition_sequence)
        if temporal_forcing
        else condition_sequence[0]
    )
    boundary_values = np.zeros((3, output_resolution, output_resolution), dtype=np.float64)
    boundary_values[0, :, -1] = lid_profile(axis, boundary)
    constraint_mask = np.zeros_like(boundary_values, dtype=bool)
    constraint_mask[:2, 0, :] = True
    constraint_mask[:2, -1, :] = True
    constraint_mask[:2, :, 0] = True
    constraint_mask[:2, :, -1] = True

    trajectory: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    predictor_endpoints: list[np.ndarray] = []
    progresses: list[list[float]] = []
    valid_masks: list[list[bool]] = []
    residual_histories: list[list[float]] = []
    nonlinear_iterations: list[int] = []
    linear_iterations: list[int] = []
    solve_timings: list[float] = []
    relative_residuals: list[float] = []
    weak_continuity: list[float] = []
    divergence_rms: list[float] = []
    fe_divergence_qp_rms: list[float] = []
    net_flux: list[float] = []
    energy: list[float] = []
    enstrophy: list[float] = []
    all_converged = True

    state0 = _mixed_state(problem_bdf1, velocity, pressure)
    initial_reduced = problem_bdf1.equations.variables.get_state(
        problem_bdf1.active_only, force=True
    )
    previous_grid = _grid_fields(
        problem_bdf1, initial_reduced, points, xx.shape, boundary
    )
    trajectory.append(previous_grid.copy())
    older_velocity: np.ndarray | None = None
    oldest_velocity: np.ndarray | None = None
    first_update_discrepancies: list[float] = []
    first_update_predictor_discrepancies: list[float] = []
    selected_velocity_discrepancies: list[float] = []
    selected_pressure_discrepancies: list[float] = []
    selected_velocity_predictor_discrepancies: list[float] = []
    selected_pressure_predictor_discrepancies: list[float] = []
    second_to_first_ratios: list[float] = []
    nonlinear_remainder_ratios: list[float] = []
    accepted_path_metrics: list[dict[str, Any]] = []
    accepted_pressure_path_metrics: list[dict[str, Any]] = []
    path_information_masks: list[list[bool]] = []

    if progress_path is not None:
        write_json(
            progress_path,
            {
                "schema": "blrpng_sfepy_ns_worker_progress_v1",
                "case_id": int(case["case_id"]),
                "state": "solving",
                "completed_steps": 0,
                "requested_steps": int(steps),
                "resolution": int(output_resolution),
                "mesh_resolution": int(mesh_resolution),
                "dt": float(dt),
                "time_integrator": time_integrator,
                "pid": os.getpid(),
            },
        )

    for _step in range(int(steps)):
        if time_integrator == "BDF3" and _step > 1:
            problem = problem_bdf3
        elif time_integrator in {"BDF2", "BDF3"} and _step > 0:
            problem = problem_bdf2
        else:
            problem = problem_bdf1
        assert problem is not None
        predictor_velocity = lift_velocity + predictor_blend * (velocity - lift_velocity)
        predictor_pressure = predictor_blend * pressure
        state0 = _mixed_state(problem, predictor_velocity, predictor_pressure)
        velocity_before = velocity.copy()
        if (
            time_integrator == "BDF3"
            and _step > 1
            and older_velocity is not None
            and oldest_velocity is not None
        ):
            history_velocity = (
                3.0 * velocity
                - 1.5 * older_velocity
                + (1.0 / 3.0) * oldest_velocity
            )
        elif (
            time_integrator in {"BDF2", "BDF3"}
            and _step > 0
            and older_velocity is not None
        ):
            history_velocity = 2.0 * velocity - 0.5 * older_velocity
        else:
            history_velocity = velocity
        force_velocity = forcing_field(
            velocity_coordinates, case["forcing"], (_step + 1) * float(dt)
        )
        accepted_states: list[np.ndarray] = []
        accepted_residuals: list[float] = []

        def iter_hook(context, solver, vector, iteration, error, error0):
            del context, iteration, error, error0
            current = np.asarray(vector, dtype=np.float64).copy()
            residual = float(np.linalg.norm(np.asarray(solver.fun(current)), ord=2))
            if not accepted_states or np.linalg.norm(current - accepted_states[-1]) > 1e-14:
                accepted_states.append(current)
                accepted_residuals.append(residual)

        problem.nls_iter_hook = iter_hook
        status: dict[str, Any] = {}
        problem.get_nls().status = status
        problem.get_nls().iter_hook = iter_hook
        started = time.perf_counter()
        solved = problem.solve(
            state0=state0.copy(),
            var_data={
                "u_history": history_velocity.reshape(-1).copy(),
                "force": force_velocity.reshape(-1),
            },
            save_results=False,
            verbose=False,
            status=status,
        )
        elapsed = time.perf_counter() - started
        final_reduced = np.asarray(
            solved.get_state(problem.active_only, force=True), dtype=np.float64
        ).copy()
        final_residual_vector = np.asarray(problem.get_nls().fun(final_reduced), dtype=np.float64)
        final_residual = float(np.linalg.norm(final_residual_vector))
        if not accepted_states or np.linalg.norm(final_reduced - accepted_states[-1]) > 1e-13:
            accepted_states.append(final_reduced.copy())
            accepted_residuals.append(final_residual)
        relative_residual = final_residual / max(float(accepted_residuals[0]), 1e-30)
        converged = bool(
            int(status.get("condition", 1)) == 0
            and np.isfinite(relative_residual)
            and relative_residual <= max(float(newton_rtol), 1e-12)
        )
        all_converged = all_converged and converged

        if path_selection_mode == "first_accepted":
            valid = bool(
                len(accepted_states) >= 2
                and np.linalg.norm(accepted_states[1] - accepted_states[0]) > 1e-13
            )
            initial = accepted_states[0]
            correction = (
                accepted_states[1] - initial
                if valid else np.zeros_like(initial)
            )
            initial_residual = max(float(accepted_residuals[0]), 1e-30)
            selected_residual = max(float(accepted_residuals[1]), 1e-30) if valid else initial_residual
            final_residual_for_progress = max(float(accepted_residuals[-1]), 1e-30)
            progress = float(np.clip(
                (math.log(initial_residual) - math.log(selected_residual))
                / max(math.log(initial_residual) - math.log(final_residual_for_progress), 1e-12),
                0.0, 1.0,
            ))
        else:
            initial, correction, progress, valid = select_progress_update(
                accepted_states, accepted_residuals, float(progress_threshold)
            )
        initial_grid = _grid_fields(problem, initial, points, xx.shape, boundary)
        selected_grid = _grid_fields(problem, initial + correction, points, xx.shape, boundary)
        correction_grid = selected_grid - initial_grid
        correction_grid[:2][constraint_mask[:2]] = 0.0
        correction_grid[2] -= float(np.mean(correction_grid[2]))
        final_grid = _grid_fields(problem, final_reduced, points, xx.shape, boundary)
        accepted_grids = [
            _grid_fields(problem, state, points, xx.shape, boundary)
            for state in accepted_states
        ]
        accepted_path_metrics.append(
            accepted_path_information(accepted_grids, accepted_residuals)
        )
        accepted_pressure_path_metrics.append(
            accepted_path_information(
                accepted_grids, accepted_residuals, field_slice=slice(2, 3)
            )
        )
        # Diagnose the finite-element field before interpolation to a Cartesian
        # output grid. Taylor--Hood enforces continuity weakly, not pointwise.
        divergence_qp = np.asarray(
            problem.evaluate(
                "ev_div.integral.Omega(u)",
                mode="qp",
                integrals=Integrals([Integral("integral", order=4)]),
                u=problem.equations.variables["u"],
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(divergence_qp)):
            raise RuntimeError("Nonfinite finite-element divergence diagnostic")
        fe_divergence_qp_rms.append(float(np.sqrt(np.mean(divergence_qp**2))))
        endpoint_grid = final_grid - previous_grid
        endpoint_grid[:2][constraint_mask[:2]] = 0.0
        endpoint_grid[2] -= float(np.mean(endpoint_grid[2]))
        predictor_endpoint_grid = final_grid - initial_grid
        predictor_endpoint_grid[:2][constraint_mask[:2]] = 0.0
        predictor_endpoint_grid[2] -= float(np.mean(predictor_endpoint_grid[2]))
        pressure_correction = correction_grid[2].reshape(-1)
        pressure_endpoint = predictor_endpoint_grid[2].reshape(-1)
        pressure_endpoint_norm = max(float(np.linalg.norm(pressure_endpoint)), 1e-30)
        pressure_dot = float(np.dot(pressure_correction, pressure_endpoint))
        pressure_orthogonal = pressure_correction - (
            pressure_dot / pressure_endpoint_norm**2
        ) * pressure_endpoint
        pressure_discrepancy = (
            float(np.linalg.norm(pressure_correction - pressure_endpoint))
            / pressure_endpoint_norm
        )
        pressure_orthogonal_fraction = (
            float(np.linalg.norm(pressure_orthogonal)) / pressure_endpoint_norm
        )
        pressure_endpoint_state_fraction = pressure_endpoint_norm / max(
            float(np.linalg.norm(final_grid[2])), 1e-30
        )
        path_information_masks.append([
            bool(
                valid and converged
                and pressure_discrepancy >= 0.05
                and pressure_orthogonal_fraction >= 0.03
                and pressure_endpoint_state_fraction >= 0.02
                and progress < 0.8
            )
        ])
        for indices, destination in (
            (slice(0, 2), selected_velocity_discrepancies),
            (slice(2, 3), selected_pressure_discrepancies),
        ):
            target = endpoint_grid[indices]
            destination.append(
                float(np.linalg.norm(correction_grid[indices] - target))
                / max(float(np.linalg.norm(target)), 1e-30)
            )
        for indices, destination in (
            (slice(0, 2), selected_velocity_predictor_discrepancies),
            (slice(2, 3), selected_pressure_predictor_discrepancies),
        ):
            target = predictor_endpoint_grid[indices]
            destination.append(
                float(np.linalg.norm(correction_grid[indices] - target))
                / max(float(np.linalg.norm(target)), 1e-30)
            )

        first_grid = _grid_fields(problem, accepted_states[1], points, xx.shape, boundary)
        first_update_grid = first_grid - initial_grid
        first_update_grid[:2][constraint_mask[:2]] = 0.0
        first_update_grid[2] -= float(np.mean(first_update_grid[2]))
        field_scale = np.asarray([1.0, 1.0, 0.25], dtype=np.float64)[:, None, None]
        endpoint_norm = float(np.linalg.norm(endpoint_grid / field_scale))
        first_norm = float(np.linalg.norm(first_update_grid / field_scale))
        first_update_discrepancies.append(
            float(np.linalg.norm((first_update_grid - endpoint_grid) / field_scale))
            / max(endpoint_norm, 1e-30)
        )
        predictor_endpoint_norm = float(np.linalg.norm(predictor_endpoint_grid / field_scale))
        first_update_predictor_discrepancies.append(
            float(np.linalg.norm((first_update_grid - predictor_endpoint_grid) / field_scale))
            / max(predictor_endpoint_norm, 1e-30)
        )
        if len(accepted_states) >= 3:
            second_grid = _grid_fields(
                problem, accepted_states[2], points, xx.shape, boundary
            )
            second_update = second_grid - first_grid
            second_update[:2][constraint_mask[:2]] = 0.0
            second_update[2] -= float(np.mean(second_update[2]))
            second_to_first_ratios.append(
                float(np.linalg.norm(second_update / field_scale))
                / max(first_norm, 1e-30)
            )
        else:
            second_to_first_ratios.append(0.0)
        nonlinear_remainder_ratios.append(
            float(accepted_residuals[1]) / max(float(accepted_residuals[0]), 1e-30)
        )

        p_slice = problem.equations.variables.adi.indx["p"]
        weak = float(np.linalg.norm(final_residual_vector[p_slice])) / math.sqrt(
            max(p_slice.stop - p_slice.start, 1)
        )
        div, flux, kinetic, ens = _diagnostics(final_grid)
        corrections.append(correction_grid)
        endpoints.append(endpoint_grid)
        predictor_endpoints.append(predictor_endpoint_grid)
        progresses.append([float(progress)])
        valid_masks.append([bool(valid and converged)])
        residual_histories.append([float(value) for value in accepted_residuals])
        nonlinear_iterations.append(int(status.get("n_iter", len(accepted_states) - 1)))
        linear_iterations.append(int(status.get("ls_n_iter", -1)))
        solve_timings.append(elapsed)
        relative_residuals.append(relative_residual)
        weak_continuity.append(weak)
        divergence_rms.append(div)
        net_flux.append(flux)
        energy.append(kinetic)
        enstrophy.append(ens)
        trajectory.append(final_grid)

        if progress_path is not None:
            write_json(
                progress_path,
                {
                    "schema": "blrpng_sfepy_ns_worker_progress_v1",
                    "case_id": int(case["case_id"]),
                    "state": "solving",
                    "completed_steps": _step + 1,
                    "requested_steps": int(steps),
                    "resolution": int(output_resolution),
                    "mesh_resolution": int(mesh_resolution),
                    "dt": float(dt),
                    "time_integrator": time_integrator,
                    "pid": os.getpid(),
                    "latest_step_seconds": float(elapsed),
                    "latest_relative_residual": float(relative_residual),
                    "peak_rss_gib": float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    )
                    / (2**20),
                },
            )

        velocity = np.asarray(solved["u"](), dtype=np.float64).reshape(-1, 2).copy()
        pressure = np.asarray(solved["p"](), dtype=np.float64).reshape(-1).copy()
        oldest_velocity = older_velocity
        older_velocity = velocity_before
        previous_grid = final_grid

    maximum_history = max(len(values) for values in residual_histories)
    residual_array = np.zeros((steps, maximum_history), dtype=np.float64)
    residual_counts = np.zeros(steps, dtype=np.int16)
    for index, values in enumerate(residual_histories):
        residual_array[index, : len(values)] = values
        residual_counts[index] = len(values)
    arrays = {
        "trajectory": np.stack(trajectory).astype(np.float32),
        "boundary_values": boundary_values.astype(np.float32),
        "boundary_lift": lift.astype(np.float32),
        "constraint_mask": constraint_mask,
        "newton_correction": np.stack(corrections).astype(np.float32),
        "endpoint_increment": np.stack(endpoints).astype(np.float32),
        "predictor_endpoint_increment": np.stack(predictor_endpoints).astype(np.float32),
        "residual_progress": np.asarray(progresses, dtype=np.float32),
        "valid_correction_mask": np.asarray(valid_masks, dtype=bool),
        "path_information_mask": np.asarray(path_information_masks, dtype=bool),
        "condition_fields": condition_fields.astype(np.float32),
        "accepted_newton_residuals": residual_array,
        "accepted_newton_counts": residual_counts,
    }
    boundary_error = float(
        np.max(np.abs(arrays["trajectory"][:, :2][:, constraint_mask[:2]] - boundary_values[:2][constraint_mask[:2]]))
    )
    metadata = {
        "schema": "blrpng_sfepy_navier_stokes_worker_v1",
        "equation": "navier_stokes",
        "discretization": "taylor_hood_p2_p1",
        "time_integrator": time_integrator,
        "nonlinear_solver": "newton_line_search",
        "linear_solver": linear_solver,
        "worker_python": sys.executable,
        "output_resolution": int(output_resolution),
        "mesh_resolution": int(mesh_resolution),
        "steps": int(steps),
        "dt": float(dt),
        "reynolds": reynolds,
        "viscosity": viscosity,
        "grad_div_gamma": grad_div_gamma,
        "case": case,
        "solver_converged": all_converged,
        "relative_residuals": relative_residuals,
        "nonlinear_iterations": nonlinear_iterations,
        "linear_iterations": linear_iterations,
        "solver_timing": solve_timings,
        "assembly_time": assembly_seconds,
        "lift_time": lift_seconds,
        "weak_continuity": weak_continuity,
        "grid_divergence_rms": divergence_rms,
        "fe_divergence_qp_rms": fe_divergence_qp_rms,
        "net_flux": net_flux,
        "energy": energy,
        "enstrophy": enstrophy,
        "initial_perturbation_scale": float(
            case.get("initial_perturbation_scale", 0.08)
        ),
        "forcing_temporal_mode": str(
            case["forcing"].get("temporal_mode", "static")
        ),
        "first_update_endpoint_discrepancy": first_update_discrepancies,
        "first_update_predictor_endpoint_discrepancy": first_update_predictor_discrepancies,
        "selected_velocity_endpoint_discrepancy": selected_velocity_discrepancies,
        "selected_pressure_endpoint_discrepancy": selected_pressure_discrepancies,
        "selected_velocity_predictor_endpoint_discrepancy": selected_velocity_predictor_discrepancies,
        "selected_pressure_predictor_endpoint_discrepancy": selected_pressure_predictor_discrepancies,
        "path_selection_mode": path_selection_mode,
        "residual_progress_threshold": float(progress_threshold),
        "newton_predictor_blend": predictor_blend,
        "second_to_first_correction_ratio": second_to_first_ratios,
        "nonlinear_remainder_ratio": nonlinear_remainder_ratios,
        "accepted_newton_path_metrics": accepted_path_metrics,
        "accepted_newton_pressure_path_metrics": accepted_pressure_path_metrics,
        "boundary_error": boundary_error,
        "pressure_gauge_error": float(np.max(np.abs(arrays["trajectory"][:, 2].mean(axis=(1, 2))))),
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 2**20,
        "condition_names": ["viscosity", "dt", "force_x", "force_y"],
        "field_names": ["u", "v", "p"],
        "correction_semantics": (
            "first_accepted_Newton_correction"
            if path_selection_mode == "first_accepted" else
            f"cumulative_accepted_newton_state_at_log_residual_progress_{progress_threshold:g}"
        ),
        "endpoint_semantics": "physical_time_step_increment",
    }
    return arrays, metadata


def generate_steady_continuation(
    case: dict[str, Any],
    *,
    output_resolution: int,
    mesh_resolution: int,
    steps: int,
    dt: float,
    progress_threshold: float,
    path_selection_mode: str,
    linear_solver: str,
    max_newton_iterations: int,
    newton_atol: float,
    newton_rtol: float,
    progress_path: str | Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate steady NS solutions along a Reynolds continuation path.

    The trajectory axis is a PDE-parameter path, not physical time. Each
    accepted Newton path belongs to its own fully nonlinear steady solve.
    """
    schedule = [float(value) for value in case.get("reynolds_schedule", ())]
    if len(schedule) != steps or any(value <= 0 for value in schedule):
        raise ValueError("reynolds_schedule must contain one positive value per step")
    if any(right <= left for left, right in zip(schedule, schedule[1:])):
        raise ValueError("reynolds_schedule must be strictly increasing")
    if path_selection_mode not in {"progress", "first_accepted"}:
        raise ValueError(f"Unsupported path selection mode: {path_selection_mode}")
    blend = float(case.get("newton_predictor_blend", 0.5))
    if not 0 <= blend <= 1:
        raise ValueError("newton_predictor_blend must be within [0, 1]")
    gamma = float(case.get("grad_div_gamma", 100.0))
    boundary = case["boundary"]
    amplitude = float(boundary["amplitude"])
    force_spec = case.get("forcing", {"active": False})
    force_multipliers = [float(v) for v in case.get("continuation_force_multipliers", [1.0] * steps)]
    if len(force_multipliers) != steps:
        raise ValueError("continuation_force_multipliers must match steps")
    lift_velocity, lift_pressure, lift, lift_seconds = _solve_stokes_lift(
        mesh_resolution, output_resolution, boundary, linear_solver, gamma
    )
    axis = np.linspace(0.0, 1.0, output_resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    boundary_values = np.zeros((3, output_resolution, output_resolution), dtype=np.float64)
    boundary_values[0, :, -1] = lid_profile(axis, boundary)
    constraint_mask = np.zeros_like(boundary_values, dtype=bool)
    constraint_mask[:2, 0, :] = True
    constraint_mask[:2, -1, :] = True
    constraint_mask[:2, :, 0] = True
    constraint_mask[:2, :, -1] = True
    velocity = lift_velocity.copy()
    pressure = lift_pressure.copy()
    trajectory: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    predictor_endpoints: list[np.ndarray] = []
    progresses: list[list[float]] = []
    valid_masks: list[list[bool]] = []
    residual_histories: list[list[float]] = []
    condition_sequence: list[np.ndarray] = []
    relative_residuals: list[float] = []
    nonlinear_iterations: list[int] = []
    solve_timings: list[float] = []
    fe_divergence: list[float] = []
    grid_divergence: list[float] = []
    weak_continuity: list[float] = []
    net_flux: list[float] = []
    energy: list[float] = []
    enstrophy: list[float] = []
    selected_velocity_discrepancy: list[float] = []
    selected_pressure_discrepancy: list[float] = []
    all_converged = True
    def build_step_problem(reynolds: float):
        candidate, velocity_field, _pressure_field = _build_problem(
            mesh_resolution,
            boundary=boundary,
            viscosity=amplitude / reynolds,
            dt=None,
            include_convection=True,
            include_forcing=bool(force_spec.get("active", False)),
            linear_solver=linear_solver,
            grad_div_gamma=gamma,
        )
        candidate.get_nls().conf.i_max = int(max_newton_iterations)
        candidate.get_nls().conf.eps_a = float(newton_atol)
        candidate.get_nls().conf.eps_r = float(newton_rtol)
        return candidate, velocity_field

    assembly_started = time.perf_counter()
    problem, velocity_field = build_step_problem(schedule[0])
    assembly_seconds = time.perf_counter() - assembly_started
    _mixed_state(problem, velocity, pressure)
    previous_grid = _grid_fields(problem, problem.equations.variables.get_state(
        problem.active_only, force=True
    ), points, xx.shape, boundary)
    trajectory.append(previous_grid.copy())
    if progress_path is not None:
        write_json(progress_path, {
            "schema": "blrpng_sfepy_ns_worker_progress_v1", "case_id": int(case["case_id"]),
            "state": "solving", "completed_steps": 0, "requested_steps": steps,
            "resolution": output_resolution, "mesh_resolution": mesh_resolution,
            "time_integrator": "STEADY_CONTINUATION", "pid": os.getpid(),
        })
    for step in range(steps):
        if step > 0:
            assembly_started = time.perf_counter()
            problem, velocity_field = build_step_problem(schedule[step])
            assembly_seconds += time.perf_counter() - assembly_started
        predictor_velocity = lift_velocity + blend * (velocity - lift_velocity)
        predictor_pressure = blend * pressure + (1.0 - blend) * lift_pressure
        state0 = _mixed_state(problem, predictor_velocity, predictor_pressure)
        local_force = dict(force_spec)
        local_force["amplitude"] = float(local_force.get("amplitude", 0.0)) * force_multipliers[step]
        force_velocity = forcing_field(velocity_field.get_coor(), local_force)
        force_grid = forcing_field(points, local_force).reshape(*xx.shape, 2)
        condition_sequence.append(np.stack((
            np.full_like(xx, amplitude / schedule[step]),
            np.full_like(xx, (step + 1) / steps),
            force_grid[..., 0], force_grid[..., 1],
        )))
        accepted_states: list[np.ndarray] = []
        accepted_residuals: list[float] = []

        def iter_hook(context, solver, vector, iteration, error, error0):
            del context, iteration, error, error0
            current = np.asarray(vector, dtype=np.float64).copy()
            residual = float(np.linalg.norm(np.asarray(solver.fun(current))))
            if not accepted_states or np.linalg.norm(current - accepted_states[-1]) > 1e-14:
                accepted_states.append(current)
                accepted_residuals.append(residual)

        problem.nls_iter_hook = iter_hook
        problem.get_nls().iter_hook = iter_hook
        status: dict[str, Any] = {}
        problem.get_nls().status = status
        started = time.perf_counter()
        solved = problem.solve(
            state0=state0.copy(),
            var_data={"force": force_velocity.reshape(-1)} if force_spec.get("active", False) else None,
            save_results=False,
            verbose=False,
            status=status,
        )
        elapsed = time.perf_counter() - started
        final_reduced = np.asarray(solved.get_state(problem.active_only, force=True), dtype=np.float64).copy()
        final_residual_vector = np.asarray(problem.get_nls().fun(final_reduced), dtype=np.float64)
        final_residual = float(np.linalg.norm(final_residual_vector))
        if not accepted_states or np.linalg.norm(final_reduced - accepted_states[-1]) > 1e-13:
            accepted_states.append(final_reduced)
            accepted_residuals.append(final_residual)
        relative_residual = final_residual / max(accepted_residuals[0], 1e-30)
        converged = bool(int(status.get("condition", 1)) == 0 and relative_residual <= max(newton_rtol, 1e-12))
        all_converged = all_converged and converged
        if path_selection_mode == "first_accepted":
            valid = len(accepted_states) >= 2 and np.linalg.norm(accepted_states[1] - accepted_states[0]) > 1e-13
            initial = accepted_states[0]
            correction = accepted_states[1] - initial if valid else np.zeros_like(initial)
            progress = float(np.clip(
                (math.log(max(accepted_residuals[0], 1e-30)) - math.log(max(accepted_residuals[1], 1e-30)))
                / max(math.log(max(accepted_residuals[0], 1e-30)) - math.log(max(accepted_residuals[-1], 1e-30)), 1e-12),
                0.0, 1.0,
            )) if valid else 0.0
        else:
            initial, correction, progress, valid = select_progress_update(
                accepted_states, accepted_residuals, progress_threshold
            )
        initial_grid = _grid_fields(problem, initial, points, xx.shape, boundary)
        selected_grid = _grid_fields(problem, initial + correction, points, xx.shape, boundary)
        final_grid = _grid_fields(problem, final_reduced, points, xx.shape, boundary)
        correction_grid = selected_grid - initial_grid
        endpoint_grid = final_grid - previous_grid
        predictor_endpoint_grid = final_grid - initial_grid
        for value in (correction_grid, endpoint_grid, predictor_endpoint_grid):
            value[:2][constraint_mask[:2]] = 0.0
            value[2] -= float(np.mean(value[2]))
        for field_slice, destination in (
            (slice(0, 2), selected_velocity_discrepancy),
            (slice(2, 3), selected_pressure_discrepancy),
        ):
            reference = predictor_endpoint_grid[field_slice]
            destination.append(float(np.linalg.norm(correction_grid[field_slice] - reference))
                               / max(float(np.linalg.norm(reference)), 1e-30))
        divergence_qp = np.asarray(problem.evaluate(
            "ev_div.integral.Omega(u)", mode="qp",
            integrals=Integrals([Integral("integral", order=4)]),
            u=problem.equations.variables["u"],
        ), dtype=np.float64)
        if not np.all(np.isfinite(divergence_qp)):
            raise RuntimeError("Nonfinite finite-element divergence in steady continuation")
        p_slice = problem.equations.variables.adi.indx["p"]
        weak_continuity.append(float(np.linalg.norm(final_residual_vector[p_slice]))
                               / math.sqrt(max(p_slice.stop - p_slice.start, 1)))
        div, flux, kinetic, ens = _diagnostics(final_grid)
        fe_divergence.append(float(np.sqrt(np.mean(divergence_qp**2))))
        grid_divergence.append(div)
        net_flux.append(flux)
        energy.append(kinetic)
        enstrophy.append(ens)
        trajectory.append(final_grid)
        corrections.append(correction_grid)
        endpoints.append(endpoint_grid)
        predictor_endpoints.append(predictor_endpoint_grid)
        progresses.append([progress])
        valid_masks.append([bool(valid and converged)])
        residual_histories.append([float(value) for value in accepted_residuals])
        relative_residuals.append(relative_residual)
        nonlinear_iterations.append(int(status.get("n_iter", len(accepted_states) - 1)))
        solve_timings.append(elapsed)
        velocity = np.asarray(solved["u"](), dtype=np.float64).reshape(-1, 2).copy()
        pressure = np.asarray(solved["p"](), dtype=np.float64).reshape(-1).copy()
        previous_grid = final_grid
        if progress_path is not None:
            write_json(progress_path, {
                "schema": "blrpng_sfepy_ns_worker_progress_v1", "case_id": int(case["case_id"]),
                "state": "solving", "completed_steps": step + 1, "requested_steps": steps,
                "resolution": output_resolution, "mesh_resolution": mesh_resolution,
                "time_integrator": "STEADY_CONTINUATION", "pid": os.getpid(),
                "latest_step_seconds": elapsed, "latest_relative_residual": relative_residual,
                "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 2**20,
            })
        # A sparse direct factorization is cached by ScipyDirect. Retaining
        # every Reynolds-step Problem makes memory grow roughly linearly in
        # continuation length, so release each one before the next step.
        del problem, velocity_field, solved
        gc.collect()
    maximum_history = max(len(values) for values in residual_histories)
    residual_array = np.zeros((steps, maximum_history), dtype=np.float64)
    residual_counts = np.zeros(steps, dtype=np.int16)
    for index, values in enumerate(residual_histories):
        residual_array[index, :len(values)] = values
        residual_counts[index] = len(values)
    arrays = {
        "trajectory": np.stack(trajectory).astype(np.float32),
        "boundary_values": boundary_values.astype(np.float32),
        "boundary_lift": lift.astype(np.float32),
        "constraint_mask": constraint_mask,
        "newton_correction": np.stack(corrections).astype(np.float32),
        "endpoint_increment": np.stack(endpoints).astype(np.float32),
        "predictor_endpoint_increment": np.stack(predictor_endpoints).astype(np.float32),
        "residual_progress": np.asarray(progresses, dtype=np.float32),
        "valid_correction_mask": np.asarray(valid_masks, dtype=bool),
        "condition_fields": np.stack(condition_sequence).astype(np.float32),
        "accepted_newton_residuals": residual_array,
        "accepted_newton_counts": residual_counts,
    }
    boundary_error = float(np.max(np.abs(
        arrays["trajectory"][:, :2][:, constraint_mask[:2]]
        - boundary_values[:2][constraint_mask[:2]]
    )))
    metadata = {
        "schema": "blrpng_sfepy_navier_stokes_worker_v1",
        "equation": "steady_navier_stokes_continuation",
        "discretization": "taylor_hood_p2_p1_grad_div",
        "trajectory_axis": "reynolds_continuation_not_physical_time",
        "time_integrator": "STEADY_CONTINUATION",
        "nonlinear_solver": "newton_line_search",
        "linear_solver": linear_solver,
        "worker_python": sys.executable,
        "output_resolution": int(output_resolution),
        "mesh_resolution": int(mesh_resolution),
        "steps": int(steps),
        "dt": float(dt),
        "reynolds_schedule": schedule,
        "grad_div_gamma": gamma,
        "newton_predictor_blend": blend,
        "path_selection_mode": path_selection_mode,
        "residual_progress_threshold": float(progress_threshold),
        "case": case,
        "solver_converged": all_converged,
        "relative_residuals": relative_residuals,
        "nonlinear_iterations": nonlinear_iterations,
        "linear_iterations": [-1] * steps,
        "solver_timing": solve_timings,
        "assembly_time": assembly_seconds,
        "lift_time": lift_seconds,
        "weak_continuity": weak_continuity,
        "fe_divergence_qp_rms": fe_divergence,
        "grid_divergence_rms": grid_divergence,
        "net_flux": net_flux,
        "energy": energy,
        "enstrophy": enstrophy,
        "selected_velocity_predictor_endpoint_discrepancy": selected_velocity_discrepancy,
        "selected_pressure_predictor_endpoint_discrepancy": selected_pressure_discrepancy,
        "boundary_error": boundary_error,
        "pressure_gauge_error": float(np.max(np.abs(arrays["trajectory"][:, 2].mean(axis=(1, 2))))),
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 2**20,
        "condition_names": ["viscosity", "continuation_progress", "force_x", "force_y"],
        "field_names": ["u", "v", "p"],
        "correction_semantics": "selected_accepted_newton_update_from_deterministic_predictor",
        "endpoint_semantics": "difference_between_consecutive_steady_continuation_states",
        "predictor_endpoint_semantics": "fully_converged_steady_state_minus_newton_predictor",
    }
    return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--output-resolution", type=int, required=True)
    parser.add_argument("--mesh-resolution", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--progress-threshold", type=float, default=0.25)
    parser.add_argument(
        "--path-selection-mode", choices=("progress", "first_accepted"), default="progress"
    )
    parser.add_argument("--linear-solver", choices=("direct", "gmres_ilu"), default="direct")
    parser.add_argument("--max-newton-iterations", type=int, default=20)
    parser.add_argument("--newton-atol", type=float, default=1e-10)
    parser.add_argument("--newton-rtol", type=float, default=1e-8)
    parser.add_argument(
        "--time-integrator", choices=("BDF1", "BDF2", "BDF3", "STEADY_CONTINUATION"), default="BDF1"
    )
    parser.add_argument("--max-address-space-gib", type=float, default=16.0)
    parser.add_argument("--progress-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 2.0 <= args.max_address_space_gib <= 128.0:
        raise ValueError("The SfePy worker address-space limit must be in [2, 128] GiB.")
    address_limit = int(args.max_address_space_gib * (2**30))
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    del current_soft
    if current_hard != resource.RLIM_INFINITY:
        address_limit = min(address_limit, current_hard)
    resource.setrlimit(resource.RLIMIT_AS, (address_limit, current_hard))
    output.set_output(quiet=True)
    case_path = resolve_in_project(args.case_json, must_exist=True)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    arrays, metadata = generate_rollout(
        case,
        output_resolution=args.output_resolution,
        mesh_resolution=args.mesh_resolution,
        steps=args.steps,
        dt=args.dt,
        progress_threshold=args.progress_threshold,
        path_selection_mode=args.path_selection_mode,
        linear_solver=args.linear_solver,
        max_newton_iterations=args.max_newton_iterations,
        newton_atol=args.newton_atol,
        newton_rtol=args.newton_rtol,
        time_integrator=args.time_integrator,
        progress_path=(
            ensure_artifact_path(args.progress_json)
            if args.progress_json is not None
            else None
        ),
    )
    output_npz = ensure_artifact_path(args.output_npz)
    output_meta = ensure_artifact_path(args.output_meta)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = ensure_artifact_path(output_npz.with_suffix(".partial.npz"))
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_npz.replace(output_npz)
    directory_fd = os.open(output_npz.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    write_json(output_meta, metadata)
    if args.progress_json is not None:
        write_json(
            args.progress_json,
            {
                "schema": "blrpng_sfepy_ns_worker_progress_v1",
                "case_id": int(case["case_id"]),
                "state": "complete",
                "completed_steps": int(args.steps),
                "requested_steps": int(args.steps),
                "resolution": int(args.output_resolution),
                "mesh_resolution": int(args.mesh_resolution),
                "dt": float(args.dt),
                "time_integrator": args.time_integrator,
                "pid": os.getpid(),
                "peak_rss_gib": float(metadata["peak_rss_gib"]),
            },
        )


if __name__ == "__main__":
    main()
