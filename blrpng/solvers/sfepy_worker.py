"""SfePy-only Allen--Cahn trajectory generator.

This module is launched with ``sfepy_env`` by the parent orchestrator.  It
intentionally writes NumPy/JSON only; PyTorch serialization stays in the
training environment.
"""

from __future__ import annotations

import argparse
import hashlib
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
from blrpng.solvers.progress import (
    LOG_DENOMINATOR_FLOOR,
    NONZERO_STEP_NORM_THRESHOLD,
    RESIDUAL_FLOOR,
    replay_progress_selection,
    select_progress_update,
)


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


QUERY_CASE_MANIFEST_TEMPLATE = (
    "artifacts/manifests/phase1_confirmation/{campaign}/validation_hr.json"
)
QUERY_PROTOCOL = {
    "output_resolution": 256,
    "mesh_resolution": 257,
    "steps": 20,
    "fidelity": 1,
    "dt": 0.2,
    "epsilon": 0.03,
    "gamma": 5.0,
    "progress_threshold": 0.25,
    "max_newton_iters": 20,
    "newton_atol": 1e-10,
    "newton_rtol": 1e-8,
    "case_start": 0,
}
# Pin the frozen development task/validation manifests. A self-seal alone is
# not an authorization boundary because a caller could rewrite both the tasks
# and seal; copied confirmation traces must not enter a development-HR route.
CANONICAL_DEVELOPMENT_CAMPAIGN = "hrdev_20261012"
CANONICAL_DEVELOPMENT_TASK_SHA256 = (
    "bb50c7e4d5b8d0900e21d7ae64235c6d955244cc3bfdca1a97ab6056596a334a"
)
CANONICAL_DEVELOPMENT_HR_MANIFEST_SHA256 = (
    "d491452f11d9c75b62c385fd9ecf2df40b960053782e0b2c31a63a819cd21ac5"
)
CANONICAL_DEVELOPMENT_HR_SUPPORT_CAMPAIGN = "hrdev_20261015_hrsupport_s271829"
CANONICAL_DEVELOPMENT_HR_SUPPORT_MANIFEST = (
    "artifacts/manifests/phase1_development/"
    "hrdev_20261015_hrsupport_s271829/support_hr_s271829.json"
)
CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST = (
    "artifacts/manifests/phase1_development/"
    "hrdev_20261012_t20_ss271829/support_s271829.json"
)
CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST_SHA256 = (
    "dfbc58d5b62b870b60804df9b33fad8a27cc5370c6070aee679028ddaf098138"
)
CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX = (
    "artifacts/data/phase1_development/"
    "hrdev_20261013_firstpath_ss271829/support_s271829_64.index.json"
)
CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX_SHA256 = (
    "c5bea773b1f414c2d2deaa9bee9e35343a38b6d7c4268a86505a6585113499f1"
)
CANONICAL_DEVELOPMENT_HR_SUPPORT_TASKS = (0, 1, 3, 5)
CANONICAL_DEVELOPMENT_HR_SUPPORT_M4_INDEX_SHA256 = {
    0: "cb66e634789b44cc11be7d58bd99bb442e60ffacd6362e1db0dd1c650974ef85",
    1: "2f90cc5632c5c1d6df6a798d2db88a9e094f6e52709dcaef9b4bfe513c4753e6",
    3: "efef59e67144b17f49ae7fda8784219c5ae0e4c879a9d047bdfb99ec86be576f",
    5: "3a8a35325729f3d8af76b5b6fa07c82a0fe8ae16c509f138ca86002e06f85bf2",
}
CANONICAL_DEVELOPMENT_HR_SUPPORT_M16_CAMPAIGN = (
    "hrdev_20261015_hrsupport_m16_s271829"
)
CANONICAL_DEVELOPMENT_HR_SUPPORT_M16_MANIFEST = (
    "artifacts/manifests/phase1_development/"
    "hrdev_20261015_hrsupport_m16_s271829/support_hr_s271829_m16.json"
)
CANONICAL_DEVELOPMENT_HR_SUPPORT_M16_INDEX_SHA256 = {
    0: "5e3261ff2fd825c9d3f01f67c4d1afb3f6438af55608c1a8763b1557877507bc",
    1: "97586326540dc9d5a3c288e9930b66e603a3402dfff116adccc70fa854509eeb",
    3: "f2a17dec1138e633845cafe431d9e59df76eafe31278565f2840a885a2eea93f",
    5: "54ee8bc914c4cb74ae71be2741f32e6646fa886e4b187d6771a1005150ebaac8",
}
QUERY_RUNTIME_CODE_PATHS = {
    "blrpng/paths.py",
    "blrpng/data/manifest.py",
    "blrpng/data/lifting.py",
    "blrpng/data/blind_boundaries.py",
    "blrpng/data/shards.py",
    "blrpng/solvers/progress.py",
    "blrpng/data/generate.py",
}


def _stdlib_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stdlib_sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _boundary_semantics(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _boundary_semantics(item)
            for key, item in value.items()
            if key not in {"boundary_id", "sha256"}
        }
    if isinstance(value, list):
        return [_boundary_semantics(item) for item in value]
    return value


def _references_frozen_confirmation_boundary(
    project_root: Path, cases: list[dict[str, Any]], *, requested_mesh_resolution: int
) -> bool:
    """Recognize copied/relabelled cases by their frozen boundary identity."""
    manifest_root = project_root / "artifacts/manifests/phase1_confirmation"
    case_boundary_ids = {int(case["boundary_id"]) for case in cases if "boundary_id" in case}
    case_descriptors = {
        _stdlib_sha256_json(_boundary_semantics(case["boundary_descriptor"]))
        for case in cases if isinstance(case.get("boundary_descriptor"), dict)
    }
    if not cases:
        return False
    # Compare boundary values on the exact requested and canonical LR/HR FE
    # edge nodes. A sparse generic trace can miss Fourier aliases that agree
    # on the solver's actual mesh.
    traces = []
    mesh_resolutions = {65, 257, max(2, int(requested_mesh_resolution))}
    for mesh_resolution in sorted(mesh_resolutions):
        edge_t = np.linspace(0.0, 1.0, mesh_resolution, endpoint=True, dtype=np.float64)
        trace_x = np.concatenate((edge_t, np.ones_like(edge_t), 1.0 - edge_t, np.zeros_like(edge_t)))
        trace_y = np.concatenate((np.zeros_like(edge_t), edge_t, np.ones_like(edge_t), 1.0 - edge_t))
        traces.append((trace_x, trace_y))
    for task_path in manifest_root.glob("*/tasks.json"):
        try:
            payload = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") != "blrpng_phase1_confirmation_tasks_v1":
            continue
        if payload.get("sha256") != _stdlib_sha256_json(
            {key: value for key, value in payload.items() if key != "sha256"}
        ):
            continue
        for task in payload.get("tasks", []):
            if int(task.get("boundary_id", -1)) in case_boundary_ids:
                return True
            descriptor = task.get("boundary")
            if (
                isinstance(descriptor, dict)
                and _stdlib_sha256_json(_boundary_semantics(descriptor)) in case_descriptors
            ):
                return True
            if isinstance(descriptor, dict):
                reference = {"boundary_descriptor": descriptor}
                for case in cases:
                    if not any(
                        key in case
                        for key in ("boundary_coefficients", "boundary_descriptor")
                    ):
                        continue
                    try:
                        matches_mesh_trace = False
                        for trace_x, trace_y in traces:
                            reference_trace = evaluate_case_boundary(trace_x, trace_y, reference)
                            case_trace = evaluate_case_boundary(trace_x, trace_y, case)
                            if np.allclose(case_trace, reference_trace, rtol=0.0, atol=1.0e-10):
                                matches_mesh_trace = True
                                break
                    except (KeyError, TypeError, ValueError):
                        continue
                    if matches_mesh_trace:
                        return True
    return False


def _is_canonical_development_hr_manifest(
    project_root: Path, manifest_rel: str, manifest: dict[str, Any]
) -> bool:
    metadata = manifest.get("metadata", {})
    campaign = str(metadata.get("campaign_id", ""))
    task_manifest_rel = str(metadata.get("task_manifest", ""))
    expected_case = f"artifacts/manifests/phase1_development/{campaign}/validation_hr.json"
    expected_tasks = f"artifacts/manifests/phase1_development/{campaign}/tasks.json"
    if (
        campaign != CANONICAL_DEVELOPMENT_CAMPAIGN
        or manifest_rel != expected_case
        or manifest.get("sha256") != CANONICAL_DEVELOPMENT_HR_MANIFEST_SHA256
        or task_manifest_rel != expected_tasks
        or metadata.get("role") != "validation_hr"
        or metadata.get("evaluation_role") != "validation_hr"
        or manifest.get("schema") != "blrpng_manifest_v1"
        or manifest.get("sha256") != _stdlib_sha256_json(
            {key: value for key, value in manifest.items() if key != "sha256"}
        )
    ):
        return False
    task_path = project_root / expected_tasks
    if not task_path.is_file():
        return False
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    task_sha = task_manifest.get("sha256")
    if (
        task_manifest.get("schema") != "blrpng_phase1_development_tasks_v1"
        or task_manifest.get("campaign_id") != campaign
        or task_sha != CANONICAL_DEVELOPMENT_TASK_SHA256
        or task_sha != _stdlib_sha256_json(
            {key: value for key, value in task_manifest.items() if key != "sha256"}
        )
        or metadata.get("task_manifest_sha256") != task_sha
    ):
        return False
    tasks = {int(row["target_task_id"]): row for row in task_manifest.get("tasks", [])}
    cases = manifest.get("cases", [])
    if not cases:
        return False
    for case in cases:
        task_id = int(case.get("target_task_id", -1))
        task = tasks.get(task_id)
        if task is None or (
            int(case.get("boundary_id", -1)) != int(task["boundary_id"])
            or case.get("boundary_descriptor") != task.get("boundary")
            or case.get("boundary_family") != task.get("family")
            or case.get("split") != "validation_hr"
        ):
            return False
    return True


def _is_canonical_development_hr_support_manifest(
    project_root: Path, manifest_rel: str, manifest: dict[str, Any]
) -> bool:
    """Authorize only the frozen m=4 or m=16 development HR support cases.

    The case IDs, IC coefficients, phases, and boundary descriptors are copied
    from the existing Stage-2 LR support draw. HR labels are disjoint from the
    canonical HR development validation IDs. The m=16 extension is pinned to
    the pre-existing nested LR support indices for the same draw.
    """
    if manifest_rel == CANONICAL_DEVELOPMENT_HR_SUPPORT_MANIFEST:
        support_budget = 4
        expected_campaign = CANONICAL_DEVELOPMENT_HR_SUPPORT_CAMPAIGN
        expected_index_hashes = CANONICAL_DEVELOPMENT_HR_SUPPORT_M4_INDEX_SHA256
    elif manifest_rel == CANONICAL_DEVELOPMENT_HR_SUPPORT_M16_MANIFEST:
        support_budget = 16
        expected_campaign = CANONICAL_DEVELOPMENT_HR_SUPPORT_M16_CAMPAIGN
        expected_index_hashes = CANONICAL_DEVELOPMENT_HR_SUPPORT_M16_INDEX_SHA256
    else:
        return False
    metadata = manifest.get("metadata", {})
    task_manifest_rel = (
        "artifacts/manifests/phase1_development/"
        f"{CANONICAL_DEVELOPMENT_CAMPAIGN}/tasks.json"
    )
    if (
        manifest.get("schema") != "blrpng_manifest_v1"
        or manifest.get("sha256") != _stdlib_sha256_json(
            {key: value for key, value in manifest.items() if key != "sha256"}
        )
        or metadata.get("campaign_id") != expected_campaign
        or metadata.get("role") != "support_hr"
        or metadata.get("evaluation_role") != "support_hr_dev"
        or int(metadata.get("support_seed", -1)) != 271829
        or int(metadata.get("initial_conditions_per_task", -1)) != support_budget
        or metadata.get("task_manifest") != task_manifest_rel
        or metadata.get("source_support_manifest")
        != CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST
        or metadata.get("source_support_manifest_sha256")
        != CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST_SHA256
        or metadata.get("source_support_index")
        != CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX
        or metadata.get("source_support_index_sha256")
        != CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX_SHA256
    ):
        return False

    task_path = project_root / task_manifest_rel
    support_path = project_root / CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST
    support_index_path = project_root / CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX
    validation_path = project_root / (
        f"artifacts/manifests/phase1_development/"
        f"{CANONICAL_DEVELOPMENT_CAMPAIGN}/validation_hr.json"
    )
    if not all(path.is_file() for path in (task_path, support_path, support_index_path, validation_path)):
        return False

    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    support_manifest = json.loads(support_path.read_text(encoding="utf-8"))
    support_index = json.loads(support_index_path.read_text(encoding="utf-8"))
    validation_manifest = json.loads(validation_path.read_text(encoding="utf-8"))
    task_sha = task_manifest.get("sha256")
    if (
        task_manifest.get("schema") != "blrpng_phase1_development_tasks_v1"
        or task_manifest.get("campaign_id") != CANONICAL_DEVELOPMENT_CAMPAIGN
        or task_sha != CANONICAL_DEVELOPMENT_TASK_SHA256
        or task_sha != _stdlib_sha256_json(
            {key: value for key, value in task_manifest.items() if key != "sha256"}
        )
        or metadata.get("task_manifest_sha256") != task_sha
        or support_manifest.get("sha256")
        != CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST_SHA256
        or support_manifest.get("sha256") != _stdlib_sha256_json(
            {key: value for key, value in support_manifest.items() if key != "sha256"}
        )
        or support_index.get("sha256") != CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX_SHA256
        or support_index.get("sha256") != _stdlib_sha256_json(
            {key: value for key, value in support_index.items() if key != "sha256"}
        )
        or validation_manifest.get("sha256")
        != CANONICAL_DEVELOPMENT_HR_MANIFEST_SHA256
        or validation_manifest.get("sha256") != _stdlib_sha256_json(
            {key: value for key, value in validation_manifest.items() if key != "sha256"}
        )
    ):
        return False

    support_meta = support_manifest.get("metadata", {})
    support_index_meta = support_index.get("metadata", {})
    if (
        support_meta.get("role") != "support"
        or support_meta.get("campaign_id") != "hrdev_20261012_t20_ss271829"
        or int(support_meta.get("support_seed", -1)) != 271829
        or support_meta.get("task_manifest_sha256") != task_sha
        or support_index_meta.get("manifest_sha256")
        != CANONICAL_DEVELOPMENT_LR_SUPPORT_MANIFEST_SHA256
        or support_index_meta.get("task_manifest_sha256") != task_sha
        or support_index_meta.get("split") != "support"
        or int(support_index_meta.get("resolution", -1)) != 64
        or int(support_index_meta.get("steps", -1)) != 20
        or int(support_index_meta.get("fidelity", -1)) != 0
        or int(support_index_meta.get("support_seed", -1)) != 271829
    ):
        return False

    tasks = {int(row["target_task_id"]): row for row in task_manifest.get("tasks", [])}
    parent_ids = {int(row.get("initial_condition_id", -1)) for row in support_index.get("shards", [])}
    expected_cases = []
    expected_source_indices = []
    for task_id in CANONICAL_DEVELOPMENT_HR_SUPPORT_TASKS:
        task = tasks.get(task_id)
        if task is None:
            return False
        subset_rel = (
            "artifacts/data/phase1_development/"
            f"hrdev_20261013_firstpath_ss271829/"
            f"task{task_id:02d}_support_s271829_m{support_budget}.index.json"
        )
        subset_path = project_root / subset_rel
        if not subset_path.is_file():
            return False
        subset = json.loads(subset_path.read_text(encoding="utf-8"))
        subset_sha = subset.get("sha256")
        if (
            subset_sha != expected_index_hashes[task_id]
            or subset_sha != _stdlib_sha256_json(
                {key: value for key, value in subset.items() if key != "sha256"}
            )
            or subset.get("metadata", {}).get("parent_sha256")
            != CANONICAL_DEVELOPMENT_LR_SUPPORT_INDEX_SHA256
            or int(subset.get("metadata", {}).get("target_task_id", -1)) != task_id
            or len(subset.get("shards", [])) != support_budget
        ):
            return False
        subset_rows = sorted(
            subset["shards"], key=lambda row: int(row.get("support_rank", -1))
        )
        if [int(row.get("support_rank", -1)) for row in subset_rows] != list(range(support_budget)):
            return False
        expected_source_indices.append({
            "task": task_id,
            "path": subset_rel,
            "sha256": subset_sha,
        })
        for row in subset_rows:
            initial_id = int(row.get("initial_condition_id", -1))
            if initial_id not in parent_ids or (
                int(row.get("target_task_id", -1)) != task_id
                or row.get("role") != "support"
                or int(row.get("resolution", -1)) != 64
                or int(row.get("steps", -1)) != 20
            ):
                return False
        expected_cases.extend(
            case
            for case in support_manifest.get("cases", [])
            if int(case.get("target_task_id", -1)) == task_id
            and int(case.get("support_rank", -1)) < support_budget
        )

    if len(expected_cases) != len(CANONICAL_DEVELOPMENT_HR_SUPPORT_TASKS) * support_budget:
        return False
    if support_budget == 16 and metadata.get("source_m16_indices") != expected_source_indices:
        return False
    expected_cases.sort(key=lambda case: (int(case["target_task_id"]), int(case["support_rank"])))
    expected_ids = {int(case["initial_condition_id"]) for case in expected_cases}
    if expected_ids != {
        int(row["initial_condition_id"])
        for task_id in CANONICAL_DEVELOPMENT_HR_SUPPORT_TASKS
        for row in json.loads(
            (project_root / (
                "artifacts/data/phase1_development/"
                f"hrdev_20261013_firstpath_ss271829/"
                f"task{task_id:02d}_support_s271829_m{support_budget}.index.json"
            )).read_text(encoding="utf-8")
        )["shards"]
    }:
        return False

    validation_ids = {
        int(case["initial_condition_id"])
        for case in validation_manifest.get("cases", [])
    }
    if expected_ids & validation_ids:
        return False
    normalized_expected = []
    for case in expected_cases:
        cloned = dict(case)
        cloned["split"] = "support_hr"
        normalized_expected.append(cloned)
        task = tasks[int(case["target_task_id"])]
        if (
            int(case.get("boundary_id", -1)) != int(task["boundary_id"])
            or case.get("boundary_descriptor") != task.get("boundary")
            or case.get("boundary_family") != task.get("family")
        ):
            return False
    actual_cases = manifest.get("cases", [])
    return actual_cases == normalized_expected


def _runtime_solver_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_resolution": int(args.output_resolution),
        "mesh_resolution": int(args.mesh_resolution),
        "steps": int(args.steps),
        "fidelity": int(args.fidelity),
        "dt": float(args.dt),
        "epsilon": float(args.epsilon),
        "gamma": float(args.gamma),
        "progress_threshold": float(args.progress_threshold),
        "max_newton_iters": int(args.max_newton_iters),
        "newton_atol": float(args.newton_atol),
        "newton_rtol": float(args.newton_rtol),
        "case_start": int(args.case_start),
    }


def _authorize_confirmation_query_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    project_root = Path(__file__).resolve().parents[2]
    supplied_path = Path(args.manifest)
    manifest_path = (project_root / supplied_path).resolve() if not supplied_path.is_absolute() else supplied_path.resolve()
    try:
        manifest_rel = str(manifest_path.relative_to(project_root))
    except ValueError as exc:
        raise ValueError("SfePy input manifest must stay inside the project root.") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    task_manifest_rel = str(metadata.get("task_manifest", ""))
    confirmation_manifest_root = "artifacts/manifests/phase1_confirmation/"
    task_manifest_is_confirmation = task_manifest_rel.startswith(confirmation_manifest_root)
    if task_manifest_rel:
        candidate_task_manifest = (project_root / task_manifest_rel).resolve()
        try:
            candidate_task_manifest.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("Task manifest must stay inside the project root.") from exc
        if candidate_task_manifest.is_file():
            task_payload = json.loads(candidate_task_manifest.read_text(encoding="utf-8"))
            task_manifest_is_confirmation |= (
                task_payload.get("schema") == "blrpng_phase1_confirmation_tasks_v1"
            )
    case_manifest_is_confirmation = manifest_rel.startswith(confirmation_manifest_root)
    high_resolution_request = (
        args.split == "validation_hr"
        or int(args.output_resolution) > 64
        or int(args.mesh_resolution) > 65
        or int(args.steps) > 10
        or int(args.fidelity) == 1
    )
    canonical_development_hr = high_resolution_request and _is_canonical_development_hr_manifest(
        project_root, manifest_rel, manifest
    )
    canonical_development_hr_support = (
        high_resolution_request
        and _is_canonical_development_hr_support_manifest(
            project_root, manifest_rel, manifest
        )
    )
    copied_confirmation_cases = _references_frozen_confirmation_boundary(
        project_root,
        manifest.get("cases", []),
        requested_mesh_resolution=int(args.mesh_resolution),
    )
    is_query_route = metadata.get("evaluation_role") == "query" or (
        task_manifest_is_confirmation or case_manifest_is_confirmation
    ) or copied_confirmation_cases
    if not is_query_route:
        if high_resolution_request and not (
            canonical_development_hr or canonical_development_hr_support
        ):
            raise PermissionError(
                "High-resolution generation requires the canonical development HR validation "
                "manifest, the exact frozen m=4 development support cases, or the frozen "
                "confirmation query gate."
            )
        if high_resolution_request and _runtime_solver_parameters(args) != QUERY_PROTOCOL:
            raise PermissionError(
                "Development HR generation must use the frozen HR solver protocol."
            )
        if canonical_development_hr_support and (
            args.split != "support_hr"
            or args.case_start != 0
            or args.limit != len(manifest.get("cases", []))
            or len(manifest.get("cases", []))
            != int(manifest.get("metadata", {}).get("initial_conditions_per_task", -1))
            * len(CANONICAL_DEVELOPMENT_HR_SUPPORT_TASKS)
        ):
            raise PermissionError(
                "Development HR support generation must cover the complete frozen "
                "representative-task support manifest."
            )
        return None

    campaign = str(metadata.get("campaign_id", ""))
    expected_task_manifest = f"artifacts/manifests/phase1_confirmation/{campaign}/tasks.json"
    expected_case_manifest = QUERY_CASE_MANIFEST_TEMPLATE.format(campaign=campaign)
    if (
        not campaign
        or task_manifest_rel != expected_task_manifest
        or manifest_rel != expected_case_manifest
    ):
        raise PermissionError("Confirmation HR query must use its canonical frozen case manifest.")

    gate_path = project_root / "artifacts/manifests/phase1_confirmation" / campaign / "query_gate.json"
    if not gate_path.is_file():
        raise PermissionError("Confirmation HR query is locked until its complete matrix gate is frozen.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("sha256") != _stdlib_sha256_json(
        {key: value for key, value in gate.items() if key != "sha256"}
    ):
        raise ValueError("Confirmation query gate self-seal is invalid.")

    verifier_rel = "blrpng/data/confirmation_query_gate.py"
    worker_rel = "blrpng/solvers/sfepy_worker.py"
    generator_rel = "scripts/prepare_phase1_development.py"
    freezer_rel = "scripts/freeze_phase1_confirmation_query.py"
    if (
        gate.get("campaign_id") != campaign
        or gate.get("query_worker_path") != worker_rel
        or gate.get("query_gate_verifier_path") != verifier_rel
        or gate.get("query_generator_path") != generator_rel
        or gate.get("gate_freezer_path") != freezer_rel
    ):
        raise ValueError("Confirmation query gate does not bind the canonical worker and verifier.")
    runtime_hashes = gate.get("runtime_code_hashes", {})
    if set(runtime_hashes) != QUERY_RUNTIME_CODE_PATHS:
        raise ValueError("Confirmation query gate is missing frozen SfePy runtime code hashes.")
    code_hashes = {
        **runtime_hashes,
        worker_rel: gate.get("query_worker_sha256"),
        verifier_rel: gate.get("query_gate_verifier_sha256"),
        generator_rel: gate.get("query_generator_sha256"),
        freezer_rel: gate.get("gate_freezer_sha256"),
    }
    for rel, expected in code_hashes.items():
        if not expected or _stdlib_sha256_file(project_root / rel) != expected:
            raise ValueError(f"Frozen confirmation query code changed: {rel}")

    task_manifest_path = project_root / expected_task_manifest
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    task_manifest_sha = task_manifest.get("sha256")
    if (
        task_manifest_sha != _stdlib_sha256_json(
            {key: value for key, value in task_manifest.items() if key != "sha256"}
        )
        or metadata.get("task_manifest_sha256") != task_manifest_sha
        or task_manifest.get("campaign_id") != campaign
        or task_manifest.get("schema") != "blrpng_phase1_confirmation_tasks_v1"
        or gate.get("task_manifest_sha256") != task_manifest_sha
    ):
        raise ValueError("Confirmation query case manifest is not bound to the sealed task manifest.")

    matrix_rel = f"artifacts/manifests/phase1_confirmation/{campaign}/matrix.json"
    matrix_path = project_root / matrix_rel
    if _stdlib_sha256_file(matrix_path) != gate.get("matrix_manifest_sha256"):
        raise ValueError("Frozen confirmation matrix changed after query-gate freeze.")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("sha256") != _stdlib_sha256_json(
        {key: value for key, value in matrix.items() if key != "sha256"}
    ):
        raise ValueError("Confirmation matrix self-seal is invalid.")
    query_rollouts = int(matrix.get("query_rollouts", -1))
    query_ic_seed = int(matrix.get("query_ic_seed", -1))
    expected_ic_seed = int(task_manifest.get("seed", -1)) + 30_000
    if (
        matrix.get("query_case_manifest_path") != expected_case_manifest
        or int(matrix.get("query_case_id_base", -1)) != 7_000_000
        or query_ic_seed != expected_ic_seed
        or matrix.get("query_numpy_version") != np.__version__
        or not isinstance(matrix.get("query_runtime_environment"), dict)
        or matrix["query_runtime_environment"].get("numpy") != np.__version__
        or matrix.get("query_solver_protocol") != QUERY_PROTOCOL
    ):
        raise ValueError("Frozen query IC generation or solver protocol is invalid.")
    runtime_parameters = _runtime_solver_parameters(args)
    if runtime_parameters != QUERY_PROTOCOL or args.split != "validation_hr":
        raise ValueError("Confirmation query worker arguments differ from the frozen solver protocol.")

    # Import the gate verifier only after the stdlib bootstrap has checked its
    # frozen bytes. Importing a changed verifier at module load would execute it
    # before this worker-side integrity check.
    from blrpng.data.confirmation_query_gate import (
        current_query_runtime_environment,
        verify_confirmation_query_gate,
    )

    runtime_environment = current_query_runtime_environment()
    if runtime_environment != matrix["query_runtime_environment"]:
        raise ValueError("Confirmation solver runtime differs from the frozen Python/numerical environment.")

    verify_confirmation_query_gate(
        campaign,
        task_manifest,
        query_rollouts=query_rollouts,
        query_steps=int(args.steps),
        runtime_parameters=runtime_parameters,
        query_numpy_version=np.__version__,
        query_runtime_environment=runtime_environment,
    )
    if manifest.get("sha256") != _stdlib_sha256_json(
        {key: value for key, value in manifest.items() if key != "sha256"}
    ):
        raise ValueError("Confirmation query case manifest self-seal is invalid.")
    if (
        metadata.get("role") != "validation_hr"
        or metadata.get("evaluation_role") != "query"
        or metadata.get("task_manifest_sha256") != task_manifest_sha
        or int(metadata.get("initial_conditions_per_task", -1)) != query_rollouts
        or int(metadata.get("initial_condition_seed", -1)) != query_ic_seed
        or metadata.get("support_seed") is not None
    ):
        raise ValueError("Confirmation query case manifest metadata differs from the frozen matrix.")
    tasks = task_manifest["tasks"]
    cases = manifest.get("cases", [])
    total_cases = len(tasks) * query_rollouts
    if args.case_start != 0 or args.limit != total_cases or len(cases) != total_cases:
        raise ValueError("Confirmation query generation must cover the full frozen case manifest.")
    case_id_base = int(matrix["query_case_id_base"])
    for task in tasks:
        task_id = int(task["target_task_id"])
        rng = np.random.default_rng(np.random.SeedSequence([query_ic_seed, 0, task_id]))
        for rank in range(query_rollouts):
            case = cases[task_id * query_rollouts + rank]
            case_id = case_id_base + task_id * 1000 + rank
            expected_coefficients = rng.normal(size=4).tolist()
            expected_phase = float(rng.uniform(0.0, 2.0 * np.pi))
            if (
                int(case.get("target_task_id", -1)) != task_id
                or int(case.get("boundary_id", -1)) != int(task["boundary_id"])
                or int(case.get("case_id", -1)) != case_id
                or int(case.get("pair_id", -1)) != case_id
                or int(case.get("initial_condition_id", -1)) != case_id
                or case.get("split") != "validation_hr"
                or case.get("boundary_descriptor") != task["boundary"]
                or case.get("boundary_family") != task["family"]
                or case.get("boundary_coefficients") != [0.0] * 7
                or case.get("boundary_frequencies") != [1.0, 2.0]
                or float(case.get("boundary_amplitude", float("nan")))
                != float(task["boundary"]["amplitude"])
                or case.get("initial_coefficients") != expected_coefficients
                or float(case.get("initial_phase", float("nan"))) != expected_phase
            ):
                raise ValueError("Confirmation query case differs from the preregistered deterministic generator.")
    return {
        "query_gate_sha256": _stdlib_sha256_file(gate_path),
        "query_runtime_environment": runtime_environment,
    }


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
    query_authorization = _authorize_confirmation_query_manifest(args)
    query_gate_sha256 = (
        None if query_authorization is None else query_authorization["query_gate_sha256"]
    )
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
            "newton_correction_first",
            "newton_correction_p075",
            "residual_progress",
            "residual_progress_first",
            "residual_progress_p075",
            "valid_correction_mask",
            "valid_correction_mask_first",
            "valid_correction_mask_p075",
            "selected_newton_state_index",
            "selected_newton_state_index_first",
            "selected_newton_state_index_p075",
            "accepted_newton_residuals",
            "accepted_newton_step_norms",
            "accepted_newton_counts",
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
            "solver_newton_iterations",
            "solver_seconds",
            "assembly_seconds",
        )
    }
    if any("support_seed" in case for case in cases):
        records["support_seed"] = []
        records["support_rank"] = []
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

            if len(iter_states) != len(iter_residuals) or not iter_states:
                raise RuntimeError("Newton trace states/residuals are empty or misaligned.")
            trace_bound = int(args.max_newton_iters) + 2
            if len(iter_states) > trace_bound:
                raise RuntimeError(
                    f"Newton raw hook trace has {len(iter_states)} states, exceeding "
                    f"the audited bound {trace_bound}."
                )
            residual_trace = np.asarray(iter_residuals, dtype=np.float64)
            if not np.isfinite(residual_trace).all() or np.any(residual_trace < 0.0):
                raise RuntimeError("Newton residual trace contains invalid values.")
            step_norms = np.zeros(len(iter_states), dtype=np.float64)
            for trace_index in range(1, len(iter_states)):
                step_norms[trace_index] = np.linalg.norm(
                    iter_states[trace_index] - iter_states[trace_index - 1]
                )
            if not np.isfinite(step_norms).all():
                raise RuntimeError("Newton raw hook trace contains a non-finite step norm.")
            padded_residuals = np.zeros(trace_bound, dtype=np.float64)
            padded_step_norms = np.zeros(trace_bound, dtype=np.float64)
            padded_residuals[: len(iter_states)] = residual_trace
            padded_step_norms[: len(iter_states)] = step_norms

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

            (
                pre_reduced,
                correction_reduced,
                progress,
                valid,
                selected_index,
            ) = select_progress_update(
                iter_states,
                iter_residuals,
                args.progress_threshold,
                return_index=True,
            )
            (
                pre_reduced_first,
                correction_reduced_first,
                progress_first,
                valid_first,
                selected_index_first,
            ) = select_progress_update(
                iter_states, iter_residuals, 0.0, return_index=True
            )
            (
                pre_reduced_p075,
                correction_reduced_p075,
                progress_p075,
                valid_p075,
                selected_index_p075,
            ) = select_progress_update(
                iter_states, iter_residuals, 0.75, return_index=True
            )
            for threshold, expected in (
                (0.0, (selected_index_first, progress_first, valid_first)),
                (args.progress_threshold, (selected_index, progress, valid)),
                (0.75, (selected_index_p075, progress_p075, valid_p075)),
            ):
                replayed = replay_progress_selection(
                    residual_trace, step_norms, threshold
                )
                if (
                    replayed[0] != expected[0]
                    or replayed[2] != expected[2]
                    or not np.isclose(replayed[1], expected[1], atol=1e-12, rtol=0.0)
                ):
                    raise RuntimeError(
                        f"Newton selection replay mismatch at threshold {threshold}: "
                        f"{replayed} != {expected}."
                    )
            pre_variable = _set_reduced_state(problem, pre_reduced)
            pre_grid = _interpolate(pre_variable, grid_points).reshape(xx.shape)
            post_variable = _set_reduced_state(problem, pre_reduced + correction_reduced)
            post_grid = _interpolate(post_variable, grid_points).reshape(xx.shape)
            correction_grid = post_grid - pre_grid
            pre_variable_first = _set_reduced_state(problem, pre_reduced_first)
            pre_grid_first = _interpolate(pre_variable_first, grid_points).reshape(xx.shape)
            post_variable_first = _set_reduced_state(
                problem, pre_reduced_first + correction_reduced_first
            )
            correction_grid_first = (
                _interpolate(post_variable_first, grid_points).reshape(xx.shape)
                - pre_grid_first
            )
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
            correction_grid_first[bmask] = 0.0
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
            records["newton_correction_first"].append(
                correction_grid_first[None].astype(np.float32)
            )
            records["newton_correction_p075"].append(
                correction_grid_p075[None].astype(np.float32)
            )
            records["residual_progress"].append(np.asarray([progress], dtype=np.float32))
            records["residual_progress_first"].append(
                np.asarray([progress_first], dtype=np.float32)
            )
            records["residual_progress_p075"].append(
                np.asarray([progress_p075], dtype=np.float32)
            )
            records["valid_correction_mask"].append(np.asarray([valid], dtype=bool))
            records["valid_correction_mask_first"].append(
                np.asarray([valid_first], dtype=bool)
            )
            records["valid_correction_mask_p075"].append(
                np.asarray([valid_p075], dtype=bool)
            )
            records["selected_newton_state_index"].append(
                np.asarray(selected_index, dtype=np.int16)
            )
            records["selected_newton_state_index_first"].append(
                np.asarray(selected_index_first, dtype=np.int16)
            )
            records["selected_newton_state_index_p075"].append(
                np.asarray(selected_index_p075, dtype=np.int16)
            )
            records["accepted_newton_residuals"].append(padded_residuals)
            records["accepted_newton_step_norms"].append(padded_step_norms)
            records["accepted_newton_counts"].append(
                np.asarray(len(iter_states), dtype=np.int16)
            )
            records["case_id"].append(np.asarray(case["case_id"]))
            records["pair_id"].append(np.asarray(case["pair_id"]))
            records["boundary_id"].append(np.asarray(case["boundary_id"]))
            records["initial_condition_id"].append(np.asarray(case["initial_condition_id"]))
            records["target_task_id"].append(
                np.asarray(case.get("target_task_id", case["boundary_id"]), dtype=np.int64)
            )
            if "support_seed" in case:
                records["support_seed"].append(np.asarray(case["support_seed"], dtype=np.int64))
                records["support_rank"].append(np.asarray(case["support_rank"], dtype=np.int64))
            records["rollout_id"].append(np.asarray(rollout_idx, dtype=np.int64))
            records["local_step_id"].append(np.asarray(local_step, dtype=np.int64))
            records["fidelity"].append(np.asarray(args.fidelity, dtype=np.int8))
            records["solver_converged"].append(np.asarray(converged, dtype=bool))
            records["num_newton_iters"].append(
                np.asarray(max(len(iter_states) - 1, 0), dtype=np.int16)
            )
            records["solver_newton_iterations"].append(
                np.asarray(int(status.get("n_iter", 0)), dtype=np.int16)
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
        "selected_newton_state_index_semantics": (
            "zero_based_raw_hook_trace_q0_index_0_duplicate_initial_hook_entries_may_"
            "occur_first_nonzero_state_is_first_accepted_update_final_may_be_appended; "
            "-1_invalid"
        ),
        "newton_path_selection": {
            "schema": "blrpng_newton_selection_evidence_v1",
            "thresholds": [float(args.progress_threshold), 0.75],
            "first_selection_threshold": 0.0,
            "residual_floor": RESIDUAL_FLOOR,
            "log_denominator_floor": LOG_DENOMINATOR_FLOOR,
            "nonzero_step_norm_threshold": NONZERO_STEP_NORM_THRESHOLD,
            "fallback": "last_nonzero_candidate",
            "q0_semantics": "natural_previous_physical_state",
            "trace_count_includes_q0_and_appended_final": True,
            "trace_may_contain_repeated_q0": True,
            "full_node_states_retained": False,
            "evidence_scope": "selection_replay_not_field_trajectory_reconstruction",
        },
        "convergence_failures": convergence_failures,
        "all_converged": not convergence_failures,
        "sfepy_data_dir": data_dir,
        "boundary_families": sorted(
            {str(case.get("boundary_family", "legacy_fourier")) for case in cases}
        ),
        "worker_sha256": _stdlib_sha256_file(Path(__file__).resolve()),
        "query_gate_sha256": query_gate_sha256,
    }
    if query_gate_sha256 is not None:
        meta["query_solver_protocol"] = QUERY_PROTOCOL
        meta["query_numpy_version"] = np.__version__
        meta["query_runtime_environment"] = query_authorization["query_runtime_environment"]
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
