from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.data.blind_boundaries import FAMILIES, sample_boundary_descriptor
from blrpng.data.confirmation_query_gate import (
    QUERY_CASE_ID_BASE,
    QUERY_IC_SEED_OFFSET,
    QUERY_SOLVER_PROTOCOL,
    WORKER_PATH,
    capture_query_runtime_environment,
    verify_confirmation_query_gate,
)
from blrpng.data.generate import SFEPY_PYTHON, workspace_audit
from blrpng.data.shards import read_index, subset_index, write_index, write_v4_rollouts_from_arrays
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    sha256_json,
)


CAMPAIGN_ROOT = "artifacts/data/phase1_development"
MANIFEST_ROOT = "artifacts/manifests/phase1_development"
CAMPAIGN_KIND = "development"
ROLE_CONFIG = {
    "support": (64, 65, 10, 0),
    "validation_lr": (64, 65, 10, 0),
    "validation_hr": (256, 257, 20, 1),
}


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = ensure_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_reusable(path: Path, payload: dict[str, Any]) -> Path:
    path = ensure_artifact_path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _check_seal(existing) != payload["sha256"]:
            raise FileExistsError(f"Existing manifest differs and will not be overwritten: {path}")
        return path
    _write_exclusive(path, payload)
    return path


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = sha256_json(payload)
    return result


def _check_seal(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    actual = sha256_json(unsigned)
    if payload.get("sha256") != actual:
        raise ValueError("Campaign manifest hash mismatch.")
    return actual


def _verify_confirmation_query_gate(
    campaign: str,
    task_manifest: dict[str, Any],
    *,
    query_rollouts: int,
    query_steps: int,
) -> None:
    runtime_environment = capture_query_runtime_environment()
    verify_confirmation_query_gate(
        campaign,
        task_manifest,
        query_rollouts=query_rollouts,
        query_steps=query_steps,
        query_numpy_version=np.__version__,
        query_runtime_environment=runtime_environment,
    )


def freeze_tasks(campaign: str, seed: int, tasks_per_family: int, boundary_id_offset: int) -> Path:
    if not campaign.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Campaign IDs may contain only letters, numbers, '_' and '-'.")
    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be positive.")
    tasks = []
    boundary_id = int(boundary_id_offset)
    for family in FAMILIES:
        for family_task_id in range(tasks_per_family):
            descriptor = sample_boundary_descriptor(
                family, boundary_id=boundary_id, seed=int(seed)
            )
            tasks.append({
                "target_task_id": len(tasks),
                "family": family,
                "family_task_id": family_task_id,
                "boundary_id": boundary_id,
                "boundary": descriptor,
            })
            boundary_id += 1
    payload = _seal({
        "schema": "blrpng_phase1_development_tasks_v1",
        "campaign_id": campaign,
        "seed": int(seed),
        "families": list(FAMILIES),
        "tasks_per_family": int(tasks_per_family),
        "tasks": tasks,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query_role": "validation_hr",
        "note": "Development-only boundary tasks; never reuse these tasks as confirmation tasks.",
    })
    output = PROJECT_ROOT / MANIFEST_ROOT / campaign / "tasks.json"
    _write_exclusive(output, payload)
    return output


def make_case_manifest(
    task_manifest_path: Path,
    *,
    role: str,
    campaign: str,
    support_seed: int,
    count: int,
    ic_seed: int,
) -> Path:
    if role not in ROLE_CONFIG:
        raise ValueError(f"Unsupported development role: {role}")
    tasks_payload = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    task_hash = _check_seal(tasks_payload)
    tasks = tasks_payload["tasks"]
    role_base = {
        "support": 5_000_000,
        "validation_lr": 6_000_000,
        "validation_hr": QUERY_CASE_ID_BASE,
    }[role]
    cases = []
    for task in tasks:
        task_id = int(task["target_task_id"])
        boundary_id = int(task["boundary_id"])
        draw_id = support_seed if role == "support" else 0
        rng = np.random.default_rng(np.random.SeedSequence([ic_seed, draw_id, task_id]))
        for rank in range(count):
            case_id = role_base + task_id * 1000 + rank
            if role == "support":
                case_id += int(support_seed) * 100_000
            case = {
                "case_id": case_id,
                "pair_id": case_id,
                "boundary_id": boundary_id,
                "initial_condition_id": case_id,
                "split": role,
                "boundary_coefficients": [0.0] * 7,
                "boundary_frequencies": [1.0, 2.0],
                "boundary_amplitude": float(task["boundary"]["amplitude"]),
                "boundary_descriptor": task["boundary"],
                "initial_coefficients": rng.normal(size=4).tolist(),
                "initial_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "target_task_id": task_id,
                "boundary_family": task["family"],
            }
            if role == "support":
                case["support_seed"] = int(support_seed)
                case["support_rank"] = rank
            cases.append(case)
    payload = _seal({
        "schema": "blrpng_manifest_v1",
        "metadata": {
            "experiment": f"phase1_{CAMPAIGN_KIND}_{campaign}",
            "campaign_id": campaign,
            "role": role,
            "evaluation_role": (
                "query" if CAMPAIGN_KIND == "confirmation" and role == "validation_hr" else role
            ),
            "support_seed": int(support_seed) if role == "support" else None,
            "initial_conditions_per_task": int(count),
            "initial_condition_seed": int(ic_seed),
            "task_manifest": str(task_manifest_path.relative_to(PROJECT_ROOT)),
            "task_manifest_sha256": task_hash,
            "contains_solutions": False,
        },
        "cases": cases,
    })
    support_label = f"_s{support_seed}" if role == "support" else ""
    output = PROJECT_ROOT / MANIFEST_ROOT / campaign / f"{role}{support_label}.json"
    return _write_reusable(output, payload)


def _generate_role(
    manifest_path: Path,
    *,
    campaign: str,
    role: str,
    support_seed: int,
    steps_override: int | None = None,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _check_seal(manifest)
    cases = manifest["cases"]
    if not cases:
        raise ValueError("Cannot generate an empty role manifest.")
    resolution, mesh, default_steps, fidelity = ROLE_CONFIG[role]
    steps = int(default_steps if steps_override is None else steps_override)
    if steps < 1:
        raise ValueError("Rollout steps must be positive.")
    support_label = f"_s{support_seed}" if role == "support" else ""
    stem = f"phase1_{CAMPAIGN_KIND}_{campaign}_{role}{support_label}_{resolution}"
    index_path = PROJECT_ROOT / CAMPAIGN_ROOT / campaign / f"{role}{support_label}_{resolution}.index.json"
    if index_path.exists():
        index = read_index(index_path, verify_hashes=True)
        existing_meta = index["metadata"]
        expected_meta = {
            "manifest_sha256": manifest_hash,
            "split": role,
            "steps": steps,
            "resolution": resolution,
            "fidelity": fidelity,
        }
        if role == "support":
            expected_meta["support_seed"] = support_seed
        is_confirmation_query = CAMPAIGN_KIND == "confirmation" and role == "validation_hr"
        if is_confirmation_query:
            expected_meta.update({
                "worker_sha256": sha256_file(PROJECT_ROOT / WORKER_PATH),
                "query_gate_sha256": sha256_file(
                    PROJECT_ROOT / MANIFEST_ROOT / campaign / "query_gate.json"
                ),
                "query_solver_protocol": QUERY_SOLVER_PROTOCOL,
                "query_numpy_version": np.__version__,
                "query_runtime_environment": capture_query_runtime_environment(),
                "query_ic_seed": int(manifest["metadata"]["initial_condition_seed"]),
            })
        mismatches = {
            key: (existing_meta.get(key), value)
            for key, value in expected_meta.items()
            if existing_meta.get(key) != value
        }
        if mismatches:
            raise FileExistsError(
                f"Existing development index does not match the requested generation: {mismatches}"
            )
        return index_path

    npz = ensure_artifact_path(f"artifacts/tmp/{stem}.npz")
    meta_path = ensure_artifact_path(f"artifacts/tmp/{stem}.json")
    log_path = ensure_artifact_path(f"artifacts/experiments/phase1/{stem}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(PROJECT_ROOT),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    command = [
        str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_worker",
        "--manifest", str(manifest_path), "--split", role,
        "--output-npz", str(npz), "--output-meta", str(meta_path),
        "--output-resolution", str(resolution), "--mesh-resolution", str(mesh),
        "--steps", str(steps), "--fidelity", str(fidelity),
        "--dt", "0.2", "--epsilon", "0.03", "--gamma", "5.0",
        "--progress-threshold", "0.25", "--case-start", "0", "--limit", str(len(cases)),
    ]
    is_confirmation_query = CAMPAIGN_KIND == "confirmation" and role == "validation_hr"
    expected_query_runtime_environment = (
        capture_query_runtime_environment() if is_confirmation_query else None
    )
    with workspace_audit(f"phase1_{CAMPAIGN_KIND}_generate_{role}") as audit:
        audit.update({
            "campaign_id": campaign,
            "role": role,
            "support_seed": support_seed if role == "support" else None,
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": manifest_hash,
            "case_count": len(cases),
            "command": command,
            "query_used": False,
            "query_data_generated": is_confirmation_query,
        })
        with log_path.open("w", encoding="utf-8") as log_stream:
            result = subprocess.run(
                command, cwd=PROJECT_ROOT, env=environment, stdout=log_stream,
                stderr=subprocess.STDOUT, text=True,
            )
        if result.returncode:
            raise RuntimeError(
                f"SfePy worker failed ({result.returncode}); see {log_path}"
            )
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
        worker_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if is_confirmation_query and (
            worker_meta.get("worker_sha256") != sha256_file(PROJECT_ROOT / WORKER_PATH)
            or worker_meta.get("query_gate_sha256")
            != sha256_file(PROJECT_ROOT / MANIFEST_ROOT / campaign / "query_gate.json")
            or worker_meta.get("query_solver_protocol") != QUERY_SOLVER_PROTOCOL
            or worker_meta.get("query_numpy_version") != np.__version__
            or worker_meta.get("query_runtime_environment") != expected_query_runtime_environment
            or int(worker_meta.get("steps", -1)) != QUERY_SOLVER_PROTOCOL["steps"]
            or int(worker_meta.get("output_resolution", -1))
            != QUERY_SOLVER_PROTOCOL["output_resolution"]
            or int(worker_meta.get("mesh_resolution", -1))
            != QUERY_SOLVER_PROTOCOL["mesh_resolution"]
            or int(worker_meta.get("fidelity", -1)) != QUERY_SOLVER_PROTOCOL["fidelity"]
        ):
            raise ValueError("Confirmation worker output differs from the frozen query protocol.")
        if not bool(worker_meta.get("all_converged", False)):
            raise RuntimeError(
                f"Refusing to package non-converged {role} solve output; see {log_path}"
            )
        if "solver_converged" not in arrays or not bool(arrays["solver_converged"].all()):
            raise RuntimeError(
                f"Refusing to package a role containing non-converged rollouts; see {log_path}"
            )
        boundary_to_task = {
            int(case["boundary_id"]): int(case["target_task_id"])
            for case in cases
        }
        rows = write_v4_rollouts_from_arrays(
            arrays,
            worker_meta,
            shard_dir=PROJECT_ROOT / CAMPAIGN_ROOT / campaign / f"{role}{support_label}_{resolution}",
            target_task_by_boundary=boundary_to_task,
            role=role,
            source_sha256=manifest_hash,
        )
        metadata = {
            "experiment": f"phase1_{CAMPAIGN_KIND}_{campaign}",
            "campaign_id": campaign,
            "split": role,
            "support_seed": support_seed if role == "support" else None,
            "resolution": resolution,
            "steps": steps,
            "dt": 0.2,
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": manifest_hash,
            "task_manifest_sha256": manifest["metadata"]["task_manifest_sha256"],
            "generation_workers": 1,
            "fidelity": fidelity,
        }
        if is_confirmation_query:
            metadata.update({
                "worker_sha256": worker_meta["worker_sha256"],
                "query_gate_sha256": worker_meta["query_gate_sha256"],
                "query_solver_protocol": QUERY_SOLVER_PROTOCOL,
                "query_numpy_version": np.__version__,
                "query_runtime_environment": worker_meta["query_runtime_environment"],
                "query_ic_seed": int(manifest["metadata"]["initial_condition_seed"]),
            })
        write_index(index_path, rows, metadata=metadata)
        if "newton_path_selection" in worker_meta:
            replay_checks = 0
            selected = {}
            first_selected = {}
            for row in rows:
                payload = torch.load(PROJECT_ROOT / row["path"], map_location="cpu", weights_only=False)
                from blrpng.data.shards import validate_shard
                validation = validate_shard(payload)
                replay_checks += int(validation.get("newton_selection_replay_pass_count", 0))
                for value in payload["selected_newton_state_index"].tolist():
                    selected[str(int(value))] = selected.get(str(int(value)), 0) + 1
                for value in payload.get("selected_newton_state_index_first", torch.empty(0)).tolist():
                    first_selected[str(int(value))] = first_selected.get(str(int(value)), 0) + 1
            from blrpng.paths import write_json
            write_json(index_path.with_suffix(".generation_audit.json"), {
                "schema": "phase1_generation_selection_replay_audit_v1",
                "query_read": False, "campaign": campaign, "role": role,
                "manifest_sha256": manifest_hash, "index": str(index_path.relative_to(PROJECT_ROOT)),
                "index_sha256": sha256_file(index_path), "shards": rows,
                "transitions": int(arrays["y"].shape[0]),
                "solver_rollouts": int(worker_meta["num_rollouts"]),
                "all_solvers_converged": bool(arrays["solver_converged"].all()),
                "selection_replay_checks": replay_checks,
                "selected_index_histogram": selected, "first_index_histogram": first_selected,
                "selection_metadata": worker_meta["newton_path_selection"],
                "worker_sha256": sha256_file(PROJECT_ROOT / "blrpng/solvers/sfepy_worker.py"),
                "selector_sha256": sha256_file(PROJECT_ROOT / "blrpng/solvers/progress.py"),
                "shards_module_sha256": sha256_file(PROJECT_ROOT / "blrpng/data/shards.py"),
                "raw_npz_sha256": sha256_file(npz), "raw_meta_sha256": sha256_file(meta_path),
                "limitation": "Selection replay evidence; full node-state histories are not retained.",
            })
    npz.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return index_path


def _make_task_indices(
    index_path: Path, *, campaign: str, role: str, support_seed: int, task_count: int
) -> None:
    if role == "support":
        budgets = (1, 2, 4, 8, 16)
        for task in range(task_count):
            for budget in budgets:
                output = PROJECT_ROOT / CAMPAIGN_ROOT / campaign / (
                    f"task{task:02d}_support_s{support_seed}_m{budget}.index.json"
                )
                if output.exists():
                    existing = read_index(output, verify_hashes=True)
                    expected_count = min(budget, sum(
                        int(row["target_task_id"]) == task for row in read_index(index_path)["shards"]
                    ))
                    if (
                        existing["metadata"].get("parent_sha256") != read_index(index_path)["sha256"]
                        or existing["metadata"].get("target_task_id") != task
                        or len(existing["shards"]) != expected_count
                    ):
                        raise FileExistsError(f"Existing support subset differs: {output}")
                    continue
                subset_index(
                    index_path, output, target_task_id=task, max_rollouts=budget
                )
    else:
        resolution = ROLE_CONFIG[role][0]
        for task in range(task_count):
            output = PROJECT_ROOT / CAMPAIGN_ROOT / campaign / (
                f"task{task:02d}_{role}_{resolution}.index.json"
            )
            if output.exists():
                existing = read_index(output, verify_hashes=True)
                parent = read_index(index_path)
                expected_count = sum(
                    int(row["target_task_id"]) == task for row in parent["shards"]
                )
                if (
                    existing["metadata"].get("parent_sha256") != parent["sha256"]
                    or existing["metadata"].get("target_task_id") != task
                    or len(existing["shards"]) != expected_count
                ):
                    raise FileExistsError(f"Existing validation subset differs: {output}")
                continue
            subset_index(index_path, output, target_task_id=task)


def command_freeze(args: argparse.Namespace) -> None:
    path = freeze_tasks(args.campaign, args.seed, args.tasks_per_family, args.boundary_id_offset)
    print(json.dumps({"task_manifest": str(path), "tasks_per_family": args.tasks_per_family}, indent=2))


def command_generate(args: argparse.Namespace) -> None:
    global CAMPAIGN_KIND, CAMPAIGN_ROOT, MANIFEST_ROOT
    CAMPAIGN_KIND = args.campaign_type
    if CAMPAIGN_KIND == "confirmation":
        CAMPAIGN_ROOT = "artifacts/data/phase1_confirmation"
        MANIFEST_ROOT = "artifacts/manifests/phase1_confirmation"
    else:
        CAMPAIGN_ROOT = "artifacts/data/phase1_development"
        MANIFEST_ROOT = "artifacts/manifests/phase1_development"
    campaign_root = PROJECT_ROOT / MANIFEST_ROOT / args.campaign
    task_manifest = (
        (PROJECT_ROOT / args.task_manifest).resolve()
        if args.task_manifest
        else campaign_root / "tasks.json"
    )
    payload = json.loads(task_manifest.read_text(encoding="utf-8"))
    _check_seal(payload)
    if payload.get("campaign_id") != args.campaign:
        raise ValueError("Requested campaign does not match the frozen task manifest.")
    confirmation_schema = "blrpng_phase1_confirmation_tasks_v1"
    development_schema = "blrpng_phase1_development_tasks_v1"
    if args.campaign_type == "confirmation":
        expected_manifest = campaign_root / "tasks.json"
        if task_manifest != expected_manifest.resolve():
            raise ValueError("Confirmation generation must use its sealed campaign task manifest.")
        if payload.get("schema") != confirmation_schema:
            raise ValueError("Confirmation generation requires a confirmation task manifest.")
    elif payload.get("schema") == confirmation_schema:
        raise ValueError(
            "A confirmation task manifest cannot be generated under the development campaign root."
        )
    elif payload.get("schema") != development_schema:
        raise ValueError("Development generation requires a development task manifest.")
    if args.campaign_type == "confirmation" and args.roles is None:
        raise ValueError(
            "Confirmation data generation requires explicit roles. Generate LR roles first; "
            "generate validation_hr only after confirmation checkpoints are frozen."
        )
    roles = list(args.roles or ROLE_CONFIG)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("Choose one or more distinct roles to generate.")
    if (
        args.campaign_type == "confirmation"
        and "validation_hr" in roles
    ):
        if roles != ["validation_hr"]:
            raise ValueError("Generate the final confirmation HR query role by itself after matrix freeze.")
        _verify_confirmation_query_gate(
            args.campaign,
            payload,
            query_rollouts=args.hr_validation_rollouts,
            query_steps=args.hr_steps,
        )
    elif args.campaign_type == "confirmation":
        if "support" in roles and args.support_seed not in {271829, 271830, 271831}:
            raise ValueError("Confirmation support seed is outside the preregistered set.")
        if "support" in roles and args.support_rollouts != 16:
            raise ValueError("Confirmation support generation is frozen at 16 rollouts per task.")
        if "validation_lr" in roles and args.lr_validation_rollouts != 8:
            raise ValueError("Confirmation LR validation is frozen at 8 rollouts per task.")
        if ("support" in roles or "validation_lr" in roles) and args.lr_steps != 20:
            raise ValueError("Confirmation LR roles are frozen at 20 rollout steps.")
    role_counts = {
        "support": args.support_rollouts,
        "validation_lr": args.lr_validation_rollouts,
        "validation_hr": args.hr_validation_rollouts,
    }
    role_ic_seeds = {
        "support": args.support_seed + 101,
        "validation_lr": int(payload["seed"]) + 20_000,
        "validation_hr": int(payload["seed"]) + QUERY_IC_SEED_OFFSET,
    }
    for role in roles:
        path = make_case_manifest(
            task_manifest, role=role, campaign=args.campaign,
            support_seed=args.support_seed, count=role_counts[role],
            ic_seed=role_ic_seeds[role],
        )
        index_path = _generate_role(
            path,
            campaign=args.campaign,
            role=role,
            support_seed=args.support_seed,
            steps_override=args.hr_steps if role == "validation_hr" else args.lr_steps,
        )
        _make_task_indices(
            index_path,
            campaign=args.campaign,
            role=role,
            support_seed=args.support_seed,
            task_count=len(payload["tasks"]),
        )
    print(json.dumps({
        "campaign_id": args.campaign,
        "task_manifest_sha256": payload["sha256"],
        "tasks": len(payload["tasks"]),
        "support_seed": args.support_seed,
        "support_rollouts_per_task": args.support_rollouts,
        "lr_validation_rollouts_per_task": args.lr_validation_rollouts,
        "hr_validation_rollouts_per_task": args.hr_validation_rollouts,
        "roles_generated": roles,
        "lr_steps": args.lr_steps,
        "hr_steps": args.hr_steps,
        "task_manifest": str(task_manifest.relative_to(PROJECT_ROOT)),
        "query_generated": args.campaign_type == "confirmation" and "validation_hr" in roles,
        "hr_role_semantics": "final_confirmation_query" if args.campaign_type == "confirmation" else "development_validation",
        "data_root": str(PROJECT_ROOT / CAMPAIGN_ROOT / args.campaign),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze development tasks or generate explicitly gated Phase-1 confirmation roles."
    )
    subparsers = parser.add_subparsers(required=True)
    freeze = subparsers.add_parser("freeze", help="Freeze new development boundary tasks.")
    freeze.add_argument("--campaign", required=True)
    freeze.add_argument("--seed", type=int, default=20261018)
    freeze.add_argument("--tasks-per-family", type=int, default=1)
    freeze.add_argument("--boundary-id-offset", type=int, default=5_000_000)
    freeze.set_defaults(func=command_freeze)
    generate = subparsers.add_parser("generate", help="Generate development data or explicitly gated confirmation roles.")
    generate.add_argument("--campaign", required=True)
    generate.add_argument("--campaign-type", choices=("development", "confirmation"), default="development")
    generate.add_argument("--support-seed", type=int, default=271828)
    generate.add_argument("--support-rollouts", type=int, default=16)
    generate.add_argument("--lr-validation-rollouts", type=int, default=8)
    generate.add_argument("--hr-validation-rollouts", type=int, default=8)
    generate.add_argument(
        "--task-manifest",
        help="Use an already-frozen task manifest, including when building a new rollout campaign.",
    )
    generate.add_argument(
        "--roles", nargs="+", choices=tuple(ROLE_CONFIG),
        help="Roles to generate; defaults to all three roles.",
    )
    generate.add_argument("--lr-steps", type=int, default=10)
    generate.add_argument("--hr-steps", type=int, default=20)
    generate.set_defaults(func=command_generate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
