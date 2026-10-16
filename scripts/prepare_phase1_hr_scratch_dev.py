from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from blrpng.data.generate import SFEPY_PYTHON, workspace_audit
from blrpng.data.shards import read_index, subset_index, write_index, write_v4_rollouts_from_arrays
from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json, write_json


CAMPAIGN = "hrdev_20261015_hrsupport_s271829"
TASK_MANIFEST = Path("artifacts/manifests/phase1_development/hrdev_20261012/tasks.json")
SUPPORT_MANIFEST = Path(
    "artifacts/manifests/phase1_development/hrdev_20261012_t20_ss271829/support_s271829.json"
)
SUPPORT_PARENT_INDEX = Path(
    "artifacts/data/phase1_development/hrdev_20261013_firstpath_ss271829/"
    "support_s271829_64.index.json"
)
HR_VALIDATION_MANIFEST = Path(
    "artifacts/manifests/phase1_development/hrdev_20261012/validation_hr.json"
)
TASKS = (0, 1, 3, 5)
SUPPORT_SEED = 271829
SUPPORT_MANIFEST_SHA256 = "dfbc58d5b62b870b60804df9b33fad8a27cc5370c6070aee679028ddaf098138"
SUPPORT_PARENT_INDEX_SHA256 = "c5bea773b1f414c2d2deaa9bee9e35343a38b6d7c4268a86505a6585113499f1"
TASK_SHA256 = "bb50c7e4d5b8d0900e21d7ae64235c6d955244cc3bfdca1a97ab6056596a334a"
HR_VALIDATION_MANIFEST_SHA256 = "d491452f11d9c75b62c385fd9ecf2df40b960053782e0b2c31a63a819cd21ac5"
M4_INDEX_SHA256 = {
    0: "cb66e634789b44cc11be7d58bd99bb442e60ffacd6362e1db0dd1c650974ef85",
    1: "2f90cc5632c5c1d6df6a798d2db88a9e094f6e52709dcaef9b4bfe513c4753e6",
    3: "efef59e67144b17f49ae7fda8784219c5ae0e4c879a9d047bdfb99ec86be576f",
    5: "3a8a35325729f3d8af76b5b6fa07c82a0fe8ae16c509f138ca86002e06f85bf2",
}
MANIFEST_OUTPUT = Path(
    f"artifacts/manifests/phase1_development/{CAMPAIGN}/support_hr_s{SUPPORT_SEED}.json"
)
DATA_ROOT = Path(f"artifacts/data/phase1_development/{CAMPAIGN}")
INDEX_OUTPUT = DATA_ROOT / f"support_hr_s{SUPPORT_SEED}_256.index.json"
GENERATION_REPORT = Path(
    f"artifacts/reports/phase1_hr_scratch_data_generation_{CAMPAIGN}.json"
)
WORKER_EXECUTION_SHA256 = "21abb108bb5fe8eab9c36b2f6823b34c0a4c1276b2a062c8fc44ae2eec95fec1"
WORKER_COMPLETION_SHA256 = "566aa1f373e971004c4ccf895e864bb9e6f776295e44fbffbd0acf3177a8e3b8"
WORKER_SOURCE_SNAPSHOT = Path(
    "artifacts/reports/phase1_hr_scratch_sfepy_worker_at_generation_start.py"
)


def _load_sealed(path: Path, expected_sha: str) -> dict[str, Any]:
    payload = json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    actual = payload.get("sha256")
    if actual != sha256_json({key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError(f"Manifest or index seal is invalid: {path}")
    if actual != expected_sha:
        raise ValueError(f"Frozen artifact hash changed for {path}: {actual}")
    return payload


def _make_manifest() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    tasks = _load_sealed(TASK_MANIFEST, TASK_SHA256)
    support = _load_sealed(SUPPORT_MANIFEST, SUPPORT_MANIFEST_SHA256)
    hr_validation = _load_sealed(HR_VALIDATION_MANIFEST, HR_VALIDATION_MANIFEST_SHA256)
    parent_index = read_index(PROJECT_ROOT / SUPPORT_PARENT_INDEX, verify_hashes=True)
    if parent_index["sha256"] != SUPPORT_PARENT_INDEX_SHA256:
        raise ValueError("The Stage-2 LR support parent index changed.")
    if (
        support.get("metadata", {}).get("task_manifest_sha256") != TASK_SHA256
        or int(support.get("metadata", {}).get("support_seed", -1)) != SUPPORT_SEED
        or support.get("metadata", {}).get("role") != "support"
    ):
        raise ValueError("The frozen Stage-2 support manifest metadata changed.")

    task_lookup = {int(row["target_task_id"]): row for row in tasks["tasks"]}
    source_cases = {
        (int(case["target_task_id"]), int(case["support_rank"])): case
        for case in support["cases"]
    }
    parent_case_ids = {int(row["initial_condition_id"]) for row in parent_index["shards"]}
    source_m4_indices: list[dict[str, Any]] = []
    expected_cases = []
    for task in TASKS:
        rel = Path(
            f"artifacts/data/phase1_development/hrdev_20261013_firstpath_ss271829/"
            f"task{task:02d}_support_s{SUPPORT_SEED}_m4.index.json"
        )
        index = read_index(PROJECT_ROOT / rel, verify_hashes=True)
        if index["sha256"] != M4_INDEX_SHA256[task]:
            raise ValueError(f"Frozen m=4 support index changed: {rel}")
        rows = sorted(index["shards"], key=lambda row: int(row.get("support_rank", -1)))
        if [int(row.get("support_rank", -1)) for row in rows] != [0, 1, 2, 3]:
            raise ValueError(f"Frozen support index is not the first four support ranks: {rel}")
        if any(
            int(row["initial_condition_id"]) not in parent_case_ids
            or int(row["target_task_id"]) != task
            for row in rows
        ):
            raise ValueError(f"Support index rows are not from the Stage-2 parent: {rel}")
        source_m4_indices.append({
            "task": task,
            "path": str(rel),
            "sha256": index["sha256"],
        })
        expected_cases.extend(source_cases[(task, rank)] for rank in range(4))

    validation_ids = {
        int(case["initial_condition_id"]) for case in hr_validation["cases"]
    }
    support_ids = {int(case["initial_condition_id"]) for case in expected_cases}
    if support_ids & validation_ids:
        raise ValueError("HR support and HR-dev validation initial conditions overlap.")
    if len(expected_cases) != 16 or len(support_ids) != 16:
        raise ValueError("Expected 16 unique HR support cases across four development tasks.")

    task_manifest_rel = str(TASK_MANIFEST)
    cases = []
    for source_case in expected_cases:
        task = task_lookup[int(source_case["target_task_id"])]
        if (
            int(source_case["boundary_id"]) != int(task["boundary_id"])
            or source_case["boundary_descriptor"] != task["boundary"]
            or source_case["boundary_family"] != task["family"]
        ):
            raise ValueError("Support case does not match its frozen task boundary.")
        case = dict(source_case)
        case["split"] = "support_hr"
        cases.append(case)

    metadata = {
        "experiment": f"phase1_development_{CAMPAIGN}",
        "campaign_id": CAMPAIGN,
        "role": "support_hr",
        "evaluation_role": "support_hr_dev",
        "support_seed": SUPPORT_SEED,
        "initial_conditions_per_task": 4,
        "initial_condition_seed": int(support["metadata"]["initial_condition_seed"]),
        "task_manifest": task_manifest_rel,
        "task_manifest_sha256": tasks["sha256"],
        "source_support_manifest": str(SUPPORT_MANIFEST),
        "source_support_manifest_sha256": support["sha256"],
        "source_support_index": str(SUPPORT_PARENT_INDEX),
        "source_support_index_sha256": parent_index["sha256"],
        "source_m4_indices": source_m4_indices,
        "contains_solutions": False,
        "note": "Exact Stage-2 m=4 support ICs relabelled for HR support generation; HR-dev validation IDs are disjoint.",
    }
    payload = {
        "schema": "blrpng_manifest_v1",
        "metadata": metadata,
        "cases": cases,
    }
    payload["sha256"] = sha256_json(payload)
    output = PROJECT_ROOT / MANIFEST_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Existing HR support manifest differs: {output}")
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, payload, parent_index


def _freeze_worker_source_at_generation_start() -> tuple[Path, str]:
    worker_path = PROJECT_ROOT / "blrpng/solvers/sfepy_worker.py"
    current = worker_path.read_text(encoding="utf-8")
    completed_comment = (
        "# Pin the frozen development task/validation manifests. A self-seal alone is\n"
        "# not an authorization boundary because a caller could rewrite both the tasks\n"
        "# and seal; copied confirmation traces must not enter a development-HR route.\n"
    )
    original_comment = (
        "# Phase 1 has one frozen development HR case manifest. Its self-seal alone is\n"
        "# not an authorization boundary because a caller could rewrite both the tasks\n"
        "# and seal; pin both hashes so a relabelled confirmation trace cannot enter the\n"
        "# development-HR route.\n"
    )
    if current.count(completed_comment) != 1:
        raise ValueError("The only worker edit since label generation is no longer the audited comment.")
    execution_source = current.replace(completed_comment, original_comment)
    execution_sha = hashlib.sha256(execution_source.encode("utf-8")).hexdigest()
    if execution_sha != WORKER_EXECUTION_SHA256:
        raise ValueError("Cannot reconstruct the exact worker source that launched the solver.")
    snapshot_path = PROJECT_ROOT / WORKER_SOURCE_SNAPSHOT
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        if snapshot_path.read_text(encoding="utf-8") != execution_source:
            raise FileExistsError("Existing solver source snapshot differs; preserving it.")
    else:
        snapshot_path.write_text(execution_source, encoding="utf-8")
    if sha256_file(snapshot_path) != WORKER_EXECUTION_SHA256:
        raise OSError("Worker execution-source snapshot failed its SHA256 read-back check.")
    if sha256_file(worker_path) != WORKER_COMPLETION_SHA256:
        raise ValueError("Worker source changed after the HR support solver finished.")
    return snapshot_path, execution_sha


def _original_generation_audit(manifest_sha256: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted(
        (PROJECT_ROOT / "artifacts/reports/workspace_audits").glob(
            f"phase1_{CAMPAIGN}_support_hr_generation_*.json"
        )
    )
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one preserved original HR support generation audit, got {len(matches)}")
    path = matches[0]
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (
        audit.get("manifest_sha256") != manifest_sha256
        or audit.get("case_count") != 16
        or audit.get("steps") != 20
        or audit.get("resolution") != 256
        or audit.get("fidelity") != 1
        or audit.get("query_used") is not False
        or audit.get("query_data_generated") is not False
    ):
        raise ValueError("Preserved HR generation attempt audit failed its input/protocol binding.")
    return path, audit


def _generate(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not SFEPY_PYTHON.exists():
        raise FileNotFoundError(f"SfePy worker interpreter not found: {SFEPY_PYTHON}")
    npz = PROJECT_ROOT / f"artifacts/tmp/phase1_{CAMPAIGN}_support_hr_256.npz"
    worker_meta_path = PROJECT_ROOT / f"artifacts/tmp/phase1_{CAMPAIGN}_support_hr_256.json"
    log_path = PROJECT_ROOT / f"artifacts/experiments/phase1/phase1_{CAMPAIGN}_support_hr_256.log"
    raw_presence = (npz.exists(), worker_meta_path.exists())
    if raw_presence[0] != raw_presence[1]:
        raise FileExistsError("Only one solver temporary output exists; preserving the incomplete pair.")
    reused_raw_outputs = all(raw_presence)
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
        "--manifest", str(manifest_path), "--split", "support_hr",
        "--output-npz", str(npz), "--output-meta", str(worker_meta_path),
        "--output-resolution", "256", "--mesh-resolution", "257",
        "--steps", "20", "--fidelity", "1", "--dt", "0.2",
        "--epsilon", "0.03", "--gamma", "5.0", "--progress-threshold", "0.25",
        "--case-start", "0", "--limit", "16",
    ]
    prior_audit_path = None
    prior_audit = None
    if reused_raw_outputs:
        prior_audit_path, prior_audit = _original_generation_audit(manifest["sha256"])
    worker_snapshot_path, execution_worker_sha256 = _freeze_worker_source_at_generation_start()
    current_attempt_wall_seconds = 0.0
    packaging_wall_seconds = 0.0
    audit_name = (
        f"phase1_{CAMPAIGN}_support_hr_packaging_recovery"
        if reused_raw_outputs else f"phase1_{CAMPAIGN}_support_hr_generation"
    )
    with workspace_audit(audit_name) as audit:
        audit["started_monotonic"] = time.perf_counter()
        audit.update({
            "campaign_id": CAMPAIGN,
            "role": "support_hr",
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": manifest["sha256"],
            "source_support_index_sha256": manifest["metadata"]["source_support_index_sha256"],
            "case_count": len(manifest["cases"]),
            "tasks": list(TASKS),
            "steps": 20,
            "resolution": 256,
            "fidelity": 1,
            "command": command,
            "query_used": False,
            "query_data_generated": False,
            "reused_solver_outputs": reused_raw_outputs,
        })
        if reused_raw_outputs:
            audit["original_generation_audit"] = str(prior_audit_path.relative_to(PROJECT_ROOT))
            audit["original_generation_audit_sha256"] = sha256_file(prior_audit_path)
            audit["recovery_reason"] = "The original post-solve wrapper incorrectly required fidelity in worker JSON metadata; fidelity is recorded per transition in the NPZ."
        else:
            worker_started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log_stream:
                result = subprocess.run(
                    command, cwd=PROJECT_ROOT, env=environment,
                    stdout=log_stream, stderr=subprocess.STDOUT, text=True,
                )
            current_attempt_wall_seconds = time.perf_counter() - worker_started
            if result.returncode:
                raise RuntimeError(f"SfePy worker failed ({result.returncode}); see {log_path}")

        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
        worker_meta = json.loads(worker_meta_path.read_text(encoding="utf-8"))
        if (
            worker_meta.get("manifest_sha256") != manifest["sha256"]
            or int(worker_meta.get("num_rollouts", -1)) != 16
            or int(worker_meta.get("num_samples", -1)) != 320
            or int(worker_meta.get("output_resolution", -1)) != 256
            or int(worker_meta.get("mesh_resolution", -1)) != 257
            or int(worker_meta.get("steps", -1)) != 20
            or int(worker_meta.get("case_start", -1)) != 0
            or worker_meta.get("split") != "support_hr"
            or float(worker_meta.get("dt", float("nan"))) != 0.2
            or float(worker_meta.get("epsilon", float("nan"))) != 0.03
            or float(worker_meta.get("gamma", float("nan"))) != 5.0
            or float(worker_meta.get("progress_threshold", float("nan"))) != 0.25
            or worker_meta.get("query_gate_sha256") is not None
            or worker_meta.get("worker_sha256") != WORKER_COMPLETION_SHA256
            or not bool(worker_meta.get("all_converged", False))
            or not bool(arrays["solver_converged"].all())
            or not bool((arrays["fidelity"] == 1).all())
            or tuple(arrays["solver_converged"].shape) != (320,)
        ):
            raise RuntimeError("HR support worker output failed protocol or convergence checks.")
        cases = manifest["cases"]
        for rollout, case in enumerate(cases):
            start, end = rollout * 20, (rollout + 1) * 20
            expected_scalars = {
                "case_id": int(case["case_id"]),
                "pair_id": int(case["pair_id"]),
                "boundary_id": int(case["boundary_id"]),
                "initial_condition_id": int(case["initial_condition_id"]),
                "target_task_id": int(case["target_task_id"]),
                "support_seed": SUPPORT_SEED,
                "support_rank": int(case["support_rank"]),
                "rollout_id": rollout,
                "fidelity": 1,
            }
            if not np.array_equal(arrays["local_step_id"][start:end].numpy(), np.arange(20)):
                raise RuntimeError(f"Solver step order failed verification for rollout {rollout}.")
            for key, expected in expected_scalars.items():
                values = arrays[key][start:end].numpy()
                if not np.all(values == expected):
                    raise RuntimeError(f"Solver output {key} does not match frozen rollout {rollout}.")
        for key in ("x", "y", "boundary_values", "boundary_lift", "newton_correction"):
            if not bool(torch.isfinite(arrays[key]).all()):
                raise RuntimeError(f"Solver output contains non-finite values in {key}.")
        worker_snapshot_path, execution_worker_sha256 = _freeze_worker_source_at_generation_start()
        task_by_boundary = {
            int(case["boundary_id"]): int(case["target_task_id"])
            for case in manifest["cases"]
        }
        index_meta = {
            "experiment": f"phase1_development_{CAMPAIGN}",
            "campaign_id": CAMPAIGN,
            "split": "support_hr",
            "support_seed": SUPPORT_SEED,
            "resolution": 256,
            "steps": 20,
            "dt": 0.2,
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": manifest["sha256"],
            "task_manifest_sha256": manifest["metadata"]["task_manifest_sha256"],
            "source_support_manifest_sha256": manifest["metadata"]["source_support_manifest_sha256"],
            "source_support_index_sha256": manifest["metadata"]["source_support_index_sha256"],
            "source_m4_indices": manifest["metadata"]["source_m4_indices"],
            "generation_workers": 1,
            "fidelity": 1,
            "worker_sha256": execution_worker_sha256,
            "worker_sha256_reported_at_completion": worker_meta["worker_sha256"],
            "worker_source_snapshot": str(worker_snapshot_path.relative_to(PROJECT_ROOT)),
            "worker_source_snapshot_sha256": execution_worker_sha256,
            "query_used": False,
        }
        if INDEX_OUTPUT.exists():
            existing_index = read_index(INDEX_OUTPUT, verify_hashes=True)
            if existing_index.get("metadata") != index_meta or len(existing_index["shards"]) != 16:
                raise FileExistsError(f"Existing HR support index differs: {INDEX_OUTPUT}")
            rows = existing_index["shards"]
            for rollout, case in enumerate(cases):
                row = rows[rollout]
                if (
                    int(row["target_task_id"]) != int(case["target_task_id"])
                    or int(row["initial_condition_id"]) != int(case["initial_condition_id"])
                    or int(row["boundary_id"]) != int(case["boundary_id"])
                    or int(row["support_seed"]) != SUPPORT_SEED
                    or int(row["support_rank"]) != int(case["support_rank"])
                    or int(row["resolution"]) != 256
                    or int(row["steps"]) != 20
                    or row.get("role") != "support_hr"
                ):
                    raise ValueError(f"Existing HR support shard row differs at rollout {rollout}.")
        else:
            rows = write_v4_rollouts_from_arrays(
                arrays,
                worker_meta,
                shard_dir=PROJECT_ROOT / DATA_ROOT / "support_hr_s271829_256",
                target_task_by_boundary=task_by_boundary,
                role="support_hr",
                source_sha256=manifest["sha256"],
            )
            write_index(INDEX_OUTPUT, rows, metadata=index_meta)
        for task in TASKS:
            subset_path = PROJECT_ROOT / DATA_ROOT / f"task{task:02d}_support_hr_s{SUPPORT_SEED}_m4.index.json"
            if subset_path.exists():
                existing_subset = read_index(subset_path, verify_hashes=True)
                expected_ids = {
                    int(case["initial_condition_id"])
                    for case in cases if int(case["target_task_id"]) == task
                }
                actual_ids = {int(row["initial_condition_id"]) for row in existing_subset["shards"]}
                if (
                    existing_subset["metadata"].get("parent_sha256")
                    != json.loads(INDEX_OUTPUT.read_text(encoding="utf-8"))["sha256"]
                    or existing_subset["metadata"].get("target_task_id") != task
                    or len(existing_subset["shards"]) != 4
                    or actual_ids != expected_ids
                ):
                    raise FileExistsError(f"Existing HR support subset differs: {subset_path}")
            else:
                subset_index(
                    INDEX_OUTPUT,
                    subset_path,
                    target_task_id=task,
                    max_rollouts=4,
                )
        solver_s = float(arrays["solver_seconds"].sum())
        assembly_s = float(arrays["assembly_seconds"].sum())
        audit_wall_seconds = time.perf_counter() - audit["started_monotonic"]
        if reused_raw_outputs:
            packaging_wall_seconds = audit_wall_seconds
            current_attempt_wall_seconds = float(
                prior_audit["finished_unix"] - prior_audit["started_unix"]
            )
        else:
            packaging_wall_seconds = max(audit_wall_seconds - current_attempt_wall_seconds, 0.0)
        audit.update({
            "worker_sha256_executed_source": execution_worker_sha256,
            "worker_sha256_reported_at_completion": worker_meta["worker_sha256"],
            "worker_source_snapshot": str(worker_snapshot_path.relative_to(PROJECT_ROOT)),
            "worker_source_snapshot_sha256": execution_worker_sha256,
            "index_sha256": json.loads(INDEX_OUTPUT.read_text(encoding="utf-8"))["sha256"],
            "solver_seconds_sum": solver_s,
            "assembly_seconds_sum": assembly_s,
            "raw_npz_sha256": sha256_file(npz),
            "raw_worker_meta_sha256": sha256_file(worker_meta_path),
            "rollouts_converged": int(arrays["solver_converged"].sum()),
            "solver_seconds_by_task": {
                str(task): float(arrays["solver_seconds"][arrays["target_task_id"] == task].sum())
                for task in TASKS
            },
            "query_used": False,
            "generation_attempt_wall_seconds": current_attempt_wall_seconds,
            "packaging_wall_seconds": packaging_wall_seconds,
        })
        audit["generation_wall_seconds"] = current_attempt_wall_seconds + packaging_wall_seconds
    generation = {
        "schema": "blrpng_phase1_hr_scratch_support_generation_v1",
        "campaign_id": CAMPAIGN,
        "tasks": list(TASKS),
        "model_seed": 0,
        "support_seed": SUPPORT_SEED,
        "support_budget": 4,
        "rollouts": 16,
        "rollout_steps": 20,
        "resolution": 256,
        "fidelity": 1,
        "protocol": {"dt": 0.2, "epsilon": 0.03, "gamma": 5.0, "progress_threshold": 0.25},
        "source_support_manifest": str(SUPPORT_MANIFEST),
        "source_support_manifest_sha256": SUPPORT_MANIFEST_SHA256,
        "source_support_index": str(SUPPORT_PARENT_INDEX),
        "source_support_index_sha256": SUPPORT_PARENT_INDEX_SHA256,
        "source_m4_indices": manifest["metadata"]["source_m4_indices"],
        "hr_validation_manifest_sha256": HR_VALIDATION_MANIFEST_SHA256,
        "hr_support_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "hr_support_manifest_sha256": manifest["sha256"],
        "hr_support_index": str(INDEX_OUTPUT),
        "hr_support_index_sha256": json.loads(INDEX_OUTPUT.read_text(encoding="utf-8"))["sha256"],
        "worker_sha256": execution_worker_sha256,
        "worker_sha256_reported_at_completion": worker_meta["worker_sha256"],
        "worker_source_snapshot": str(worker_snapshot_path.relative_to(PROJECT_ROOT)),
        "worker_source_snapshot_sha256": execution_worker_sha256,
        "original_generation_audit": (
            None if prior_audit_path is None else str(prior_audit_path.relative_to(PROJECT_ROOT))
        ),
        "original_generation_audit_sha256": (
            None if prior_audit_path is None else sha256_file(prior_audit_path)
        ),
        "raw_npz": str(npz.relative_to(PROJECT_ROOT)),
        "raw_npz_sha256": sha256_file(npz),
        "raw_worker_meta": str(worker_meta_path.relative_to(PROJECT_ROOT)),
        "raw_worker_meta_sha256": sha256_file(worker_meta_path),
        "generation_attempt_wall_seconds": current_attempt_wall_seconds,
        "packaging_wall_seconds": packaging_wall_seconds,
        "generation_wall_seconds": current_attempt_wall_seconds + packaging_wall_seconds,
        "solver_seconds_sum": float(arrays["solver_seconds"].sum()),
        "assembly_seconds_sum": float(arrays["assembly_seconds"].sum()),
        "per_task_solver_seconds": {
            str(task): float(arrays["solver_seconds"][arrays["target_task_id"] == task].sum())
            for task in TASKS
        },
        "all_converged": bool(arrays["solver_converged"].all()),
        "query_used": False,
    }
    write_json(GENERATION_REPORT, generation)
    return generation


def main() -> None:
    orchestration_started = time.perf_counter()
    manifest_path, manifest, _parent_index = _make_manifest()
    if INDEX_OUTPUT.exists():
        index = read_index(INDEX_OUTPUT, verify_hashes=True)
        if (
            index["metadata"].get("manifest_sha256") != manifest["sha256"]
            or index["metadata"].get("split") != "support_hr"
            or len(index["shards"]) != 16
        ):
            raise FileExistsError(f"Existing HR support index differs: {INDEX_OUTPUT}")
        for task in TASKS:
            subset_path = PROJECT_ROOT / DATA_ROOT / f"task{task:02d}_support_hr_s{SUPPORT_SEED}_m4.index.json"
            subset = read_index(subset_path, verify_hashes=True)
            if subset["metadata"].get("target_task_id") != task or len(subset["shards"]) != 4:
                raise ValueError(f"Existing HR support subset differs: {subset_path}")
        if GENERATION_REPORT.exists():
            print(json.dumps({"index": str(INDEX_OUTPUT), "reused": True}, indent=2))
            return
    generation = _generate(manifest_path, manifest)
    current_orchestration = time.perf_counter() - orchestration_started
    prior_generation_attempt = (
        float(generation.get("generation_attempt_wall_seconds", 0.0))
        if generation.get("original_generation_audit") else 0.0
    )
    orchestration_total = current_orchestration + prior_generation_attempt
    generation["manifest_and_input_validation_seconds"] = max(
        orchestration_total - generation["generation_wall_seconds"], 0.0
    )
    generation["orchestration_total_wall_seconds"] = orchestration_total
    write_json(GENERATION_REPORT, generation)
    print(json.dumps(generation, indent=2))


if __name__ == "__main__":
    main()
