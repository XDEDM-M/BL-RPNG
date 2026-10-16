from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from blrpng.data.generate import SFEPY_PYTHON, workspace_audit
from blrpng.data.shards import read_index, subset_index, write_index, write_v4_rollouts_from_arrays
from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json, write_json


CAMPAIGN = "hrdev_20261015_hrsupport_m16_s271829"
TASKS = (0, 1, 3, 5)
SUPPORT_SEED = 271829
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
TASK_SHA256 = "bb50c7e4d5b8d0900e21d7ae64235c6d955244cc3bfdca1a97ab6056596a334a"
SUPPORT_MANIFEST_SHA256 = "dfbc58d5b62b870b60804df9b33fad8a27cc5370c6070aee679028ddaf098138"
SUPPORT_PARENT_INDEX_SHA256 = "c5bea773b1f414c2d2deaa9bee9e35343a38b6d7c4268a86505a6585113499f1"
HR_VALIDATION_MANIFEST_SHA256 = "d491452f11d9c75b62c385fd9ecf2df40b960053782e0b2c31a63a819cd21ac5"
M16_INDEX_SHA256 = {
    0: "5e3261ff2fd825c9d3f01f67c4d1afb3f6438af55608c1a8763b1557877507bc",
    1: "97586326540dc9d5a3c288e9930b66e603a3402dfff116adccc70fa854509eeb",
    3: "f2a17dec1138e633845cafe431d9e59df76eafe31278565f2840a885a2eea93f",
    5: "54ee8bc914c4cb74ae71be2741f32e6646fa886e4b187d6771a1005150ebaac8",
}
MANIFEST_OUTPUT = Path(
    f"artifacts/manifests/phase1_development/{CAMPAIGN}/support_hr_s{SUPPORT_SEED}_m16.json"
)
DATA_ROOT = Path(f"artifacts/data/phase1_development/{CAMPAIGN}")
INDEX_OUTPUT = DATA_ROOT / f"support_hr_s{SUPPORT_SEED}_m16_256.index.json"
GENERATION_REPORT = Path(f"artifacts/reports/phase1_hr_scratch_data_generation_{CAMPAIGN}.json")


def _load_sealed(path: Path, expected_sha: str) -> dict[str, Any]:
    payload = json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    actual = payload.get("sha256")
    if actual != sha256_json({key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError(f"Manifest or index seal is invalid: {path}")
    if actual != expected_sha:
        raise ValueError(f"Frozen artifact hash changed for {path}: {actual}")
    return payload


def _make_manifest() -> tuple[Path, dict[str, Any]]:
    tasks = _load_sealed(TASK_MANIFEST, TASK_SHA256)
    support = _load_sealed(SUPPORT_MANIFEST, SUPPORT_MANIFEST_SHA256)
    validation = _load_sealed(HR_VALIDATION_MANIFEST, HR_VALIDATION_MANIFEST_SHA256)
    parent = read_index(PROJECT_ROOT / SUPPORT_PARENT_INDEX, verify_hashes=True)
    if parent["sha256"] != SUPPORT_PARENT_INDEX_SHA256:
        raise ValueError("Frozen Stage-2 LR support parent index changed.")
    if (
        support.get("metadata", {}).get("role") != "support"
        or int(support.get("metadata", {}).get("support_seed", -1)) != SUPPORT_SEED
        or support.get("metadata", {}).get("task_manifest_sha256") != TASK_SHA256
    ):
        raise ValueError("Frozen LR support manifest metadata changed.")

    task_lookup = {int(row["target_task_id"]): row for row in tasks["tasks"]}
    source_cases = {
        (int(case["target_task_id"]), int(case["support_rank"])): case
        for case in support["cases"]
    }
    parent_ids = {int(row["initial_condition_id"]) for row in parent["shards"]}
    validation_ids = {int(case["initial_condition_id"]) for case in validation["cases"]}
    source_indices = []
    selected_cases = []
    for task_id in TASKS:
        rel = Path(
            "artifacts/data/phase1_development/hrdev_20261013_firstpath_ss271829/"
            f"task{task_id:02d}_support_s{SUPPORT_SEED}_m16.index.json"
        )
        index = read_index(PROJECT_ROOT / rel, verify_hashes=True)
        if index["sha256"] != M16_INDEX_SHA256[task_id]:
            raise ValueError(f"Frozen m=16 LR support index changed: {rel}")
        rows = sorted(index["shards"], key=lambda row: int(row.get("support_rank", -1)))
        if [int(row.get("support_rank", -1)) for row in rows] != list(range(16)):
            raise ValueError(f"Support index does not contain ranks 0..15: {rel}")
        if any(
            int(row["target_task_id"]) != task_id
            or row.get("role") != "support"
            or int(row["resolution"]) != 64
            or int(row["steps"]) != 20
            or int(row["initial_condition_id"]) not in parent_ids
            for row in rows
        ):
            raise ValueError(f"m=16 support rows are not from the frozen LR parent: {rel}")
        source_indices.append({"task": task_id, "path": str(rel), "sha256": index["sha256"]})
        for rank in range(16):
            case = source_cases[(task_id, rank)]
            task = task_lookup[task_id]
            if (
                int(case["boundary_id"]) != int(task["boundary_id"])
                or case["boundary_descriptor"] != task["boundary"]
                or case["boundary_family"] != task["family"]
                or int(case["initial_condition_id"]) != int(rows[rank]["initial_condition_id"])
            ):
                raise ValueError("HR support case differs from the frozen task or LR support index.")
            if int(case["initial_condition_id"]) in validation_ids:
                raise ValueError("HR support and HR-dev validation initial conditions overlap.")
            selected_cases.append({**case, "split": "support_hr"})

    if len(selected_cases) != 64 or len({int(c["initial_condition_id"]) for c in selected_cases}) != 64:
        raise ValueError("Expected 64 unique HR support rollouts across the four tasks.")
    metadata = {
        "experiment": f"phase1_development_{CAMPAIGN}",
        "campaign_id": CAMPAIGN,
        "role": "support_hr",
        "evaluation_role": "support_hr_dev",
        "support_seed": SUPPORT_SEED,
        "initial_conditions_per_task": 16,
        "initial_condition_seed": int(support["metadata"]["initial_condition_seed"]),
        "task_manifest": str(TASK_MANIFEST),
        "task_manifest_sha256": tasks["sha256"],
        "source_support_manifest": str(SUPPORT_MANIFEST),
        "source_support_manifest_sha256": support["sha256"],
        "source_support_index": str(SUPPORT_PARENT_INDEX),
        "source_support_index_sha256": parent["sha256"],
        "source_m16_indices": source_indices,
        "contains_solutions": False,
        "note": "Nested m=16 development support cases copied exactly from the frozen Stage-2 LR support draw; HR-dev IDs are disjoint.",
    }
    payload = {"schema": "blrpng_manifest_v1", "metadata": metadata, "cases": selected_cases}
    payload["sha256"] = sha256_json(payload)
    path = PROJECT_ROOT / MANIFEST_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"Existing HR m=16 manifest differs: {path}")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, payload


def _generate(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not SFEPY_PYTHON.exists():
        raise FileNotFoundError(f"SfePy worker interpreter not found: {SFEPY_PYTHON}")
    npz = PROJECT_ROOT / f"artifacts/tmp/phase1_{CAMPAIGN}_support_hr_256.npz"
    worker_meta_path = PROJECT_ROOT / f"artifacts/tmp/phase1_{CAMPAIGN}_support_hr_256.json"
    log_path = PROJECT_ROOT / f"artifacts/experiments/phase1/phase1_{CAMPAIGN}_support_hr_256.log"
    if npz.exists() or worker_meta_path.exists():
        if not (npz.exists() and worker_meta_path.exists()):
            raise FileExistsError("Only one HR solver output exists; preserving the incomplete pair.")
        raise FileExistsError("Preserving existing raw HR m=16 solver outputs; no overwrite.")
    command = [
        str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_worker",
        "--manifest", str(manifest_path), "--split", "support_hr",
        "--output-npz", str(npz), "--output-meta", str(worker_meta_path),
        "--output-resolution", "256", "--mesh-resolution", "257",
        "--steps", "20", "--fidelity", "1", "--dt", "0.2",
        "--epsilon", "0.03", "--gamma", "5.0", "--progress-threshold", "0.25",
        "--case-start", "0", "--limit", "64",
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(PROJECT_ROOT),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with workspace_audit(f"phase1_{CAMPAIGN}_support_hr_generation") as audit:
        audit.update({
            "campaign_id": CAMPAIGN,
            "role": "support_hr",
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": manifest["sha256"],
            "case_count": 64,
            "tasks": list(TASKS),
            "steps": 20,
            "resolution": 256,
            "fidelity": 1,
            "command": command,
            "query_used": False,
            "query_data_generated": False,
        })
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, text=True,
            )
        solver_wall = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(f"HR support SfePy worker failed; see {log_path}")
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
        worker_meta = json.loads(worker_meta_path.read_text(encoding="utf-8"))
        if (
            worker_meta.get("manifest_sha256") != manifest["sha256"]
            or int(worker_meta.get("num_rollouts", -1)) != 64
            or int(worker_meta.get("num_samples", -1)) != 1280
            or int(worker_meta.get("output_resolution", -1)) != 256
            or int(worker_meta.get("mesh_resolution", -1)) != 257
            or int(worker_meta.get("steps", -1)) != 20
            or worker_meta.get("split") != "support_hr"
            or not worker_meta.get("all_converged")
            or not bool(arrays["solver_converged"].all())
            or not bool((arrays["fidelity"] == 1).all())
        ):
            raise RuntimeError("HR m=16 solver output failed protocol/convergence checks.")
        cases = manifest["cases"]
        for rollout, case in enumerate(cases):
            start, end = rollout * 20, (rollout + 1) * 20
            expected = {
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
                raise RuntimeError(f"HR m=16 step ordering failed for rollout {rollout}.")
            for key, value in expected.items():
                if not np.all(arrays[key][start:end].numpy() == value):
                    raise RuntimeError(f"HR m=16 solver output {key} mismatch at rollout {rollout}.")
        for key in ("x", "y", "boundary_values", "boundary_lift", "newton_correction"):
            if not bool(torch.isfinite(arrays[key]).all()):
                raise RuntimeError(f"HR m=16 solver output has non-finite {key} values.")
        source_hash = sha256_file(PROJECT_ROOT / "blrpng/solvers/sfepy_worker.py")
        task_by_boundary = {int(c["boundary_id"]): int(c["target_task_id"]) for c in cases}
        source_indices = manifest["metadata"]["source_m16_indices"]
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
            "task_manifest_sha256": TASK_SHA256,
            "source_support_manifest_sha256": SUPPORT_MANIFEST_SHA256,
            "source_support_index_sha256": SUPPORT_PARENT_INDEX_SHA256,
            "source_m16_indices": source_indices,
            "generation_workers": 1,
            "fidelity": 1,
            "worker_sha256": source_hash,
            "worker_source_snapshot_sha256": source_hash,
            "query_used": False,
        }
        rows = write_v4_rollouts_from_arrays(
            arrays, worker_meta,
            shard_dir=PROJECT_ROOT / DATA_ROOT / "support_hr_s271829_m16_256",
            target_task_by_boundary=task_by_boundary,
            role="support_hr", source_sha256=manifest["sha256"],
        )
        write_index(INDEX_OUTPUT, rows, metadata=index_meta)
        for task_id in TASKS:
            subset_index(
                INDEX_OUTPUT,
                DATA_ROOT / f"task{task_id:02d}_support_hr_s271829_m16.index.json",
                target_task_id=task_id,
                max_rollouts=16,
            )
        report = {
            "schema": "blrpng_phase1_hr_scratch_support_generation_v1",
            "campaign_id": CAMPAIGN,
            "tasks": list(TASKS),
            "support_seed": SUPPORT_SEED,
            "support_budget": 16,
            "rollouts": 64,
            "rollout_steps": 20,
            "resolution": 256,
            "fidelity": 1,
            "protocol": {"dt": 0.2, "epsilon": 0.03, "gamma": 5.0, "progress_threshold": 0.25},
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": manifest["sha256"],
            "source_m16_indices": source_indices,
            "hr_validation_manifest_sha256": HR_VALIDATION_MANIFEST_SHA256,
            "hr_support_index": str(INDEX_OUTPUT),
            "hr_support_index_sha256": json.loads(INDEX_OUTPUT.read_text())["sha256"],
            "worker_sha256": source_hash,
            "raw_npz": str(npz.relative_to(PROJECT_ROOT)),
            "raw_npz_sha256": sha256_file(npz),
            "raw_worker_meta": str(worker_meta_path.relative_to(PROJECT_ROOT)),
            "raw_worker_meta_sha256": sha256_file(worker_meta_path),
            "generation_wall_seconds": solver_wall,
            "solver_seconds_sum": float(arrays["solver_seconds"].sum()),
            "assembly_seconds_sum": float(arrays["assembly_seconds"].sum()),
            "all_converged": bool(arrays["solver_converged"].all()),
            "query_used": False,
        }
        audit.update({
            "index_sha256": report["hr_support_index_sha256"],
            "solver_seconds_sum": report["solver_seconds_sum"],
            "rollouts_converged": int(arrays["solver_converged"].sum()),
            "raw_npz_sha256": report["raw_npz_sha256"],
            "query_used": False,
        })
    write_json(GENERATION_REPORT, report)
    return report


def main() -> None:
    manifest_path, manifest = _make_manifest()
    if INDEX_OUTPUT.exists() and GENERATION_REPORT.exists():
        index = read_index(INDEX_OUTPUT, verify_hashes=True)
        report = json.loads((PROJECT_ROOT / GENERATION_REPORT).read_text(encoding="utf-8"))
        if (
            len(index["shards"]) == 64
            and index["metadata"].get("manifest_sha256") == manifest["sha256"]
            and report.get("all_converged") is True
        ):
            print(json.dumps({"index": str(INDEX_OUTPUT), "reused": True}, indent=2))
            return
        raise ValueError("Existing HR m=16 artifacts failed reuse verification.")
    if INDEX_OUTPUT.exists() and not GENERATION_REPORT.exists():
        index = read_index(PROJECT_ROOT / INDEX_OUTPUT, verify_hashes=True)
        npz = PROJECT_ROOT / f"artifacts/tmp/phase1_{CAMPAIGN}_support_hr_256.npz"
        worker_meta_path = PROJECT_ROOT / f"artifacts/tmp/phase1_{CAMPAIGN}_support_hr_256.json"
        if not npz.is_file() or not worker_meta_path.is_file():
            raise FileNotFoundError("Cannot recover the report without both raw solver outputs.")
        worker_meta = json.loads(worker_meta_path.read_text(encoding="utf-8"))
        audit_paths = sorted(
            (PROJECT_ROOT / "artifacts/reports/workspace_audits").glob(
                f"phase1_{CAMPAIGN}_support_hr_generation_*.json"
            )
        )
        if not audit_paths:
            raise FileNotFoundError("No preserved workspace audit exists for HR m=16 generation.")
        audit = json.loads(audit_paths[-1].read_text(encoding="utf-8"))
        if (
            len(index["shards"]) != 64
            or index["metadata"].get("manifest_sha256") != manifest["sha256"]
            or index["metadata"].get("query_used") is not False
            or worker_meta.get("manifest_sha256") != manifest["sha256"]
            or int(worker_meta.get("num_rollouts", -1)) != 64
            or int(worker_meta.get("num_samples", -1)) != 1280
            or worker_meta.get("all_converged") is not True
            or audit.get("manifest_sha256") != manifest["sha256"]
            or audit.get("case_count") != 64
            or audit.get("query_used") is not False
            or audit.get("query_data_generated") is not False
        ):
            raise ValueError("Existing HR m=16 outputs failed report-recovery provenance checks.")
        for task_id in TASKS:
            subset = read_index(
                PROJECT_ROOT / DATA_ROOT / f"task{task_id:02d}_support_hr_s271829_m16.index.json",
                verify_hashes=True,
            )
            if (
                len(subset["shards"]) != 16
                or subset["metadata"].get("target_task_id") != task_id
                or sorted(int(row["support_rank"]) for row in subset["shards"]) != list(range(16))
            ):
                raise ValueError(f"Existing HR m=16 task subset failed recovery audit: {task_id}")
        with np.load(npz, allow_pickle=False) as archive:
            converged = archive["solver_converged"]
            fidelity = archive["fidelity"]
            solver_seconds = archive["solver_seconds"]
            assembly_seconds = archive["assembly_seconds"]
            if (
                tuple(converged.shape) != (1280,)
                or tuple(fidelity.shape) != (1280,)
                or not bool(converged.all())
                or not bool((fidelity == 1).all())
            ):
                raise ValueError("Raw HR m=16 solver arrays failed recovery convergence audit.")
            solver_seconds_sum = float(solver_seconds.sum())
            assembly_seconds_sum = float(assembly_seconds.sum())
        source_hash = str(index["metadata"]["worker_sha256"])
        report = {
            "schema": "blrpng_phase1_hr_scratch_support_generation_v1",
            "campaign_id": CAMPAIGN,
            "tasks": list(TASKS),
            "support_seed": SUPPORT_SEED,
            "support_budget": 16,
            "rollouts": 64,
            "rollout_steps": 20,
            "resolution": 256,
            "fidelity": 1,
            "protocol": {"dt": 0.2, "epsilon": 0.03, "gamma": 5.0, "progress_threshold": 0.25},
            "manifest": str(MANIFEST_OUTPUT),
            "manifest_sha256": manifest["sha256"],
            "source_m16_indices": manifest["metadata"]["source_m16_indices"],
            "hr_validation_manifest_sha256": HR_VALIDATION_MANIFEST_SHA256,
            "hr_support_index": str(INDEX_OUTPUT),
            "hr_support_index_sha256": index["sha256"],
            "worker_sha256": source_hash,
            "raw_npz": str(npz.relative_to(PROJECT_ROOT)),
            "raw_npz_sha256": sha256_file(npz),
            "raw_worker_meta": str(worker_meta_path.relative_to(PROJECT_ROOT)),
            "raw_worker_meta_sha256": sha256_file(worker_meta_path),
            "generation_wall_seconds": float(audit["finished_unix"] - audit["started_unix"]),
            "solver_seconds_sum": solver_seconds_sum,
            "assembly_seconds_sum": assembly_seconds_sum,
            "all_converged": True,
            "query_used": False,
            "report_recovered_after_packaging_path_error": True,
        }
        write_json(GENERATION_REPORT, report)
        print(json.dumps(report, indent=2))
        return
    if INDEX_OUTPUT.exists() or GENERATION_REPORT.exists():
        raise FileExistsError("Partial HR m=16 artifacts exist; preserving them without overwrite.")
    report = _generate(manifest_path, manifest)
    write_json(GENERATION_REPORT, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
