from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.capacity import artifact_size_bytes, require_capacity, require_prerequisite_gate
from blrpng.data.generate import SFEPY_PYTHON, _case_chunks, workspace_audit
from blrpng.data.manifest import read_manifest
from blrpng.data.shards import read_index, subset_index, write_index, write_v4_rollouts_from_arrays
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file


def _role_settings(role: str, resolution: int, physics: dict) -> tuple[int, int, int]:
    if role in {"support", "validation"}:
        if resolution != 64:
            raise ValueError(f"{role} must use the frozen 64x64 support resolution.")
        return 65, int(physics["data"]["train_steps"]), 0
    if role == "query":
        if resolution not in {128, 256}:
            raise ValueError("query resolution must be 128 or 256.")
        return resolution + 1, int(physics["data"]["query_steps"]), 1
    if role == "query_512":
        if resolution != 512:
            raise ValueError("query_512 must use resolution 512.")
        return 513, int(physics["data"]["query_steps"]), 1
    raise ValueError(f"Unsupported reconfirmation role: {role}")


def _estimate_bytes(case_count: int, resolution: int, steps: int) -> int:
    # Restrict sampling to known atomic-v4 shard directories.  Never probe an
    # arbitrary .pt here: legacy materialized datasets can be tens of GiB.
    references = (
        (64, 10, "artifacts/data/allen_cahn_v2_support_lr_64"),
        (128, 50, "artifacts/data/allen_cahn_v2_query_hr_128"),
        (256, 50, "artifacts/data/allen_cahn_v2_query_hr_256"),
    )
    ranked = sorted(references, key=lambda item: abs(item[0] - resolution))
    for source_resolution, source_steps, relative in ranked:
        samples = sorted(resolve_in_project(relative).glob("rollout_*.pt"))[:16]
        if samples:
            mean = sum(path.stat().st_size for path in samples) / len(samples)
            scaled = mean * (resolution / source_resolution) ** 2
            scaled *= max(steps, 1) / max(source_steps, 1)
            return int(case_count * scaled * 1.10)
    # Conservative tensor-level fallback for trajectory, correction and metadata.
    return int(case_count * (steps * 3 + 4) * resolution * resolution * 4 * 1.25)


def _write_task_indices(
    global_index: Path,
    *,
    role: str,
    resolution: int,
    task_count: int,
    support_budgets: list[int],
) -> list[Path]:
    outputs: list[Path] = []
    for task in range(task_count):
        if role == "support":
            for budget in support_budgets:
                outputs.append(
                    subset_index(
                        global_index,
                        f"artifacts/data/reconfirmation/task{task:02d}_{role}_{resolution}_m{budget}.index.json",
                        target_task_id=task,
                        max_rollouts=budget,
                    )
                )
        else:
            outputs.append(
                subset_index(
                    global_index,
                    f"artifacts/data/reconfirmation/task{task:02d}_{role}_{resolution}_all.index.json",
                    target_task_id=task,
                )
            )
    return outputs


def _generate(
    *,
    manifest_path: Path,
    role: str,
    resolution: int,
    mesh_resolution: int,
    steps: int,
    fidelity: int,
    workers: int,
    physics: dict,
) -> Path:
    stem = f"allen_cahn_reconfirmation_{role}_{resolution}"
    index_path = ensure_artifact_path(f"artifacts/data/reconfirmation/{stem}.index.json")
    if index_path.exists():
        read_index(index_path, verify_hashes=True)
        return index_path
    manifest = read_manifest(manifest_path)
    cases = [case for case in manifest["cases"] if case["split"] == role]
    chunks = _case_chunks(len(cases), min(workers, len(cases)))
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(PROJECT_ROOT),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    processes = []
    for part, (start, count) in enumerate(chunks):
        prefix = f"artifacts/tmp/{stem}.part{part:03d}"
        npz = ensure_artifact_path(prefix + ".npz")
        meta = ensure_artifact_path(prefix + ".json")
        log = ensure_artifact_path(prefix + ".log")
        command = [
            str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_worker",
            "--manifest", str(manifest_path), "--split", role,
            "--output-npz", str(npz), "--output-meta", str(meta),
            "--output-resolution", str(resolution),
            "--mesh-resolution", str(mesh_resolution),
            "--steps", str(steps), "--fidelity", str(fidelity),
            "--dt", str(physics["data"]["dt"]),
            "--epsilon", str(physics["data"]["eps"]),
            "--gamma", str(physics["data"]["gamma"]),
            "--progress-threshold", str(physics["data"]["residual_progress_threshold"]),
            "--case-start", str(start), "--limit", str(count),
        ]
        stream = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
        processes.append((process, stream, log, npz, meta, command))
    failures = []
    for process, stream, log, _npz, _meta, command in processes:
        process.wait()
        stream.close()
        if process.returncode:
            failures.append(
                f"{' '.join(command)}\n{log.read_text(encoding='utf-8', errors='replace')}"
            )
    if failures:
        raise RuntimeError("\n\n".join(failures))
    mapping = {
        int(case["boundary_id"]): int(case["target_task_id"])
        for case in cases
    }
    rows = []
    for _process, _stream, log, npz, meta_path, _command in processes:
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.extend(
            write_v4_rollouts_from_arrays(
                arrays, meta,
                shard_dir=f"artifacts/data/reconfirmation/{stem}",
                target_task_by_boundary=mapping,
                role=role,
                source_sha256=sha256_file(manifest_path),
            )
        )
        npz.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)
    rows.sort(key=lambda row: (int(row["target_task_id"]), int(row["initial_condition_id"])))
    return write_index(
        index_path,
        rows,
        metadata={
            "experiment": "allen_cahn_blind_reconfirmation",
            "split": role,
            "resolution": resolution,
            "steps": steps,
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "generation_workers": len(chunks),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen Allen-Cahn reconfirmation shards.")
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    parser.add_argument("--physics-config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--role", choices=("support", "validation", "query", "query_512"), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    physics = load_config(args.physics_config)
    oracle_gate = require_prerequisite_gate(config["gates"]["required_oracle_manifest"])
    if args.role.startswith("query"):
        require_prerequisite_gate(config["gates"]["required_method_freeze_manifest"])
        require_prerequisite_gate(config["gates"]["required_checkpoint_freeze_manifest"])
        require_prerequisite_gate(config["gates"]["required_query_protocol_freeze_manifest"])
    manifest_path = resolve_in_project(
        f"artifacts/manifests/allen_cahn_reconfirmation_{args.role}.json",
        must_exist=True,
    )
    manifest = read_manifest(manifest_path)
    mesh, steps, fidelity = _role_settings(args.role, args.resolution, physics)
    estimate = _estimate_bytes(len(manifest["cases"]), args.resolution, steps)
    capacity = require_capacity(
        estimate, maximum_gb=float(config["resources"]["max_total_artifacts_gb"])
    )
    summary = {
        "role": args.role,
        "resolution": args.resolution,
        "cases": len(manifest["cases"]),
        "estimated_new_gib": estimate / 2**30,
        "current_artifacts_gib": artifact_size_bytes() / 2**30,
        "capacity": capacity,
        "oracle_status": oracle_gate.get("oracle_status"),
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return
    with workspace_audit(f"generate_reconfirmation_{args.role}_{args.resolution}") as audit:
        global_index = _generate(
            manifest_path=manifest_path, role=args.role, resolution=args.resolution,
            mesh_resolution=mesh, steps=steps, fidelity=fidelity,
            workers=args.workers, physics=physics,
        )
        outputs = _write_task_indices(
            global_index,
            role=args.role,
            resolution=args.resolution,
            task_count=len(config["blind_protocol"]["families"])
            * int(config["blind_protocol"]["tasks_per_family"]),
            support_budgets=[int(value) for value in config["blind_protocol"]["support_budgets"]],
        )
        audit.update({**summary, "global_index": str(global_index), "task_indices": len(outputs)})
    print(json.dumps({**summary, "global_index": str(global_index), "task_indices": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
