from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.generate import SFEPY_PYTHON, _case_chunks, workspace_audit
from blrpng.data.manifest import CaseSpec, read_manifest, write_manifest
from blrpng.data.shards import read_index, write_index, write_v4_rollouts_from_arrays
from blrpng.paths import (
    ARTIFACT_ROOT,
    PROJECT_ROOT,
    ensure_artifact_path,
    sha256_file,
)


def _artifact_bytes() -> int:
    return sum(path.stat().st_size for path in ARTIFACT_ROOT.rglob("*") if path.is_file())


def _parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or len(result) != len(set(result)):
        raise ValueError("Expected a non-empty comma-separated list of unique integers.")
    return result


def _extension_manifest(
    config: dict, *, tasks: list[int], base_budget: int, maximum_budget: int
) -> Path:
    calibration_path = PROJECT_ROOT / "artifacts/manifests/allen_cahn_v2_calibration.json"
    calibration = read_manifest(calibration_path)
    mapping = {
        int(boundary): int(task)
        for boundary, task in calibration["metadata"]["task_mapping"].items()
    }
    by_task: dict[int, dict] = {}
    for row in calibration["cases"]:
        task = mapping[int(row["boundary_id"])]
        by_task.setdefault(task, row)
    missing = sorted(set(tasks) - set(by_task))
    if missing:
        raise ValueError(f"Unknown calibration tasks: {missing}")
    count = maximum_budget - base_budget
    if count <= 0:
        raise ValueError("maximum_budget must exceed base_budget")
    rows: list[CaseSpec] = []
    seed = int(config["experiment"]["seed"])
    for task in tasks:
        boundary = by_task[task]
        rng = np.random.default_rng(seed + 700_001 + 104_729 * task)
        for local in range(base_budget, maximum_budget):
            identifier = 4_000_000 + 1_000 * task + local
            rows.append(
                CaseSpec(
                    case_id=identifier,
                    pair_id=identifier,
                    boundary_id=int(boundary["boundary_id"]),
                    initial_condition_id=identifier,
                    split="oracle_train_extension",
                    boundary_coefficients=[float(v) for v in boundary["boundary_coefficients"]],
                    boundary_frequencies=[float(v) for v in boundary["boundary_frequencies"]],
                    boundary_amplitude=float(boundary["boundary_amplitude"]),
                    initial_coefficients=[float(v) for v in rng.normal(size=4)],
                    initial_phase=float(0.17 * local + 0.031 * task),
                )
            )
    task_tag = "_".join(map(str, tasks))
    output = ensure_artifact_path(
        f"artifacts/manifests/allen_cahn_oracle_extension_m{maximum_budget}_tasks_{task_tag}.json"
    )
    metadata = {
        "schema": "blrpng_oracle_extension_manifest_v1",
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "source_calibration_manifest": str(calibration_path.relative_to(PROJECT_ROOT)),
        "source_calibration_sha256": sha256_file(calibration_path),
        "tasks": tasks,
        "base_budget": base_budget,
        "maximum_budget": maximum_budget,
        "rollouts_per_task": count,
        "task_mapping": {str(key): value for key, value in mapping.items()},
        "frozen": True,
    }
    if output.exists():
        existing = read_manifest(output)
        expected_rows = [row.__dict__ for row in rows]
        if existing["metadata"] != metadata or existing["cases"] != expected_rows:
            raise RuntimeError(f"Frozen extension manifest disagrees with request: {output}")
        return output
    return write_manifest(output, rows, metadata=metadata)


def _estimate_bytes(base_budget: int, maximum_budget: int, tasks: list[int]) -> int:
    samples = sorted(
        (PROJECT_ROOT / "artifacts/data/allen_cahn_v2_oracle_train_256").glob("*.pt")
    )[:16]
    if not samples:
        raise FileNotFoundError("Cannot estimate extension size without the m64 oracle shards.")
    mean = sum(path.stat().st_size for path in samples) / len(samples)
    return int(mean * (maximum_budget - base_budget) * len(tasks) * 1.05)


def _generate_extension(
    config: dict, manifest_path: Path, *, tasks: list[int], workers: int
) -> Path:
    task_tag = "_".join(map(str, tasks))
    stem = f"allen_cahn_oracle_extension_256_tasks_{task_tag}"
    index_path = ensure_artifact_path(f"artifacts/data/{stem}.index.json")
    if index_path.exists():
        read_index(index_path, verify_hashes=True)
        return index_path
    manifest = read_manifest(manifest_path)
    cases = [row for row in manifest["cases"] if row["split"] == "oracle_train_extension"]
    chunks = _case_chunks(len(cases), min(workers, len(cases)))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    processes = []
    for part, (start, count) in enumerate(chunks):
        prefix = f"artifacts/tmp/{stem}.part{part:03d}"
        npz = ensure_artifact_path(prefix + ".npz")
        meta = ensure_artifact_path(prefix + ".json")
        log = ensure_artifact_path(prefix + ".log")
        command = [
            str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_worker",
            "--manifest", str(manifest_path), "--split", "oracle_train_extension",
            "--output-npz", str(npz), "--output-meta", str(meta),
            "--output-resolution", "256", "--mesh-resolution", "257",
            "--steps", str(config["data"]["train_steps"]), "--fidelity", "1",
            "--dt", str(config["data"]["dt"]), "--epsilon", str(config["data"]["eps"]),
            "--gamma", str(config["data"]["gamma"]),
            "--progress-threshold", str(config["data"]["residual_progress_threshold"]),
            "--case-start", str(start), "--limit", str(count),
        ]
        stream = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
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
    mapping = {int(key): int(value) for key, value in manifest["metadata"]["task_mapping"].items()}
    rows = []
    for _process, _stream, log, npz, meta_path, _command in processes:
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.extend(
            write_v4_rollouts_from_arrays(
                arrays,
                meta,
                shard_dir=f"artifacts/data/{stem}",
                target_task_by_boundary=mapping,
                role="oracle_train_extension",
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
            "experiment": config["experiment"]["name"],
            "split": "oracle_train_extension",
            "resolution": 256,
            "steps": int(config["data"]["train_steps"]),
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "generation_workers": len(chunks),
            "tasks": tasks,
        },
    )


def _merge_nested_indices(
    extension_path: Path, *, tasks: list[int], base_budget: int, budgets: list[int]
) -> list[Path]:
    extension = read_index(extension_path, verify_hashes=True)
    outputs = []
    for task in tasks:
        base_path = PROJECT_ROOT / (
            f"artifacts/data/tasks/task{task:02d}_oracle_train_256_m{base_budget}.index.json"
        )
        base = read_index(base_path, verify_hashes=True)
        base_rows = list(base["shards"])
        extra_rows = sorted(
            [row for row in extension["shards"] if int(row["target_task_id"]) == task],
            key=lambda row: (int(row["initial_condition_id"]), row["path"]),
        )
        if len(base_rows) != base_budget:
            raise ValueError(f"Task {task} base oracle has {len(base_rows)} rather than {base_budget} rollouts.")
        base_ids = {int(row["initial_condition_id"]) for row in base_rows}
        if base_ids & {int(row["initial_condition_id"]) for row in extra_rows}:
            raise ValueError(f"Task {task} extension overlaps base initial conditions.")
        for budget in budgets:
            rows = base_rows + extra_rows[: budget - base_budget]
            if len(rows) != budget:
                raise ValueError(f"Task {task} cannot construct nested oracle budget m{budget}.")
            output = write_index(
                f"artifacts/data/tasks/task{task:02d}_oracle_train_256_m{budget}.index.json",
                rows,
                metadata={
                    "experiment": "allen_cahn_oracle_convergence",
                    "split": "oracle_train",
                    "resolution": 256,
                    "steps": int(base["metadata"]["steps"]),
                    "target_task_id": task,
                    "max_rollouts": budget,
                    "nested_base_index": str(base_path.relative_to(PROJECT_ROOT)),
                    "nested_base_sha256": base["sha256"],
                    "extension_index": str(extension_path.relative_to(PROJECT_ROOT)),
                    "extension_sha256": extension["sha256"],
                    "frozen": True,
                },
            )
            read_index(output, verify_hashes=True)
            outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nested m128/m256 HR oracle support.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--tasks", default="0,2,7")
    parser.add_argument("--base-budget", type=int, default=64)
    parser.add_argument("--budgets", default="128,256")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = _parse_ints(args.tasks)
    budgets = sorted(_parse_ints(args.budgets))
    if min(budgets) <= args.base_budget:
        raise ValueError("All extension budgets must exceed base-budget.")
    estimate = _estimate_bytes(args.base_budget, max(budgets), tasks)
    projected = _artifact_bytes() + estimate
    hard_limit = int(float(config["data"]["storage_limit_gb"]) * 2**30)
    summary = {
        "tasks": tasks,
        "base_budget": args.base_budget,
        "budgets": budgets,
        "estimated_new_gib": estimate / 2**30,
        "projected_total_gib": projected / 2**30,
        "hard_limit_gib": hard_limit / 2**30,
    }
    if projected > hard_limit:
        raise RuntimeError(f"Oracle extension storage preflight failed: {summary}")
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return
    with workspace_audit("extend_oracle_support") as audit:
        manifest = _extension_manifest(
            config, tasks=tasks, base_budget=args.base_budget, maximum_budget=max(budgets)
        )
        extension = _generate_extension(config, manifest, tasks=tasks, workers=args.workers)
        outputs = _merge_nested_indices(
            extension, tasks=tasks, base_budget=args.base_budget, budgets=budgets
        )
        audit.update({**summary, "manifest": str(manifest), "outputs": [str(v) for v in outputs]})
    print(json.dumps({**summary, "outputs": [str(v) for v in outputs]}, indent=2))


if __name__ == "__main__":
    main()
