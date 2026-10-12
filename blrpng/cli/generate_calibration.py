from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.generate import SFEPY_PYTHON, _case_chunks, workspace_audit
from blrpng.data.manifest import read_manifest
from blrpng.data.shards import subset_index, write_index, write_v4_rollouts_from_arrays
from blrpng.data.tasks import create_calibration_manifest, validate_task_isolation
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, write_json


def _artifact_bytes() -> int:
    return sum(path.stat().st_size for path in (PROJECT_ROOT / "artifacts").rglob("*") if path.is_file())


def _resource_baseline(config: dict) -> dict:
    path = ensure_artifact_path("artifacts/manifests/v2_resource_baseline.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema": "blrpng_resource_baseline_v1",
        "artifact_bytes": _artifact_bytes(),
        "max_new_bytes": int(float(config["resources"]["max_new_artifacts_gb"]) * 1024**3),
    }
    write_json(path, payload)
    return payload


def _check_storage(config: dict, estimated_bytes: int = 0) -> None:
    baseline = _resource_baseline(config)
    new_bytes = _artifact_bytes() - int(baseline["artifact_bytes"])
    if new_bytes + estimated_bytes > int(baseline["max_new_bytes"]):
        raise RuntimeError(
            f"v2 storage gate: used={new_bytes / 1024**3:.2f} GiB, "
            f"estimated={estimated_bytes / 1024**3:.2f} GiB, "
            f"limit={int(baseline['max_new_bytes']) / 1024**3:.2f} GiB"
        )


def _estimate(rollouts: int, steps: int, resolution: int, fields: int = 1) -> int:
    # trajectory + correction + three static fields, plus 35% serialization margin.
    values = rollouts * ((steps + 1) + steps + 3) * fields * resolution**2
    return int(values * 4 * 1.35)


def _generate(
    config: dict,
    manifest_path: Path,
    *,
    split: str,
    resolution: int,
    mesh: int,
    steps: int,
    fidelity: int,
) -> Path:
    stem = f"allen_cahn_v2_{split}_{resolution}"
    index_path = ensure_artifact_path(f"artifacts/data/{stem}.index.json")
    if index_path.exists():
        return index_path
    manifest = read_manifest(manifest_path)
    cases = [row for row in manifest["cases"] if row["split"] == split]
    _check_storage(config, _estimate(len(cases), steps, resolution))
    workers = min(int(os.environ.get("BLRPNG_GENERATION_WORKERS", "1")), len(cases))
    chunks = _case_chunks(len(cases), workers)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    processes = []
    for part, (start, count) in enumerate(chunks):
        npz = ensure_artifact_path(f"artifacts/tmp/{stem}.part{part:03d}.npz")
        meta = ensure_artifact_path(f"artifacts/tmp/{stem}.part{part:03d}.json")
        log = ensure_artifact_path(f"artifacts/tmp/{stem}.part{part:03d}.log")
        command = [
            str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_worker",
            "--manifest", str(manifest_path), "--split", split,
            "--output-npz", str(npz), "--output-meta", str(meta),
            "--output-resolution", str(resolution), "--mesh-resolution", str(mesh),
            "--steps", str(steps), "--fidelity", str(fidelity),
            "--dt", str(config["data"]["dt"]), "--epsilon", str(config["data"]["eps"]),
            "--gamma", str(config["data"]["gamma"]),
            "--progress-threshold", str(config["data"]["residual_progress_threshold"]),
            "--case-start", str(start), "--limit", str(count),
        ]
        stream = log.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True)
        processes.append((process, stream, log, npz, meta, command))
    failures = []
    for process, stream, log, _npz, _meta, command in processes:
        process.wait(); stream.close()
        if process.returncode:
            failures.append(f"{' '.join(command)}\n{log.read_text(encoding='utf-8', errors='replace')}")
    if failures:
        raise RuntimeError("\n\n".join(failures))
    mapping = {int(key): int(value) for key, value in manifest["metadata"]["task_mapping"].items()}
    rows = []
    shard_dir = f"artifacts/data/{stem}"
    for _process, _stream, log, npz, meta_path, _command in processes:
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.extend(write_v4_rollouts_from_arrays(
            arrays, meta, shard_dir=shard_dir, target_task_by_boundary=mapping,
            role=split, source_sha256=sha256_file(manifest_path),
        ))
        npz.unlink(missing_ok=True); meta_path.unlink(missing_ok=True); log.unlink(missing_ok=True)
    rows.sort(key=lambda row: (int(row["target_task_id"]), int(row["initial_condition_id"])))
    output = write_index(index_path, rows, metadata={
        "experiment": config["experiment"]["name"], "split": split,
        "resolution": resolution, "steps": steps,
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256_file(manifest_path), "generation_workers": workers,
    })
    _make_task_indices(config, output)
    return output


def _make_task_indices(config: dict, source: Path) -> None:
    index = json.loads(source.read_text(encoding="utf-8"))
    split = index["metadata"]["split"]
    resolution = int(index["metadata"]["resolution"])
    tasks = int(config["data"]["target_boundaries"])
    for task in range(tasks):
        limits = [None]
        if split == "support_lr": limits = list(config["data"]["support_budgets"])
        if split == "oracle_train": limits = list(config["data"]["oracle_budgets"])
        for limit in limits:
            suffix = "all" if limit is None else f"m{int(limit)}"
            subset_index(
                source,
                f"artifacts/data/tasks/task{task:02d}_{split}_{resolution}_{suffix}.index.json",
                target_task_id=task, max_rollouts=limit,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gated Allen--Cahn v2 target-task shards.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--stage", choices=("manifest", "support", "query128", "query256", "oracle", "frozen512"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with workspace_audit("generate_calibration") as audit:
        manifest = create_calibration_manifest(config)
        audit["manifest_validation"] = validate_task_isolation(read_manifest(manifest))
        outputs = [manifest]
        data = config["data"]
        if args.stage == "support":
            outputs += [
                _generate(config, manifest, split="support_lr", resolution=64, mesh=65, steps=int(data["train_steps"]), fidelity=0),
                _generate(config, manifest, split="validation_lr", resolution=64, mesh=65, steps=int(data["train_steps"]), fidelity=0),
            ]
        elif args.stage in {"query128", "query256"}:
            resolution = int(args.stage[-3:])
            mesh = resolution + 1
            outputs.append(_generate(config, manifest, split="query_hr", resolution=resolution, mesh=mesh, steps=int(data["query_steps"]), fidelity=1))
        elif args.stage == "oracle":
            outputs += [
                _generate(config, manifest, split="oracle_train", resolution=256, mesh=257, steps=int(data["train_steps"]), fidelity=1),
                _generate(config, manifest, split="oracle_validation", resolution=256, mesh=257, steps=int(data["train_steps"]), fidelity=1),
            ]
        elif args.stage == "frozen512":
            gate = ensure_artifact_path("artifacts/manifests/allen_cahn_v2_seed42_gate.json")
            if not gate.exists() or not json.loads(gate.read_text(encoding="utf-8")).get("passed"):
                raise PermissionError("512 generation is locked until the seed42 parity gate passes.")
            outputs.append(_generate(config, manifest, split="query_512", resolution=512, mesh=513, steps=int(data["query_steps"]), fidelity=2))
        audit["outputs"] = [str(path) for path in outputs]
        _check_storage(config)
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
