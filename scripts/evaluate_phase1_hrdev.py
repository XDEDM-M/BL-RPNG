from __future__ import annotations

import argparse
import atexit
import json
import os
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import (
    PROJECT_ROOT,
    exclusive_artifact_lock,
    sha256_file,
    sha256_json,
)
from scripts.run_phase1_trials import _append_ledger


def _write_json_durable(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if path.read_text(encoding="utf-8") != encoded:
        raise OSError(f"HR-development report failed read-back verification: {path}")


def evaluate(task_result_path: Path, task: int, index_path: Path, device: torch.device) -> dict:
    task_result = json.loads(task_result_path.read_text(encoding="utf-8"))
    if int(task_result.get("task", -1)) != task:
        raise ValueError("Task result ID does not match the requested HR development task.")
    checkpoint = str(task_result["checkpoint"])
    if sha256_file(PROJECT_ROOT / checkpoint) != task_result.get("checkpoint_sha256"):
        raise ValueError("LR-selected checkpoint changed before HR development evaluation.")
    index = read_index(index_path, verify_hashes=True)
    metadata = index["metadata"]
    expected_manifest = PROJECT_ROOT / "artifacts/manifests/phase1_development/hrdev_20261012/tasks.json"
    manifest = json.loads(expected_manifest.read_text(encoding="utf-8"))
    manifest_hash = sha256_json({key: value for key, value in manifest.items() if key != "sha256"})
    if manifest.get("sha256") != manifest_hash:
        raise ValueError("HR development task manifest seal is invalid.")
    if metadata.get("split") != "validation_hr":
        raise ValueError("HR development evaluation requires the validation_hr role.")
    if int(metadata.get("resolution", -1)) != 256 or int(metadata.get("steps", -1)) != 20:
        raise ValueError("Expected 256x256, 20-step HR development validation.")
    if int(metadata.get("fidelity", -1)) != 1:
        raise ValueError("HR validation index is not marked high fidelity.")
    if metadata.get("task_manifest_sha256") != manifest_hash:
        raise ValueError("HR validation index uses a different task manifest.")
    task_rows = [row for row in index["shards"] if int(row["target_task_id"]) == task]
    if len(task_rows) != 8:
        raise ValueError(f"Expected eight HR validation rollouts for task {task}, got {len(task_rows)}")
    task_meta = next(
        row for row in manifest["tasks"]
        if int(row["target_task_id"]) == task
    )
    boundary_id = int(task_meta["boundary_id"])
    if any(int(row["boundary_id"]) != boundary_id for row in task_rows):
        raise ValueError("HR development index boundary does not match the frozen task manifest.")

    model, checkpoint_payload = load_checkpoint_model(checkpoint, device=device)
    metrics = projected_rollout_index(
        model,
        str(index_path),
        device=device,
        horizon=20,
        rollout_batch_size=8,
    )
    return {
        "schema": "blrpng_phase1_hrdev_evaluation_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task,
        "family": str(task_meta["family"]),
        "boundary_id": boundary_id,
        "model_trial_id": task_result.get("trial_id"),
        "model_seed": task_result.get("model_seed"),
        "support_seed": task_result.get("support_seed"),
        "support_budget": task_result.get("support_budget"),
        "model_config_sha256": task_result.get("config_sha256"),
        "lr_validation_metric": float(task_result["best_validation_metric"]),
        "task_result": str(task_result_path.relative_to(PROJECT_ROOT)),
        "task_result_sha256": sha256_file(task_result_path),
        "smoke_only": str(task_result.get("trial_id", "")).endswith("-smoke"),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(PROJECT_ROOT / checkpoint),
        "checkpoint_format": checkpoint_payload.get("format"),
        "hr_validation_index": str(index_path.relative_to(PROJECT_ROOT)),
        "hr_validation_index_sha256": index["sha256"],
        "task_manifest_sha256": manifest_hash,
        "hr_validation_rollouts": len(task_rows),
        "hr_resolution": 256,
        "rollout_steps": 20,
        "metrics": metrics,
        "training_or_within_task_checkpoint_selection_used_hr": False,
        "hr_validation_is_development_only": True,
        "query_read": False,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "rollout_module_sha256": sha256_file(PROJECT_ROOT / "blrpng/evaluation/rollout.py"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an R1 LR-selected task checkpoint on independent HR development validation.")
    parser.add_argument("--task-result", required=True, help="R1 runner task00.json result file.")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument(
        "--index",
        help="Override the default 256x256 HR validation index for the task.",
    )
    parser.add_argument("--output", help="Output JSON path; defaults to a unique artifact report.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # HR-dev uses GPU0 too. Share the training lock so an evaluator cannot
    # overlap a trainer or another evaluator, and so output-exists checks are
    # serialized before any report is written.
    gpu_lock = exclusive_artifact_lock("phase1_training_gpu0")
    gpu_lock.__enter__()
    atexit.register(gpu_lock.__exit__, None, None, None)
    task_result_path = PROJECT_ROOT / args.task_result
    index_path = PROJECT_ROOT / (args.index or (
        f"artifacts/data/phase1_development/hrdev_20261012/"
        f"task{args.task:02d}_validation_hr_256.index.json"
    ))
    task_result = json.loads(task_result_path.read_text(encoding="utf-8"))
    trial_id = str(task_result["trial_id"])
    budget_tag = "" if int(task_result["support_budget"]) == 4 else f"_m{task_result['support_budget']}"
    output = PROJECT_ROOT / (
        args.output or f"artifacts/reports/phase1_hrdev_evaluations/"
        f"{trial_id}{budget_tag}_task{args.task:02d}_s{task_result['model_seed']}_ss{task_result['support_seed']}.json"
    )
    if output.exists():
        raise FileExistsError(f"Preserve existing HR-development evidence; choose a new --output: {output}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    report = evaluate(task_result_path, args.task, index_path, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report["timing"] = {
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if device.type == "cuda" else 0.0,
        "scope": "index verification, model loading, and un-warmed HR development evaluation; not a deployment inference benchmark",
        "device": str(device),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    _write_json_durable(output, report)
    _append_ledger({
        "trial_id": f"HRDEV-{trial_id}-task{args.task:02d}-seed{report['model_seed']}-support{report['support_seed']}-{output.stem}",
        "status": "DONE", "round": "HRDEV", "objective": "development_evaluation",
        "seed": report["model_seed"], "task_ids": args.task,
        "config_sha256": report["evaluator_sha256"],
        "elapsed_gpu_hours": report["timing"]["gpu_hours"],
        "validation_metric": report["metrics"]["rollout_rel_l2"],
        "raw_output_path": str(output),
        "notes": "Independent HR development; LR-selected checkpoint; query_read=false; timed code-hashed evaluation",
    })
    print(json.dumps({
        "output": str(output),
        "task": args.task,
        "family": report["family"],
        "lr_validation_rollout_rel_l2": report["lr_validation_metric"],
        "hr_validation_rollout_rel_l2": report["metrics"]["rollout_rel_l2"],
        "query_read": report["query_read"],
    }, indent=2))


if __name__ == "__main__":
    main()
