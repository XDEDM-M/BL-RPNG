from __future__ import annotations

import argparse
import json
import time

import torch

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import resolve_in_project, sha256_file, write_json


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Gray--Scott query shards.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 <= args.task_start < args.task_end <= 16:
        raise ValueError("Task interval must satisfy 0 <= start < end <= 16.")
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if any(method not in METHODS for method in methods):
        raise ValueError(f"Unsupported method set: {methods}")
    require_prerequisite_gate(
        f"artifacts/manifests/gray_scott_seed42_query{args.resolution}_data_gate.json"
    )
    freeze_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_lr_repair_summary.json",
        must_exist=True,
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if tuple(freeze["precommitted_hr_methods"]) != METHODS:
        raise RuntimeError("Frozen HR method list does not match the evaluator.")

    device = torch.device(args.device)
    records = []
    model_reports = []
    audit_name = f"evaluate_gray_scott_query{args.resolution}_{args.task_start}_{args.task_end}"
    with workspace_audit(audit_name) as audit:
        for task in range(args.task_start, args.task_end):
            index_path = (
                f"artifacts/data/gray_scott_seed42/"
                f"task{task:02d}_query_{args.resolution}_all.index.json"
            )
            index = read_index(index_path, verify_hashes=True)
            boundary_id = int(index["shards"][0]["boundary_id"])
            family = str(index["metadata"]["boundary_regime"])
            for method in methods:
                checkpoint_path = (
                    f"artifacts/checkpoints/gray_scott/"
                    f"gray_scott_task{task:02d}_m4_s42_{method}.adapter.pt"
                )
                expected_hash = freeze["adapter_sha256"][method][task]
                actual_hash = sha256_file(checkpoint_path)
                if actual_hash != expected_hash:
                    raise RuntimeError(f"Frozen adapter hash mismatch: {checkpoint_path}")
                model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
                training_seconds = float(checkpoint["metadata"].get("elapsed_seconds", 0.0))
                for horizon in (20, 50):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    metrics = projected_rollout_index(
                        model, index_path, device=device, horizon=horizon,
                        rollout_batch_size=8,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    if len(metrics["initial_condition_ids"]) != len(metrics["per_rollout_rel_l2"]):
                        raise RuntimeError("Query metrics lost rollout pairing.")
                    for initial_id, error in zip(
                        metrics["initial_condition_ids"], metrics["per_rollout_rel_l2"]
                    ):
                        records.append({
                            "seed": 42,
                            "target_task_id": task,
                            "boundary_id": boundary_id,
                            "initial_condition_id": int(initial_id),
                            "boundary_regime": family,
                            "method": method,
                            "relative_l2": float(error),
                            "rollout_mse": float(metrics["rollout_mse"]),
                            "resolution": args.resolution,
                            "horizon": horizon,
                            "training_seconds": training_seconds,
                            "evaluation_seconds_task": elapsed,
                            "inference_seconds_per_query": elapsed / len(index["shards"]),
                        })
                    model_reports.append({
                        "task": task, "method": method, "horizon": horizon,
                        "checkpoint": checkpoint_path, "checkpoint_sha256": actual_hash,
                        "mean_relative_l2": float(metrics["rollout_rel_l2"]),
                        "evaluation_seconds": elapsed,
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        output = write_json(
            args.output or (
                f"artifacts/reports/gray_scott/query{args.resolution}_"
                f"tasks{args.task_start:02d}_{args.task_end:02d}.json"
            ),
            {
                "schema": "blrpng_gray_scott_query_part_v1",
                "frozen": True,
                "selection_used_hr_query": False,
                "resolution": args.resolution,
                "task_start": args.task_start,
                "task_end": args.task_end,
                "methods": list(methods),
                "records": records,
                "model_reports": model_reports,
            },
        )
        audit.update({"output": str(output), "records": len(records)})
    means = {
        f"{method}_h{horizon}": sum(
            row["relative_l2"] for row in records
            if row["method"] == method and row["horizon"] == horizon
        ) / sum(
            row["method"] == method and row["horizon"] == horizon for row in records
        )
        for method in methods for horizon in (20, 50)
    }
    print(json.dumps({"output": str(output), "means": means}, indent=2))


if __name__ == "__main__":
    main()
