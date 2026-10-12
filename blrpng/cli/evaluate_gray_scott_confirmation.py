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


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated", "lifted_scratch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the unopened Gray--Scott confirmation query.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require_prerequisite_gate(
        f"artifacts/manifests/gray_scott_seed42_confirmation{args.resolution}_data_gate.json"
    )
    transfer = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_converged_lr_summary.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    scratch = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_saturated_scratch_freeze.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    records = []
    model_reports = []
    with workspace_audit(f"evaluate_gray_scott_confirmation{args.resolution}") as audit:
        for task in range(16):
            index_path = (
                "artifacts/data/gray_scott_seed42_confirmation/"
                f"task{task:02d}_query_{args.resolution}_all.index.json"
            )
            index = read_index(index_path, verify_hashes=True)
            boundary_id = int(index["shards"][0]["boundary_id"])
            family = str(index["metadata"]["boundary_regime"])
            for method in METHODS:
                if method == "lifted_scratch":
                    frozen = scratch["tasks"][task]
                    checkpoint_path = frozen["selected_path"]
                    expected_hash = frozen["selected_sha256"]
                    training_seconds = float(frozen["total_training_seconds"])
                else:
                    checkpoint_path = (
                        f"artifacts/checkpoints/gray_scott/gray_scott_task{task:02d}_"
                        f"m4_s42_{method}_e100.adapter.pt"
                    )
                    expected_hash = transfer["adapter_sha256"][method][task]
                    training_seconds = 0.0
                if sha256_file(checkpoint_path) != expected_hash:
                    raise RuntimeError(f"Frozen checkpoint hash mismatch: {checkpoint_path}")
                model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
                if method != "lifted_scratch":
                    training_seconds = float(checkpoint["metadata"]["elapsed_seconds"])
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
                            "resolution": args.resolution,
                            "horizon": horizon,
                            "training_seconds": training_seconds,
                            "evaluation_seconds_task": elapsed,
                            "inference_seconds_per_query": elapsed / len(index["shards"]),
                        })
                    model_reports.append({
                        "task": task, "method": method, "horizon": horizon,
                        "checkpoint": checkpoint_path,
                        "checkpoint_sha256": expected_hash,
                        "mean_relative_l2": float(metrics["rollout_rel_l2"]),
                        "rollout_mse": float(metrics["rollout_mse"]),
                        "evaluation_seconds": elapsed,
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        output = write_json(
            f"artifacts/reports/gray_scott/confirmation{args.resolution}.json",
            {
                "schema": "blrpng_gray_scott_confirmation_evaluation_v1",
                "frozen": True,
                "selection_used_confirmation_query": False,
                "resolution": args.resolution,
                "methods": list(METHODS),
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
        for method in METHODS for horizon in (20, 50)
    }
    print(json.dumps({"output": str(output), "means": means}, indent=2))


if __name__ == "__main__":
    main()
