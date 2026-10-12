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


TASKS = (0, 4, 8, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pre-frozen saturated LR scratch baseline.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require_prerequisite_gate(
        "artifacts/manifests/gray_scott_r2_blind_query_data_gate.json"
    )
    frozen = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_saturated_scratch_freeze.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    if not frozen.get("frozen") or frozen.get("query_used") is not False:
        raise PermissionError("Saturated LR scratch was not frozen before query generation.")
    task_frozen = {int(row["task"]): row for row in frozen["tasks"]}
    device = torch.device(args.device)
    records = []
    reports = []
    with workspace_audit(f"evaluate_gray_scott_r2_scratch{args.resolution}") as audit:
        for task in TASKS:
            spec = task_frozen[task]
            checkpoint = spec["selected_path"]
            if sha256_file(checkpoint) != spec["selected_sha256"]:
                raise RuntimeError("Pre-frozen saturated scratch hash mismatch.")
            index_path = (
                "artifacts/data/gray_scott_r2_seed42/blind_query/"
                f"task{task:02d}_query_{args.resolution}.index.json"
            )
            index = read_index(index_path, verify_hashes=True)
            model, _ = load_checkpoint_model(checkpoint, device=device)
            for horizon in (20, 50):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                metrics = projected_rollout_index(
                    model, index_path, device=device, horizon=horizon,
                    rollout_batch_size=4,
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
                        "boundary_id": int(index["shards"][0]["boundary_id"]),
                        "initial_condition_id": int(initial_id),
                        "boundary_regime": index["metadata"]["boundary_regime"],
                        "method": "lifted_lr_scratch_saturated",
                        "relative_l2": float(error),
                        "resolution": args.resolution,
                        "horizon": horizon,
                    })
                reports.append({
                    "task": task,
                    "horizon": horizon,
                    "mean_relative_l2": metrics["rollout_rel_l2"],
                    "evaluation_seconds": elapsed,
                    "checkpoint_sha256": spec["selected_sha256"],
                    "training_seconds": spec["total_training_seconds"],
                })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        output = write_json(
            f"artifacts/reports/gray_scott/r2_blind_query{args.resolution}_scratch.json",
            {
                "schema": "blrpng_gray_scott_r2_prefrozen_scratch_evaluation_v1",
                "frozen": True,
                "resolution": args.resolution,
                "baseline_frozen_before_r2_query": True,
                "input_feature_mode": "raw",
                "records": records,
                "model_reports": reports,
            },
        )
        audit.update({"output": str(output), "records": len(records)})
    means = {
        f"h{horizon}": sum(row["relative_l2"] for row in records if row["horizon"] == horizon) / 32
        for horizon in (20, 50)
    }
    print(json.dumps({"output": str(output), "means": means}, indent=2))


if __name__ == "__main__":
    main()
