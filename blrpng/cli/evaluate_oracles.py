from __future__ import annotations

import argparse
import json
import time

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate each task's frozen saturated HR specialist.")
    parser.add_argument("--curves", default="artifacts/reports/v2_oracle_curves.json")
    parser.add_argument("--resolution", type=int, choices=(128, 256), default=256)
    parser.add_argument("--horizon", type=int, choices=(20, 50), default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/reports/v2_seed42_oracle_query256_h20.json")
    args = parser.parse_args()
    curves = json.loads(resolve_in_project(args.curves, must_exist=True).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    with workspace_audit("evaluate_oracles"):
        for task_text, curve in sorted(curves["tasks"].items(), key=lambda item: int(item[0])):
            task = int(task_text)
            budget = int(curve["selected_budget"])
            checkpoint = f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{args.seed}_hr_oracle.pt"
            index = f"artifacts/data/tasks/task{task:02d}_query_hr_{args.resolution}_all.index.json"
            index_payload = read_index(index)
            boundary_id = int(index_payload["shards"][0]["boundary_id"])
            model, payload = load_checkpoint_model(checkpoint, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = projected_rollout_index(model, index, device=device, horizon=args.horizon)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            initial_ids = metrics["initial_condition_ids"]
            training_seconds = float(payload.get("elapsed_seconds", 0.0))
            for initial_id, error in zip(initial_ids, metrics["per_rollout_rel_l2"]):
                records.append({
                    "seed": args.seed, "target_task_id": task,
                    "boundary_id": boundary_id, "initial_condition_id": initial_id,
                    "method": "hr_oracle", "relative_l2": float(error),
                    "resolution": args.resolution, "horizon": args.horizon,
                    "selected_budget": budget, "training_seconds": training_seconds,
                    "evaluation_seconds_task": elapsed,
                    "inference_seconds_per_query": elapsed / max(len(initial_ids), 1),
                })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        output = write_json(args.output, {
            "schema": "blrpng_oracle_query_v1", "frozen": True,
            "resolution": args.resolution, "horizon": args.horizon,
            "seed": args.seed, "records": records,
        })
    print(json.dumps({
        "output": str(output),
        "mean_relative_l2": sum(row["relative_l2"] for row in records) / max(len(records), 1),
    }, indent=2))


if __name__ == "__main__":
    main()
