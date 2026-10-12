from __future__ import annotations

import argparse
import json
import time

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import PROJECT_ROOT, write_json


def _checkpoint(task: int, method: str, budget: int, seed: int) -> str:
    suffix = "adapter.pt" if method in {
        "lora", "nlt", "ordinary_lora", "ordinary_nlt", "nlt_path",
        "nlt_commutator", "nlt_commutator_defect", "blrpng_v2",
        "blrpng_plus", "blrpng_plus_d1e3", "blrpng_plus_d1e2",
        "blrpng_v2_nlt", "blrpng_plus_nlt",
    } else "pt"
    return f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{seed}_{method}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--horizon", type=int, choices=(20, 50), default=20)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default="lr_scratch,ordinary_final,pathpre_final,pathpre_fixed,lora,nlt,blrpng_v2,blrpng_plus")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    split = "query_512" if args.resolution == 512 else "query_hr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    with workspace_audit("quick_evaluate_matrix"):
        for task in range(8):
            index = f"artifacts/data/tasks/task{task:02d}_{split}_{args.resolution}_all.index.json"
            index_payload = read_index(index)
            boundary_id = int(index_payload["shards"][0]["boundary_id"])
            for method in methods:
                checkpoint = _checkpoint(task, method, args.budget, args.seed)
                model, payload = load_checkpoint_model(checkpoint, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                metrics = projected_rollout_index(model, index, device=device, horizon=args.horizon)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                initial_ids = metrics["initial_condition_ids"]
                record_start = len(records)
                for initial_id, error in zip(initial_ids, metrics["per_rollout_rel_l2"]):
                    records.append({
                        "seed": args.seed, "target_task_id": task,
                        "boundary_id": boundary_id, "initial_condition_id": initial_id,
                        "method": method, "relative_l2": error,
                        "resolution": args.resolution, "horizon": args.horizon,
                    })
                training_seconds = float(payload.get("elapsed_seconds", payload.get("metadata", {}).get("elapsed_seconds", 0.0)))
                for row in records[record_start:]:
                    row["evaluation_seconds_task"] = elapsed
                    row["inference_seconds_per_query"] = elapsed / max(len(initial_ids), 1)
                    row["training_seconds"] = training_seconds
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
        output = write_json(args.output, {
            "schema": "blrpng_quick_matrix_v1", "frozen": True,
            "resolution": args.resolution, "horizon": args.horizon,
            "budget": args.budget, "seed": args.seed, "records": records,
        })
    means = {method: sum(r["relative_l2"] for r in records if r["method"] == method) / sum(r["method"] == method for r in records) for method in methods}
    print(json.dumps({"output": str(output), "means": means}, indent=2))


if __name__ == "__main__":
    main()
