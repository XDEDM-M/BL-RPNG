from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import materialize_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.metrics import evaluate_bundle
from blrpng.paths import ensure_artifact_path, sha256_file, write_json


def _support_cost(index_path: str) -> dict[str, float]:
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    solve = assembly = 0.0
    for row in index["shards"]:
        shard = torch.load(row["path"], map_location="cpu", weights_only=False)
        solve += sum(float(value) for value in shard["meta"].get("solver_timing", []))
        assembly += float(shard["meta"].get("assembly_time", 0.0))
    return {"solver_seconds": solve, "assembly_seconds": assembly}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one adapted task on frozen HR query shards.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--support-budget", type=int, required=True)
    parser.add_argument("--resolution", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--horizon", type=int, choices=(20, 50), default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    split = "query_512" if args.resolution == 512 else "query_hr"
    index_path = f"artifacts/data/tasks/task{args.task:02d}_{split}_{args.resolution}_all.index.json"
    support_path = f"artifacts/data/tasks/task{args.task:02d}_support_lr_64_m{args.support_budget}.index.json"
    with workspace_audit("evaluate_task") as audit:
        bundle = materialize_index(index_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint = load_checkpoint_model(args.checkpoint, device=device)
        metrics = evaluate_bundle(
            model, bundle, device=device, horizon=args.horizon,
            near_boundary_width=4,
            equation={"dt": config["data"]["dt"], "epsilon": config["data"]["eps"], "gamma": config["data"]["gamma"]},
            coarse_initial_resolution=64,
        )
        support_cost = _support_cost(support_path) if Path(support_path).exists() else None
        output = write_json(args.output, {
            "schema": "blrpng_target_evaluation_v1", "task": args.task,
            "support_budget": args.support_budget, "resolution": args.resolution,
            "horizon": args.horizon, "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "training_seconds": float(checkpoint.get("elapsed_seconds", 0.0)),
            "support_cost": support_cost, "metrics": metrics,
        })
        audit["output"] = str(output)
    print(json.dumps({"output": str(output), "primary": metrics["primary_rollout_rel_l2"]}, indent=2))


if __name__ == "__main__":
    main()
