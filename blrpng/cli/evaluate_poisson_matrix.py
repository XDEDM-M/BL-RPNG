from __future__ import annotations

import argparse
import json
import time

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.multiphysics import evaluate_multiphysics_index
from blrpng.paths import sha256_file, write_json


def _checkpoint(task: int, method: str) -> str:
    return (
        f"artifacts/checkpoints/poisson/"
        f"poisson_task{task:02d}_m4_s42_{method}.adapter.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Poisson NLT matrix.")
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument("--method", choices=("ordinary_nlt", "path_nlt"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    require_prerequisite_gate(
        "artifacts/manifests/poisson_negative_control_query_data_gate.json"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluations = []
    with workspace_audit(f"evaluate_poisson_matrix_{args.method}") as audit:
        for task in range(30):
            checkpoint = _checkpoint(task, args.method)
            model, payload = load_checkpoint_model(checkpoint, device=device)
            for resolution in (128, 256):
                index = (
                    f"artifacts/data/poisson_negative_control/"
                    f"task{task:02d}_query_{resolution}_all.index.json"
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                metrics = evaluate_multiphysics_index(
                    model, index, device=device, equation="poisson",
                    equation_parameters=config["data"], horizon=1,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                evaluations.append({
                    "task": task,
                    "method": args.method,
                    "seed": args.seed,
                    "resolution": resolution,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "inference_seconds": time.perf_counter() - started,
                    "adaptation_seconds": float(
                        payload.get("metadata", {}).get("elapsed_seconds", 0.0)
                    ),
                    "metrics": metrics,
                })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        output = write_json(
            f"artifacts/experiments/poisson_negative_control/"
            f"seed42_{args.method}_frozen_query.json",
            {
                "schema": "blrpng_poisson_matrix_evaluation_v1",
                "frozen": True,
                "query_used_for_checkpoint_selection": False,
                "method": args.method,
                "seed": args.seed,
                "evaluations": evaluations,
            },
        )
        audit.update({"output": str(output), "evaluations": len(evaluations)})
    means = {
        resolution: sum(
            row["metrics"]["primary_rollout_relative_l2"]
            for row in evaluations if row["resolution"] == resolution
        ) / 30
        for resolution in (128, 256)
    }
    print(json.dumps({
        "output": str(output), "method": args.method,
        "mean_relative_l2": means,
    }, indent=2))


if __name__ == "__main__":
    main()
