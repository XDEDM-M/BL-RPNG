from __future__ import annotations

import argparse
import json

import torch

from blrpng.cli.evaluate_reconfirmation import (
    METHODS,
    _checkpoint,
    _family,
    _support_cost,
)
from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import materialize_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.metrics import evaluate_bundle
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def _specifications() -> list[tuple[int, str, int]]:
    # Primary causal matrix at m4/h20, nested NLT support-size curve, and the
    # predeclared long-horizon endpoint.  Duplicate m4/NLT/h20 is included only
    # once by construction.
    specifications = [(4, method, 20) for method in METHODS]
    specifications.extend((budget, "nlt", 20) for budget in (1, 2, 8))
    specifications.append((4, "nlt", 50))
    return specifications


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen reconfirmation matrix with one query load per task."
    )
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    parser.add_argument("--physics-config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    args = parser.parse_args()
    protocol = load_config(args.config)
    physics = load_config(args.physics_config)
    require_prerequisite_gate(
        protocol["gates"]["required_query_protocol_freeze_manifest"]
    )
    require_prerequisite_gate(
        f"artifacts/manifests/allen_cahn_reconfirmation_query_"
        f"{args.resolution}_data_gate.json"
    )
    task_count = len(protocol["blind_protocol"]["families"]) * int(
        protocol["blind_protocol"]["tasks_per_family"]
    )
    if not 0 <= args.task < task_count:
        raise ValueError(f"task must be in [0, {task_count}).")
    query_index = (
        f"artifacts/data/reconfirmation/task{args.task:02d}_query_"
        f"{args.resolution}_all.index.json"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    with workspace_audit(
        f"evaluate_reconfirmation_matrix_{args.resolution}_task{args.task:02d}"
    ) as audit:
        bundle = materialize_index(query_index)
        support_costs = {
            budget: _support_cost(
                f"artifacts/data/reconfirmation/task{args.task:02d}_"
                f"support_64_m{budget}.index.json"
            )
            for budget in (1, 2, 4, 8)
        }
        for budget, method, horizon in _specifications():
            output_relative = (
                "artifacts/reports/reconfirmation/"
                f"task{args.task:02d}_m{budget}_s42_{method}_"
                f"query{args.resolution}_h{horizon}.json"
            )
            output_path = resolve_in_project(output_relative)
            if output_path.exists():
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                outputs.append({
                    "path": output_relative,
                    "method": method,
                    "budget": budget,
                    "horizon": horizon,
                    "relative_l2": float(
                        existing["metrics"]["primary_rollout_rel_l2"]
                    ),
                    "reused": True,
                })
                continue
            checkpoint_path = _checkpoint(args.task, budget, method)
            model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
            metrics = evaluate_bundle(
                model,
                bundle,
                device=device,
                horizon=horizon,
                near_boundary_width=4,
                equation={
                    "dt": physics["data"]["dt"],
                    "epsilon": physics["data"]["eps"],
                    "gamma": physics["data"]["gamma"],
                },
                coarse_initial_resolution=64,
            )
            metadata = checkpoint.get("metadata", checkpoint)
            output = write_json(
                output_relative,
                {
                    "schema": "blrpng_reconfirmation_query_evaluation_v1",
                    "seed": 42,
                    "task": args.task,
                    "family": _family(args.task),
                    "support_budget": budget,
                    "method": method,
                    "resolution": args.resolution,
                    "horizon": horizon,
                    "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "training_seconds": float(metadata.get("elapsed_seconds", 0.0)),
                    "support_cost": support_costs[budget],
                    "metrics": metrics,
                    "selection_used_hr_query": False,
                    "matrix_loader_reused_query_bundle": True,
                },
            )
            outputs.append({
                "path": str(output.relative_to(PROJECT_ROOT)),
                "method": method,
                "budget": budget,
                "horizon": horizon,
                "relative_l2": metrics["primary_rollout_rel_l2"],
                "reused": False,
            })
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        audit.update({
            "task": args.task,
            "resolution": args.resolution,
            "evaluations": len(outputs),
        })
    print(json.dumps({
        "task": args.task,
        "resolution": args.resolution,
        "evaluations": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
