from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import materialize_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.metrics import evaluate_bundle
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


METHODS = (
    "nlt",
    "ordinary_nlt",
    "nlt_path",
    "nlt_commutator",
    "nlt_commutator_defect",
    "lr_scratch",
)


def _family(task: int) -> str:
    path = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in payload["tasks"] if int(row["target_task_id"]) == task]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen task descriptor for task {task}.")
    return str(matches[0]["family"])


def _checkpoint(task: int, budget: int, method: str) -> Path:
    suffix = "inference.pt" if method == "lr_scratch" else "adapter.pt"
    return resolve_in_project(
        f"artifacts/checkpoints/reconfirmation/"
        f"reconfirmation_task{task:02d}_m{budget}_s42_{method}.{suffix}",
        must_exist=True,
    )


def _support_cost(path: str) -> dict[str, float]:
    index = json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))
    solve = assembly = 0.0
    for row in index["shards"]:
        shard = torch.load(
            resolve_in_project(row["path"], must_exist=True),
            map_location="cpu",
            weights_only=False,
        )
        solve += sum(float(value) for value in shard["meta"].get("solver_timing", []))
        assembly += float(shard["meta"].get("assembly_time", 0.0))
    return {"solver_seconds": solve, "assembly_seconds": assembly}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen reconfirmation checkpoint on HR query data."
    )
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    parser.add_argument("--physics-config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--horizon", type=int, choices=(20, 50), required=True)
    args = parser.parse_args()
    protocol = load_config(args.config)
    physics = load_config(args.physics_config)
    require_prerequisite_gate(protocol["gates"]["required_query_protocol_freeze_manifest"])
    query_index = (
        f"artifacts/data/reconfirmation/task{args.task:02d}_query_"
        f"{args.resolution}_all.index.json"
    )
    support_index = (
        f"artifacts/data/reconfirmation/task{args.task:02d}_support_64_m{args.budget}.index.json"
    )
    checkpoint_path = _checkpoint(args.task, args.budget, args.method)
    output_path = (
        f"artifacts/reports/reconfirmation/task{args.task:02d}_m{args.budget}_s42_"
        f"{args.method}_query{args.resolution}_h{args.horizon}.json"
    )
    with workspace_audit("evaluate_reconfirmation") as audit:
        bundle = materialize_index(query_index)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
        metrics = evaluate_bundle(
            model,
            bundle,
            device=device,
            horizon=args.horizon,
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
            output_path,
            {
                "schema": "blrpng_reconfirmation_query_evaluation_v1",
                "seed": 42,
                "task": args.task,
                "family": _family(args.task),
                "support_budget": args.budget,
                "method": args.method,
                "resolution": args.resolution,
                "horizon": args.horizon,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_seconds": float(metadata.get("elapsed_seconds", 0.0)),
                "support_cost": _support_cost(support_index),
                "metrics": metrics,
                "selection_used_hr_query": False,
            },
        )
        audit.update({"output": str(output), "task": args.task, "method": args.method})
    print(json.dumps({
        "output": str(output),
        "primary_rollout_rel_l2": metrics["primary_rollout_rel_l2"],
    }, indent=2))


if __name__ == "__main__":
    main()
