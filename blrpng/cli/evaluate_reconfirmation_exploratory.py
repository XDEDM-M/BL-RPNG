from __future__ import annotations

import argparse
import json
import re

import torch

from blrpng.cli.evaluate_reconfirmation import _family, _support_cost
from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import materialize_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.metrics import evaluate_bundle
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a clearly marked post-query exploratory NLT checkpoint."
    )
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--horizon", type=int, choices=(20, 50), default=20)
    parser.add_argument("--physics-config", default="configs/allen_cahn_v2_calibration.yaml")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.tag) is None:
        raise ValueError("Invalid exploratory tag.")
    require_prerequisite_gate(
        f"artifacts/manifests/allen_cahn_reconfirmation_query_"
        f"{args.resolution}_data_gate.json"
    )
    physics = load_config(args.physics_config)
    checkpoint_path = resolve_in_project(
        "artifacts/checkpoints/reconfirmation/"
        f"reconfirmation_task{args.task:02d}_m4_s42_nlt_{args.tag}.adapter.pt",
        must_exist=True,
    )
    query_index = (
        f"artifacts/data/reconfirmation/task{args.task:02d}_query_"
        f"{args.resolution}_all.index.json"
    )
    output_relative = (
        "artifacts/reports/reconfirmation/exploratory/"
        f"task{args.task:02d}_m4_s42_nlt_{args.tag}_"
        f"query{args.resolution}_h{args.horizon}.json"
    )
    with workspace_audit("evaluate_reconfirmation_exploratory") as audit:
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
        metadata = checkpoint["metadata"]
        output = write_json(output_relative, {
            "schema": "blrpng_reconfirmation_post_query_exploratory_v1",
            "seed": 42,
            "task": args.task,
            "family": _family(args.task),
            "support_budget": 4,
            "method": f"nlt_{args.tag}",
            "resolution": args.resolution,
            "horizon": args.horizon,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_seconds": float(metadata.get("elapsed_seconds", 0.0)),
            "support_cost": _support_cost(
                f"artifacts/data/reconfirmation/task{args.task:02d}_support_64_m4.index.json"
            ),
            "metrics": metrics,
            "selection_used_hr_query": False,
            "post_query_exploratory": True,
            "eligible_for_confirmatory_claim": False,
        })
        audit.update({"output": str(output), "tag": args.tag})
    print(json.dumps({
        "output": str(output),
        "primary_rollout_rel_l2": metrics["primary_rollout_rel_l2"],
        "post_query_exploratory": True,
    }, indent=2))


if __name__ == "__main__":
    main()
