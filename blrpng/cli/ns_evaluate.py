"""Evaluate a frozen SfePy-NS checkpoint/adapter without model selection."""
from __future__ import annotations

import argparse
import json

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.multiphysics import evaluate_multiphysics_index
from blrpng.paths import sha256_file, write_json
from blrpng.training.adapters import load_adapter_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navier_stokes_sfepy_seed42.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--horizon", type=int, choices=(20, 50), default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_prerequisite_gate("artifacts/manifests/navier_stokes_sfepy_method_freeze.json")
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") == "blrpng_adapter_delta_v1":
        model, _ = load_adapter_delta(args.checkpoint, device=device)
    else:
        model, _ = load_checkpoint_model(args.checkpoint, device=device)
    with workspace_audit("ns_evaluate") as audit:
        metrics = evaluate_multiphysics_index(
            model,
            args.index,
            device=device,
            equation="navier_stokes",
            equation_parameters=config["data"],
            horizon=args.horizon,
        )
        output = write_json(
            args.output,
            {
                "schema": "blrpng_ns_sfepy_evaluation_v1",
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "frozen_index": args.index,
                "frozen_index_sha256": sha256_file(args.index),
                "horizon": args.horizon,
                "metrics": metrics,
                "checkpoint_selection_performed": False,
            },
        )
        audit.update({"output": str(output), "horizon": args.horizon})
    print(
        json.dumps(
            {
                "output": str(output),
                "primary_velocity_relative_l2": metrics[
                    "primary_velocity_rollout_relative_l2"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
