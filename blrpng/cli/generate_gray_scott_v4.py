from __future__ import annotations

import argparse
import json

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.capacity import require_capacity
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.config import load_config
from blrpng.data.shards import write_index
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file
from blrpng.solvers.gray_scott import generate_gray_scott_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate schema-v4 Gray--Scott rollouts.")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--correlation", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/data/gray_scott_v4_smoke.index.json")
    parser.add_argument("--role", default="smoke")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    args = parser.parse_args()
    if args.role != "smoke":
        config = load_config(args.config)
        require_prerequisite_gate(config["experiment"]["prerequisite_gate"])
    rows = []
    require_capacity(args.cases * (args.steps + 4) * 2 * args.resolution**2 * 4)
    with workspace_audit("generate_gray_scott_v4"):
        shard_dir = ensure_artifact_path(args.output).with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        for case in range(args.cases):
            rollout = generate_gray_scott_rollout(
                resolution=args.resolution, steps=args.steps, seed=args.seed + case,
                case_id=case, correlation=args.correlation,
            )
            path = shard_dir / f"rollout_{case:06d}.pt"
            torch.save(rollout.payload(), path)
            rows.append({
                "path": str(path.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path), "steps": args.steps,
                "target_task_id": case, "boundary_id": case,
                "initial_condition_id": case, "role": args.role,
                "resolution": args.resolution,
            })
        output = write_index(args.output, rows, metadata={
            "equation": "gray_scott", "schema_version": 4,
            "condition_channels": 3, "seed": args.seed,
        })
    print(json.dumps({"output": str(output), "cases": args.cases}, indent=2))


if __name__ == "__main__":
    main()
