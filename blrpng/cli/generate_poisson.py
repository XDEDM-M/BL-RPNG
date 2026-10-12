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
from blrpng.solvers.poisson import generate_poisson_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v4 Poisson negative-control rollouts.")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--family", choices=("fourier", "bump", "piecewise_smooth"), default="fourier")
    parser.add_argument("--role", default="smoke")
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument("--output", default="artifacts/data/poisson_smoke.index.json")
    args = parser.parse_args()
    if args.role != "smoke":
        config = load_config(args.config)
        require_prerequisite_gate(config["experiment"]["prerequisite_gate"])
    rows = []
    require_capacity(args.cases * 8 * args.resolution**2 * 4)
    with workspace_audit("generate_poisson"):
        shard_dir = ensure_artifact_path(args.output).with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        for case in range(args.cases):
            rollout = generate_poisson_rollout(
                resolution=args.resolution, seed=args.seed + case, case_id=case,
                family=args.family, role=args.role,
            )
            path = shard_dir / f"rollout_{case:06d}.pt"
            torch.save(rollout.payload(), path)
            rows.append({
                "path": str(path.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path), "steps": 1,
                "target_task_id": case, "boundary_id": case,
                "initial_condition_id": case, "role": args.role,
                "resolution": args.resolution,
            })
        output = write_index(args.output, rows, metadata={
            "equation": "poisson", "negative_control": True,
            "condition_channels": 1, "seed": args.seed, "family": args.family,
        })
    print(json.dumps({"output": str(output), "cases": args.cases}, indent=2))


if __name__ == "__main__":
    main()
