"""Reproduce the bounded two-boundary steady SfePy-NS continuation pilot."""
from __future__ import annotations

import argparse
import copy
import json

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import build_navier_stokes_protocol
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.data.ns_projection import project_steady_shard
from blrpng.data.shards import validate_shard
from blrpng.paths import resolve_in_project, sha256_file


CASE_IDS = {
    (0, 64): 99092150,
    (0, 128): 99092170,
    (0, 256): 99092202,
    (6, 64): 99092156,
    (6, 128): 99092176,
    (6, 256): 99092206,
}
RAW_FILENAMES = {
    (0, 256): "task00_re400_256_memory_fixed.pt",
}
PAIR_IDS = {0: 99092150, 6: 99092156}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, nargs="+", choices=(0, 6), default=(0, 6))
    parser.add_argument("--resolutions", type=int, nargs="+", choices=(64, 128, 256), default=(64, 128, 256))
    parser.add_argument("--predictor-blend", type=float, choices=(0.5, 1.0), default=1.0,
                        help="1.0 is the natural previous-solution warm start; 0.5 reproduces the pilot ablation")
    args = parser.parse_args()
    tasks = tuple(sorted(set(args.tasks)))
    resolutions = tuple(sorted(set(args.resolutions)))
    config = load_config("configs/navier_stokes_sfepy_seed42.yaml")
    config["solver"]["path_selection_mode"] = "first_accepted"
    config["resources"]["max_generation_workers"] = 1
    config["resources"]["sfepy_worker_address_space_gib"] = 8.0
    config["resources"]["sfepy_case_timeout_seconds"] = 180.0
    protocol = build_navier_stokes_protocol(config)
    rows = []
    with workspace_audit("ns_generate_steady_pilot"):
        require_capacity(estimated_new_bytes=1_000_000_000, maximum_gb=300.0)
        for resolution in resolutions:
            mesh_resolution = resolution // 2 + 1
            for task in tasks:
                case = copy.deepcopy(protocol["targets"][str(task)]["support"][0])
                natural = args.predictor_blend == 1.0
                case_id = (99092300 + resolution + task) if natural else CASE_IDS[(task, resolution)]
                pair_id = (99092300 + task) if natural else PAIR_IDS[task]
                case.update({
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "rollout_id": case_id,
                    "role": (
                        "steady_natural_warm_start_sensitivity" if natural else
                        "steady_re400_extension64" if resolution == 64 else
                        "steady_re400_extension128" if resolution == 128 else
                        "steady_re400_reference256_memory_fixed"
                    ),
                    "reynolds_schedule": [50, 100, 200, 400],
                    "newton_predictor_blend": args.predictor_blend,
                    "grad_div_gamma": 100.0,
                    "forcing": {"active": False},
                })
                raw_name = (
                    f"task{task:02d}_warmstart_{resolution}.pt" if natural else
                    RAW_FILENAMES.get((task, resolution), f"task{task:02d}_re400_{resolution}.pt")
                )
                raw_path = f"artifacts/data/navier_stokes_sfepy_steady_pilot/{raw_name}"
                projected_name = (
                    f"task{task:02d}_warmstart_{resolution}.pt" if natural else
                    f"task{task:02d}_re400_{resolution}.pt"
                )
                projected_path = f"artifacts/data/navier_stokes_sfepy_steady_projected/{projected_name}"
                row = run_sfepy_navier_stokes_case(
                    case,
                    output_shard=raw_path,
                    output_resolution=resolution,
                    mesh_resolution=mesh_resolution,
                    steps=4,
                    dt=1.0,
                    config=config,
                    fidelity={64: 0, 128: 1, 256: 2}[resolution],
                    time_integrator="STEADY_CONTINUATION",
                )
                output = resolve_in_project(projected_path)
                if output.exists():
                    payload = torch.load(output, map_location="cpu", weights_only=False)
                    validate_shard(payload)
                    if payload["meta"]["grid_projection"]["source_sha256"] != row["sha256"]:
                        raise ValueError(f"Existing projected shard does not match raw input: {output}")
                    projected = {"output": str(output), "output_sha256": sha256_file(output), "reused": True}
                else:
                    projected = project_steady_shard(raw_path, projected_path)
                    projected["reused"] = False
                summary = {"task": task, "resolution": resolution, "raw": row, "projected": projected}
                rows.append(summary)
                print(json.dumps(summary, sort_keys=True), flush=True)
    print(json.dumps({"generated_or_reused": len(rows), "tasks": tasks, "resolutions": resolutions,
                      "predictor_blend": args.predictor_blend}, sort_keys=True))


if __name__ == "__main__":
    main()
