from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.data.blind_boundaries import FAMILIES, sample_boundary_descriptor
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_json


def freeze(campaign: str, seed: int, tasks_per_family: int, boundary_id_offset: int) -> Path:
    if not campaign.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Campaign ID may contain only letters, digits, '_' and '-'.")
    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be positive.")
    tasks = []
    boundary_id = int(boundary_id_offset)
    for family in FAMILIES:
        for family_task_id in range(tasks_per_family):
            boundary = sample_boundary_descriptor(
                family, boundary_id=boundary_id, seed=int(seed)
            )
            tasks.append({
                "target_task_id": len(tasks),
                "family": family,
                "family_task_id": family_task_id,
                "boundary_id": boundary_id,
                "boundary": boundary,
            })
            boundary_id += 1
    payload = {
        "schema": "blrpng_phase1_confirmation_tasks_v1",
        "campaign_id": campaign,
        "seed": int(seed),
        "families": list(FAMILIES),
        "tasks_per_family": int(tasks_per_family),
        "tasks": tasks,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "confirmation_status": "boundaries_frozen_rollouts_not_generated",
        "query_policy": "Generate independent LR support/validation first. Generate sealed HR query rollouts only after all confirmation checkpoints and analysis code are frozen.",
    }
    payload["sha256"] = sha256_json(payload)
    output = ensure_artifact_path(
        PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "tasks.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze independent unseen-boundary tasks for the Phase-1 confirmatory campaign."
    )
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tasks-per-family", type=int, default=8)
    parser.add_argument("--boundary-id-offset", type=int, default=6_000_000)
    args = parser.parse_args()
    output = freeze(args.campaign, args.seed, args.tasks_per_family, args.boundary_id_offset)
    print(f"frozen confirmation task manifest: {output}")


if __name__ == "__main__":
    main()
