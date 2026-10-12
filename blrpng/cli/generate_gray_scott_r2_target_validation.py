from __future__ import annotations

import argparse
import json
from pathlib import Path

from blrpng.cli.generate_gray_scott_formal import _base, _run_jobs
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.gray_scott_protocol import validate_gray_scott_protocol
from blrpng.data.shards import write_index
from blrpng.paths import ensure_artifact_path, resolve_in_project


PROTOCOL = "artifacts/manifests/gray_scott_seed42_protocol.json"
DATA_ROOT = "artifacts/data/gray_scott_r2_seed42"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate query-locked 50-step LR validation for Gray--Scott R2."
    )
    parser.add_argument("--config", default="configs/gray_scott_r2_development.yaml")
    parser.add_argument("--tasks", default="0,4,8,12")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    task_ids = [int(value) for value in args.tasks.split(",")]
    if task_ids != [0, 4, 8, 12]:
        raise ValueError("The frozen R2 development tasks are exactly 0,4,8,12.")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8].")

    selection = require_prerequisite_gate(
        "artifacts/manifests/gray_scott_r2_source_selection.json"
    )
    if selection.get("target_query_used") is not False:
        raise PermissionError("R2 source selection is not query locked.")
    require_capacity(1 * 2**30)
    config = load_config(args.config)
    protocol = json.loads(resolve_in_project(PROTOCOL, must_exist=True).read_text())
    validate_gray_scott_protocol(protocol)
    steps = 50
    resolution = 64
    jobs = []
    order = 0
    task_by_id = {int(task["target_task_id"]): task for task in protocol["targets"]}
    for task_id in task_ids:
        task = task_by_id[task_id]
        directory = ensure_artifact_path(
            f"{DATA_ROOT}/task{task_id:02d}_validation_long_{resolution}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for sample in task["validation"]:
            jobs.append({
                **_base(config, resolution=resolution, steps=steps),
                "seed": int(sample["case_id"]),
                "case_id": int(sample["case_id"]),
                "boundary_seed": int(task["boundary_seed"]),
                "initial_seed": int(sample["initial_seed"]),
                "boundary_regime": task["boundary_regime"],
                "correlation": float(task["correlation"]),
                "amplitude": float(task["amplitude"]),
                "feed": float(task["feed"]),
                "kill": float(task["kill"]),
                "role": "target_validation_long",
                "fidelity": 0,
                "target_task_id": task_id,
                "boundary_id": int(task["boundary_id"]),
                "initial_condition_id": int(sample["initial_condition_id"]),
                "pair_id": int(sample["pair_id"]),
                "order": order,
                "output": str(directory / f"pair_{int(sample['pair_id']):07d}.pt"),
            })
            order += 1

    with workspace_audit("generate_gray_scott_r2_target_validation") as audit:
        rows = _run_jobs(jobs, args.workers, "gray_scott_r2_target_validation_long_64")
        outputs = []
        for task_id in task_ids:
            selected = [row for row in rows if int(row["target_task_id"]) == task_id]
            output = write_index(
                f"{DATA_ROOT}/task{task_id:02d}_validation_long_64.index.json",
                selected,
                metadata={
                    "equation": "gray_scott",
                    "condition_channels": 3,
                    "split": "target_validation_long",
                    "resolution": 64,
                    "steps": steps,
                    "target_task_id": task_id,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "query_used": False,
                },
            )
            outputs.append(str(output))
        audit.update({
            "outputs": outputs,
            "tasks": task_ids,
            "query_used": False,
            "workers": args.workers,
        })
    print(json.dumps({
        "outputs": outputs,
        "rollouts": len(rows),
        "steps": steps,
        "resolution": resolution,
        "query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
