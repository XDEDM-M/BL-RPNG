from __future__ import annotations

import json

import numpy as np

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_json, write_json


TASKS = (0, 4, 8, 12)


def main() -> None:
    selection = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_target_selection.json", must_exist=True
    ).read_text(encoding="utf-8"))
    if not selection.get("frozen") or not selection.get("new_query_authorized"):
        raise PermissionError("R2 target selection does not authorize a blind query.")
    if selection.get("selection_used_high_resolution_query") is not False:
        raise PermissionError("R2 method selection was not high-resolution-query blind.")
    old = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    old_tasks = {int(row["target_task_id"]): row for row in old["targets"]}
    used_initials = {
        int(sample["initial_condition_id"])
        for task in old["targets"]
        for split in ("support", "validation", "query")
        for sample in task[split]
    }
    rng = np.random.default_rng(20260905)
    targets = []
    for task_id in TASKS:
        source = old_tasks[task_id]
        samples = []
        for local in range(8):
            initial_id = 900_000 + task_id * 100 + local
            pair_id = 9_000_000 + task_id * 100 + local
            if initial_id in used_initials:
                raise RuntimeError("Fresh R2 query initial condition leaked into an old split.")
            samples.append({
                "case_id": pair_id,
                "pair_id": pair_id,
                "initial_condition_id": initial_id,
                "initial_seed": int(rng.integers(0, 2**31 - 1)),
                "role": "r2_blind_query",
            })
        targets.append({
            key: source[key]
            for key in (
                "target_task_id", "boundary_id", "boundary_seed", "boundary_regime",
                "correlation", "amplitude", "feed", "kill",
            )
        } | {"query": samples})
    protocol = {
        "schema": "blrpng_gray_scott_r2_blind_query_protocol_v1",
        "frozen": True,
        "seed": 42,
        "split_seed": 20260905,
        "tasks": list(TASKS),
        "query_rollouts_per_task": 8,
        "query_resolutions": [128, 256],
        "query_horizon": 50,
        "evaluation_horizons": [20, 50],
        "selected_method": selection["selected_method"],
        "methods": [
            "final_nlt", "path_nlt", "final_nlt_multiscale", "path_nlt_multiscale"
        ],
        "selection_manifest_sha256": sha256_json(selection),
        "selection_used_query": False,
        "targets": targets,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    with workspace_audit("freeze_gray_scott_r2_query_protocol") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_blind_query_protocol.json", protocol
        )
        audit.update({"output": str(output), "query_opened": False})
    print(json.dumps({"output": str(output), **protocol}, indent=2))


if __name__ == "__main__":
    main()
