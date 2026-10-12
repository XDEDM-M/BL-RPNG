from __future__ import annotations

import json

import numpy as np

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_json, write_json


def main() -> None:
    diagnostic = json.loads(resolve_in_project(
        "artifacts/reports/gray_scott/r2_highfreq_repair_diagnostic.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    if not diagnostic.get("new_query_authorized"):
        raise PermissionError("High-frequency diagnostic does not authorize confirmation.")
    old = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    task = next(row for row in old["targets"] if int(row["target_task_id"]) == 0)
    rng = np.random.default_rng(20260908)
    query = [{
        "case_id": 9_100_000 + local,
        "pair_id": 9_100_000 + local,
        "initial_condition_id": 910_000 + local,
        "initial_seed": int(rng.integers(0, 2**31 - 1)),
    } for local in range(8)]
    payload = {
        "schema": "blrpng_gray_scott_r2_highfreq_confirmation_protocol_v1",
        "frozen": True,
        "split_seed": 20260908,
        "task": {
            key: task[key] for key in (
                "target_task_id", "boundary_id", "boundary_seed", "boundary_regime",
                "correlation", "amplitude", "feed", "kill",
            )
        },
        "query": query,
        "resolutions": [128, 256],
        "horizon": 50,
        "methods": ["path_nlt", "path_nlt_lora"],
        "checkpoint_selection_used_this_query": False,
    }
    payload["protocol_sha256"] = sha256_json(payload)
    with workspace_audit("freeze_gray_scott_r2_highfreq_confirmation") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_highfreq_confirmation_protocol.json",
            payload,
        )
        audit.update({"output": str(output), "query_opened": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
