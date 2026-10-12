from __future__ import annotations

import json

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import resolve_in_project, write_json


TASKS = (0, 4, 8, 12)


def main() -> None:
    old = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    old_initials = {
        int(sample["initial_condition_id"])
        for task in old["targets"]
        for split in ("support", "validation", "query")
        for sample in task[split]
    }
    pairs = {}
    records = []
    for resolution in (128, 256):
        pairs[resolution] = set()
        for task in TASKS:
            index_path = (
                "artifacts/data/gray_scott_r2_seed42/blind_query/"
                f"task{task:02d}_query_{resolution}.index.json"
            )
            index = read_index(index_path, verify_hashes=True)
            if len(index["shards"]) != 8 or int(index["metadata"]["steps"]) != 50:
                raise RuntimeError("R2 blind-query index is incomplete.")
            for row in index["shards"]:
                payload = torch.load(
                    resolve_in_project(row["path"], must_exist=True),
                    map_location="cpu", weights_only=False,
                )
                shape = validate_shard(payload)
                initial_id = int(payload["ids"]["initial_condition_id"])
                pair_id = int(payload["ids"]["pair_id"])
                if initial_id in old_initials:
                    raise RuntimeError("R2 blind-query initial condition leaked into an old split.")
                pairs[resolution].add((task, pair_id, initial_id))
                records.append({
                    "task": task,
                    "resolution": resolution,
                    "pair_id": pair_id,
                    "initial_condition_id": initial_id,
                    "solver_converged": bool(payload["meta"]["solver_converged"]),
                    "valid_fraction": float(payload["valid_correction_mask"].float().mean()),
                    "trace_error": float(shape["trace_error_max"]),
                    "final_residual_max": max(payload["meta"]["final_residual_norms"]),
                })
    checks = {
        "expected_shards": len(records) == 64,
        "paired_128_256": pairs[128] == pairs[256] and len(pairs[128]) == 32,
        "all_solvers_converged": all(row["solver_converged"] for row in records),
        "valid_correction_at_least_95pct": min(row["valid_fraction"] for row in records) >= 0.95,
        "boundary_trace_below_1e6": max(row["trace_error"] for row in records) < 1e-6,
        "nonlinear_residual_below_1e6": max(row["final_residual_max"] for row in records) < 1e-6,
        "old_split_leakage": False,
    }
    payload = {
        "schema": "blrpng_gray_scott_r2_blind_query_data_gate_v1",
        "frozen": True,
        "passed": all(value for key, value in checks.items() if key != "old_split_leakage"),
        "checks": checks,
        "rollouts_per_resolution": 32,
        "min_valid_correction_fraction": min(row["valid_fraction"] for row in records),
        "max_boundary_trace_error": max(row["trace_error"] for row in records),
        "max_final_nonlinear_residual": max(row["final_residual_max"] for row in records),
        "query_used_for_selection": False,
    }
    with workspace_audit("verify_gray_scott_r2_query") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_blind_query_data_gate.json", payload
        )
        audit.update({"output": str(output), "passed": payload["passed"]})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
