from __future__ import annotations

import argparse
import json
from collections import Counter

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def _initial_ids(path: str) -> set[int]:
    return {
        int(row["initial_condition_id"])
        for row in read_index(path)["shards"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one frozen Allen-Cahn HR blind-query resolution."
    )
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    args = parser.parse_args()
    relative = (
        "artifacts/data/reconfirmation/"
        f"allen_cahn_reconfirmation_query_{args.resolution}.index.json"
    )
    index_path = resolve_in_project(relative, must_exist=True)
    index = read_index(index_path, verify_hashes=True)
    support_ids = _initial_ids(
        "artifacts/data/reconfirmation/"
        "allen_cahn_reconfirmation_support_64.index.json"
    )
    validation_ids = _initial_ids(
        "artifacts/data/reconfirmation/"
        "allen_cahn_reconfirmation_validation_64.index.json"
    )
    counts: Counter[int] = Counter()
    query_ids: set[int] = set()
    units: set[tuple[int, int, int]] = set()
    converged = 0
    valid = total = 0
    maximum_trace = 0.0
    pair_rows = []
    for row in index["shards"]:
        path = resolve_in_project(row["path"], must_exist=True)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        validation = validate_shard(shard)
        if shard["meta"].get("role") != "query" or row.get("role") != "query":
            raise ValueError(f"Query role mismatch: {path}")
        if int(validation["resolution"]) != args.resolution:
            raise ValueError(f"Query resolution mismatch: {path}")
        if int(validation["steps"]) != 50:
            raise ValueError(f"Query horizon mismatch: {path}")
        task = int(row["target_task_id"])
        initial = int(row["initial_condition_id"])
        boundary = int(row["boundary_id"])
        pair = int(shard["ids"]["pair_id"])
        unit = (task, boundary, initial)
        if unit in units:
            raise ValueError(f"Duplicate query statistical unit: {unit}")
        units.add(unit)
        query_ids.add(initial)
        counts[task] += 1
        converged += int(bool(shard["meta"].get("solver_converged", False)))
        mask = shard["valid_correction_mask"].bool()
        valid += int(mask.sum())
        total += int(mask.numel())
        maximum_trace = max(maximum_trace, float(validation["trace_error_max"]))
        pair_rows.append((task, boundary, initial, pair))
    paired_resolution = 256 if args.resolution == 128 else 128
    paired_relative = (
        "artifacts/data/reconfirmation/"
        f"allen_cahn_reconfirmation_query_{paired_resolution}.index.json"
    )
    paired_path = resolve_in_project(paired_relative)
    cross_resolution_pairing = None
    if paired_path.exists():
        paired = read_index(paired_path, verify_hashes=True)
        paired_rows = []
        for row in paired["shards"]:
            shard = torch.load(
                resolve_in_project(row["path"], must_exist=True),
                map_location="cpu",
                weights_only=False,
            )
            paired_rows.append((
                int(row["target_task_id"]),
                int(row["boundary_id"]),
                int(row["initial_condition_id"]),
                int(shard["ids"]["pair_id"]),
            ))
        cross_resolution_pairing = sorted(pair_rows) == sorted(paired_rows)
    checks = {
        "rollout_count_768": len(index["shards"]) == 768,
        "task_count_48": len(counts) == 48,
        "rollouts_per_task_16": set(counts.values()) == {16},
        "all_solvers_converged": converged == len(index["shards"]),
        "valid_corrections_at_least_95pct": valid / max(total, 1) >= 0.95,
        "boundary_trace_below_1e_minus_6": maximum_trace < 1e-6,
        "support_query_initials_disjoint": not (query_ids & support_ids),
        "validation_query_initials_disjoint": not (query_ids & validation_ids),
        "index_role_query": index.get("metadata", {}).get("split") == "query",
        "index_resolution_matches": int(index.get("metadata", {}).get("resolution", -1))
        == args.resolution,
    }
    if cross_resolution_pairing is not None:
        checks["cross_resolution_pairing_exact"] = cross_resolution_pairing
    passed = all(checks.values())
    payload = {
        "schema": "blrpng_reconfirmation_query_data_audit_v1",
        "passed": passed,
        "resolution": args.resolution,
        "checks": checks,
        "index": str(index_path.relative_to(PROJECT_ROOT)),
        "index_sha256": sha256_file(index_path),
        "rollouts": len(index["shards"]),
        "solver_convergence_fraction": converged / max(len(index["shards"]), 1),
        "valid_correction_fraction": valid / max(total, 1),
        "maximum_trace_error": maximum_trace,
        "cross_resolution_pairing": cross_resolution_pairing,
    }
    with workspace_audit(f"verify_reconfirmation_query_{args.resolution}") as audit:
        report = write_json(
            "artifacts/reports/"
            f"allen_cahn_reconfirmation_query_{args.resolution}_data_audit.json",
            payload,
        )
        gate = None
        if passed:
            gate = write_json(
                "artifacts/manifests/"
                f"allen_cahn_reconfirmation_query_{args.resolution}_data_gate.json",
                {
                    "schema": "blrpng_reconfirmation_query_data_gate_v1",
                    "passed": True,
                    "resolution": args.resolution,
                    "report": str(report.relative_to(PROJECT_ROOT)),
                    "report_sha256": sha256_file(report),
                },
            )
        audit.update({"report": str(report), "gate": str(gate) if gate else None})
    print(json.dumps({
        "passed": passed,
        "checks": checks,
        "valid_correction_fraction": payload["valid_correction_fraction"],
        "maximum_trace_error": maximum_trace,
        "gate": str(gate) if gate else None,
    }, indent=2))


if __name__ == "__main__":
    main()
