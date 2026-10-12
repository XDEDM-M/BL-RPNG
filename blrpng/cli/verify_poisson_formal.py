from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.poisson_protocol import validate_poisson_protocol
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import resolve_in_project, write_json


DATA_ROOT = "artifacts/data/poisson_negative_control"


def _check_index(path: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    index = read_index(path, verify_hashes=True)
    shards = []
    trace_max = correction_identity_max = residual_max = 0.0
    for row in index["shards"]:
        shard = torch.load(
            resolve_in_project(row["path"], must_exist=True),
            map_location="cpu", weights_only=False,
        )
        validation = validate_shard(shard)
        if not shard["meta"].get("solver_converged", False):
            raise ValueError(f"Unconverged Poisson shard: {row['path']}")
        trace_max = max(trace_max, float(validation["trace_error_max"]))
        identity = shard["trajectory"][1] - shard["boundary_lift"]
        mask = shard["constraint_mask"].bool()
        identity[mask] = 0.0
        correction_identity_max = max(
            correction_identity_max,
            float((identity - shard["newton_correction"][0]).abs().max()),
        )
        residual_max = max(residual_max, float(shard["meta"]["residual_norm"]))
        shards.append(shard)
    return shards, {
        "trace_error_max": trace_max,
        "correction_final_identity_max": correction_identity_max,
        "linear_residual_norm_max": residual_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast validation of formal Poisson shards.")
    parser.add_argument(
        "--protocol", default="artifacts/manifests/poisson_negative_control_seed42_protocol.json"
    )
    parser.add_argument("--scope", choices=("lr", "query", "all"), default="all")
    args = parser.parse_args()
    protocol = json.loads(resolve_in_project(args.protocol, must_exist=True).read_text())
    protocol_validation = validate_poisson_protocol(protocol)
    paths = []
    if args.scope in {"lr", "all"}:
        paths.extend([
            f"{DATA_ROOT}/source_train_128.index.json",
            f"{DATA_ROOT}/source_validation_128.index.json",
            f"{DATA_ROOT}/target_support_64_all.index.json",
            f"{DATA_ROOT}/target_validation_64_all.index.json",
        ])
    if args.scope in {"query", "all"}:
        paths.extend([
            f"{DATA_ROOT}/target_query_128_all.index.json",
            f"{DATA_ROOT}/target_query_256_all.index.json",
        ])
    summaries = {}
    by_task_boundary: dict[int, set[int]] = defaultdict(set)
    split_forcing: dict[str, set[int]] = defaultdict(set)
    pair_ids: dict[int, set[int]] = defaultdict(set)
    total = 0
    with workspace_audit(f"verify_poisson_formal_{args.scope}") as audit:
        for path in paths:
            shards, summary = _check_index(path)
            summaries[path] = {**summary, "shards": len(shards)}
            total += len(shards)
            for shard in shards:
                task = int(shard["ids"]["target_task_id"])
                if task >= 0:
                    by_task_boundary[task].add(int(shard["ids"]["boundary_id"]))
                    split_forcing[shard["meta"]["role"]].add(
                        int(shard["ids"]["initial_condition_id"])
                    )
                    pair_ids[int(shard["ids"]["pair_id"])].add(
                        int(shard["meta"]["resolution"])
                    )
        if any(len(values) != 1 for values in by_task_boundary.values()):
            raise ValueError("A Poisson target task does not have one fixed boundary.")
        support = split_forcing.get("target_support", set())
        validation = split_forcing.get("target_validation", set())
        query = split_forcing.get("target_query", set())
        if support & validation or support & query or validation & query:
            raise ValueError("Poisson support/validation/query forcing leakage.")
        if args.scope in {"query", "all"}:
            bad_pairs = {pair: values for pair, values in pair_ids.items() if values != {128, 256}}
            if bad_pairs:
                raise ValueError(f"Poisson 128/256 pairing mismatch: {list(bad_pairs.items())[:3]}")
        trace_max = max(value["trace_error_max"] for value in summaries.values())
        identity_max = max(value["correction_final_identity_max"] for value in summaries.values())
        passed = trace_max <= 1e-6 and identity_max <= 1e-7
        report = {
            "schema": "blrpng_poisson_data_verification_v1",
            "scope": args.scope,
            "passed": passed,
            "protocol_validation": protocol_validation,
            "indices": summaries,
            "unique_target_boundaries": len(by_task_boundary),
            "support_validation_query_leakage": False,
            "trace_error_max": trace_max,
            "correction_equals_final_lifted_target_max": identity_max,
            "negative_control_identity": "Newton correction equals the sole final residual target",
            "shards_checked": total,
        }
        output = write_json(
            f"artifacts/manifests/poisson_negative_control_{args.scope}_data_gate.json",
            report,
        )
        audit.update({"output": str(output), "passed": passed})
        if not passed:
            raise RuntimeError(f"Poisson data gate failed: {report}")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
