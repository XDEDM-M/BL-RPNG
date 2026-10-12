from __future__ import annotations

import argparse
import json
from pathlib import Path

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.manifest import read_manifest
from blrpng.data.shards import read_index, validate_shard
from blrpng.data.tasks import validate_task_isolation
from blrpng.paths import resolve_in_project, sha256_file, write_json


def _verify_index(path: Path) -> dict:
    index = read_index(path, verify_hashes=True)
    convergence = []
    trace = []
    solve_seconds = assembly_seconds = 0.0
    for row in index["shards"]:
        import torch
        shard = torch.load(resolve_in_project(row["path"], must_exist=True), map_location="cpu", weights_only=False)
        validation = validate_shard(shard)
        trace.append(validation["trace_error_max"])
        convergence.append(bool(shard["meta"]["solver_converged"]))
        solve_seconds += sum(shard["meta"].get("solver_timing", []))
        assembly_seconds += float(shard["meta"].get("assembly_time", 0.0))
    return {
        "index_sha256": index["sha256"], "rollouts": len(index["shards"]),
        "all_hashes_valid": True, "solver_converged_fraction": sum(convergence) / len(convergence),
        "trace_error_max": max(trace), "solver_seconds": solve_seconds,
        "assembly_seconds": assembly_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--output", default="artifacts/reports/allen_cahn_v2_data_verification.json")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = resolve_in_project("artifacts/manifests/allen_cahn_v2_calibration.json", must_exist=True)
    with workspace_audit("verify_calibration"):
        report = {
            "schema": "blrpng_v2_data_verification_v1",
            "manifest_sha256": sha256_file(manifest_path),
            "task_isolation": validate_task_isolation(read_manifest(manifest_path)),
            "indices": {},
        }
        for path in sorted((resolve_in_project("artifacts/data")).glob("allen_cahn_v2_*.index.json")):
            report["indices"][path.name] = _verify_index(path)
        # Nested support membership is an invariant of index prefixes.
        for task in range(int(config["data"]["target_boundaries"])):
            previous = set()
            for budget in config["data"]["support_budgets"]:
                path = resolve_in_project(f"artifacts/data/tasks/task{task:02d}_support_lr_64_m{budget}.index.json", must_exist=True)
                current = {row["path"] for row in read_index(path)["shards"]}
                if not previous <= current or len(current) != int(budget):
                    raise ValueError(f"Support budgets are not nested for task {task}.")
                previous = current
        report["nested_support_budgets"] = True
        report["passed"] = bool(report["indices"]) and all(
            row["solver_converged_fraction"] == 1.0 and row["trace_error_max"] < 1e-6
            for row in report["indices"].values()
        )
        output = write_json(args.output, report)
    print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
