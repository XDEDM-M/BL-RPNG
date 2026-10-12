from __future__ import annotations

import json
from collections import Counter

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def _audit_index(path: str, *, expected_role: str) -> tuple[dict, dict]:
    index_path = resolve_in_project(path, must_exist=True)
    index = read_index(index_path, verify_hashes=True)
    counts: Counter[int] = Counter()
    initial_ids: set[int] = set()
    boundary_ids: set[int] = set()
    converged = 0
    valid = total = 0
    trace_error = 0.0
    for row in index["shards"]:
        payload = torch.load(
            resolve_in_project(row["path"], must_exist=True),
            map_location="cpu",
            weights_only=False,
        )
        validation = validate_shard(payload)
        if payload["meta"]["role"] != expected_role or row["role"] != expected_role:
            raise ValueError(f"Role mismatch in {row['path']}")
        if int(validation["resolution"]) != 64:
            raise ValueError(f"Unexpected resolution in {row['path']}")
        task = int(row["target_task_id"])
        initial = int(row["initial_condition_id"])
        boundary = int(row["boundary_id"])
        if initial in initial_ids:
            raise ValueError(f"Duplicate initial condition in {expected_role}: {initial}")
        initial_ids.add(initial)
        boundary_ids.add(boundary)
        counts[task] += 1
        converged += int(bool(payload["meta"].get("solver_converged", False)))
        correction_mask = payload["valid_correction_mask"].bool()
        valid += int(correction_mask.sum())
        total += correction_mask.numel()
        trace_error = max(trace_error, float(validation["trace_error_max"]))
    summary = {
        "path": str(index_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(index_path),
        "rollouts": len(index["shards"]),
        "task_counts": {str(key): value for key, value in sorted(counts.items())},
        "boundary_ids": sorted(boundary_ids),
        "initial_condition_ids": sorted(initial_ids),
        "solver_convergence_fraction": converged / max(len(index["shards"]), 1),
        "valid_correction_fraction": valid / max(total, 1),
        "maximum_trace_error": trace_error,
    }
    return index, summary


def main() -> None:
    support_path = (
        "artifacts/data/reconfirmation/allen_cahn_reconfirmation_support_64.index.json"
    )
    validation_path = (
        "artifacts/data/reconfirmation/allen_cahn_reconfirmation_validation_64.index.json"
    )
    with workspace_audit("verify_reconfirmation") as audit:
        support_index, support = _audit_index(support_path, expected_role="support")
        validation_index, validation = _audit_index(
            validation_path, expected_role="validation"
        )
        support_initials = set(support.pop("initial_condition_ids"))
        validation_initials = set(validation.pop("initial_condition_ids"))
        checks = {
            "support_rollout_count": support["rollouts"] == 384,
            "validation_rollout_count": validation["rollouts"] == 192,
            "support_tasks": len(support["task_counts"]) == 48
            and set(support["task_counts"].values()) == {8},
            "validation_tasks": len(validation["task_counts"]) == 48
            and set(validation["task_counts"].values()) == {4},
            "boundary_sets_match": support["boundary_ids"] == validation["boundary_ids"],
            "initial_conditions_disjoint": not (support_initials & validation_initials),
            "all_solvers_converged": support["solver_convergence_fraction"] == 1.0
            and validation["solver_convergence_fraction"] == 1.0,
            "valid_corrections_at_least_95pct": support["valid_correction_fraction"] >= 0.95
            and validation["valid_correction_fraction"] >= 0.95,
            "boundary_trace_below_1e6": support["maximum_trace_error"] < 1.0e-6
            and validation["maximum_trace_error"] < 1.0e-6,
            "manifest_roles_separate": support_index["metadata"]["split"] == "support"
            and validation_index["metadata"]["split"] == "validation",
        }
        passed = all(checks.values())
        report = write_json(
            "artifacts/reports/allen_cahn_reconfirmation_lr_data_audit.json",
            {
                "schema": "blrpng_reconfirmation_lr_data_audit_v1",
                "passed": passed,
                "checks": checks,
                "support": support,
                "validation": validation,
            },
        )
        gate = None
        if passed:
            gate = write_json(
                "artifacts/manifests/allen_cahn_reconfirmation_lr_data_gate.json",
                {
                    "schema": "blrpng_reconfirmation_lr_data_gate_v1",
                    "passed": True,
                    "query_opened": False,
                    "report": str(report.relative_to(PROJECT_ROOT)),
                    "report_sha256": sha256_file(report),
                },
            )
        audit.update({"report": str(report), "gate": str(gate) if gate else None})
    print(json.dumps({
        "passed": passed,
        "checks": checks,
        "support_rollouts": support["rollouts"],
        "validation_rollouts": validation["rollouts"],
        "gate": str(gate) if gate else None,
    }, indent=2))


if __name__ == "__main__":
    main()
