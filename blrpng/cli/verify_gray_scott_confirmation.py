from __future__ import annotations

import argparse
import json

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import resolve_in_project, write_json
from blrpng.training.physics import gray_scott_defect


ROOT = "artifacts/data/gray_scott_seed42_confirmation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Gray--Scott confirmation data.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    args = parser.parse_args()
    protocol = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_confirmation_query_protocol.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    original = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    expected_ids = {
        int(sample["initial_condition_id"])
        for task in protocol["tasks"] for sample in task["query"]
    }
    old_ids = {
        int(sample["initial_condition_id"])
        for task in original["targets"]
        for role in ("support", "validation", "query")
        for sample in task[role]
    }
    if expected_ids & old_ids:
        raise RuntimeError("Confirmation query leaks an old initial condition.")
    index = read_index(
        f"{ROOT}/target_query_{args.resolution}_all.index.json",
        verify_hashes=True,
    )
    if len(index["shards"]) != 128:
        raise RuntimeError("Confirmation query must contain 128 rollouts.")
    actual_ids = {int(row["initial_condition_id"]) for row in index["shards"]}
    if actual_ids != expected_ids:
        raise RuntimeError("Generated confirmation IDs differ from the frozen protocol.")
    convergence = []
    valid = []
    residual_max = defect_max = trace_max = correction_trace_max = 0.0
    boundaries = {}
    with workspace_audit(f"verify_gray_scott_confirmation_{args.resolution}") as audit:
        for row in index["shards"]:
            shard = torch.load(
                resolve_in_project(row["path"], must_exist=True),
                map_location="cpu", weights_only=False,
            )
            validation = validate_shard(shard)
            convergence.append(bool(shard["meta"]["solver_converged"]))
            valid.append(float(shard["valid_correction_mask"].float().mean()))
            residual_max = max(
                residual_max,
                max(float(value) for value in shard["meta"]["final_residual_norms"]),
            )
            trace_max = max(trace_max, float(validation["trace_error_max"]))
            mask = shard["constraint_mask"].bool()
            correction_trace_max = max(
                correction_trace_max,
                float(shard["newton_correction"][:, mask].abs().max()),
            )
            task = int(shard["ids"]["target_task_id"])
            if task in boundaries and not torch.equal(
                boundaries[task], shard["boundary_values"]
            ):
                raise RuntimeError("Confirmation task has a changing boundary.")
            boundaries.setdefault(task, shard["boundary_values"])
            conditions = shard["condition_fields"].unsqueeze(0)
            for step in range(shard["trajectory"].shape[0] - 1):
                defect = gray_scott_defect(
                    shard["trajectory"][step:step + 1],
                    shard["trajectory"][step + 1:step + 2],
                    {
                        "dt": 1.0, "diffusion_u": 2e-5, "diffusion_v": 1e-5,
                        "condition_fields": conditions,
                    },
                )
                defect_max = max(defect_max, float(defect.abs().max()))
        convergence_fraction = sum(convergence) / len(convergence)
        valid_fraction = sum(valid) / len(valid)
        passed = (
            convergence_fraction >= 0.99
            and valid_fraction >= 0.95
            and trace_max <= 1e-6
            and correction_trace_max == 0.0
            and residual_max <= 1e-7
            and defect_max <= 1e-5
            and len(boundaries) == 16
        )
        payload = {
            "schema": "blrpng_gray_scott_confirmation_data_gate_v1",
            "passed": passed,
            "resolution": args.resolution,
            "rollouts": len(index["shards"]),
            "convergence_fraction": convergence_fraction,
            "valid_correction_fraction": valid_fraction,
            "boundary_trace_error_max": trace_max,
            "correction_trace_error_max": correction_trace_max,
            "nonlinear_residual_max": residual_max,
            "ground_truth_defect_max": defect_max,
            "initial_condition_leakage": False,
            "fixed_boundary_per_target_task": True,
        }
        output = write_json(
            f"artifacts/manifests/gray_scott_seed42_confirmation{args.resolution}_data_gate.json",
            payload,
        )
        audit.update({"output": str(output), "passed": passed})
        if not passed:
            raise RuntimeError(f"Confirmation data gate failed: {payload}")
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
