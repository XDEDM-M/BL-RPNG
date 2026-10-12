from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.gray_scott_protocol import validate_gray_scott_protocol
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import resolve_in_project, write_json
from blrpng.training.physics import gray_scott_defect


ROOT = "artifacts/data/gray_scott_seed42"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Gray--Scott v4 data.")
    parser.add_argument(
        "--scope",
        choices=("lr", "query128", "query256", "query", "all"),
        default="all",
    )
    parser.add_argument(
        "--protocol", default="artifacts/manifests/gray_scott_seed42_protocol.json"
    )
    args = parser.parse_args()
    protocol = json.loads(resolve_in_project(args.protocol, must_exist=True).read_text())
    protocol_validation = validate_gray_scott_protocol(protocol)
    paths = []
    if args.scope in {"lr", "all"}:
        paths.extend([
            f"{ROOT}/source_train_128.index.json",
            f"{ROOT}/source_validation_128.index.json",
            f"{ROOT}/target_support_64_all.index.json",
            f"{ROOT}/target_validation_64_all.index.json",
        ])
    if args.scope in {"query128", "query", "all"}:
        paths.append(f"{ROOT}/target_query_128_all.index.json")
    if args.scope in {"query256", "query", "all"}:
        paths.append(f"{ROOT}/target_query_256_all.index.json")
    convergence, valid_values, residuals = [], [], []
    trace_max = correction_trace_max = defect_max = 0.0
    boundaries: dict[tuple[int, int], torch.Tensor] = {}
    roles: dict[str, set[int]] = defaultdict(set)
    pair_resolutions: dict[int, set[int]] = defaultdict(set)
    counts = {}
    with workspace_audit(f"verify_gray_scott_formal_{args.scope}") as audit:
        for path in paths:
            index = read_index(path, verify_hashes=True)
            counts[path] = len(index["shards"])
            for row in index["shards"]:
                shard = torch.load(
                    resolve_in_project(row["path"], must_exist=True),
                    map_location="cpu", weights_only=False,
                )
                validation = validate_shard(shard)
                convergence.append(bool(shard["meta"]["solver_converged"]))
                valid_values.append(float(shard["valid_correction_mask"].float().mean()))
                residuals.extend(float(value) for value in shard["meta"]["final_residual_norms"])
                trace_max = max(trace_max, float(validation["trace_error_max"]))
                mask = shard["constraint_mask"].bool()
                correction_trace_max = max(
                    correction_trace_max,
                    float(shard["newton_correction"][:, mask].abs().max()),
                )
                conditions = shard["condition_fields"].unsqueeze(0)
                for step in range(shard["trajectory"].shape[0] - 1):
                    defect = gray_scott_defect(
                        shard["trajectory"][step : step + 1],
                        shard["trajectory"][step + 1 : step + 2],
                        {
                            "dt": 1.0, "diffusion_u": 2e-5, "diffusion_v": 1e-5,
                            "condition_fields": conditions,
                        },
                    )
                    defect_max = max(defect_max, float(defect.abs().max()))
                task = int(shard["ids"]["target_task_id"])
                resolution = int(shard["meta"]["resolution"])
                if task >= 0:
                    boundary_key = (task, resolution)
                    if boundary_key in boundaries and not torch.equal(
                        boundaries[boundary_key], shard["boundary_values"]
                    ):
                        raise ValueError("A target task contains changing boundary traces.")
                    boundaries.setdefault(boundary_key, shard["boundary_values"])
                    roles[shard["meta"]["role"]].add(
                        int(shard["ids"]["initial_condition_id"])
                    )
                    pair_resolutions[int(shard["ids"]["pair_id"])].add(resolution)
                expected_conditions = (
                    float(shard["meta"]["feed"]),
                    float(shard["meta"]["kill"]),
                    float(shard["meta"]["correlation"]),
                )
                actual_conditions = tuple(
                    float(shard["condition_fields"][channel].mean()) for channel in range(3)
                )
                if any(abs(left - right) > 1e-6 for left, right in zip(actual_conditions, expected_conditions)):
                    raise ValueError("Gray--Scott condition fields do not match metadata.")
        support, validation_ids, query = (
            roles.get("target_support", set()),
            roles.get("target_validation", set()),
            roles.get("target_query", set()),
        )
        if support & validation_ids or support & query or validation_ids & query:
            raise ValueError("Gray--Scott support/validation/query initial-condition leakage.")
        if args.scope in {"query128", "query256", "query", "all"}:
            expected_resolutions = {
                "query128": {128}, "query256": {256},
                "query": {128, 256}, "all": {128, 256},
            }[args.scope]
            bad = {
                key: value for key, value in pair_resolutions.items()
                if value != expected_resolutions
            }
            if bad:
                raise ValueError(
                    "Gray--Scott query resolution pairing mismatch: "
                    f"expected {expected_resolutions}, got {list(bad.items())[:3]}"
                )
        convergence_fraction = sum(convergence) / len(convergence)
        valid_fraction = sum(valid_values) / len(valid_values)
        passed = (
            convergence_fraction >= 0.99 and valid_fraction >= 0.95
            and trace_max <= 1e-6 and correction_trace_max == 0.0
            and max(residuals) <= 1e-7 and defect_max <= 1e-5
        )
        report = {
            "schema": "blrpng_gray_scott_data_gate_v1",
            "scope": args.scope, "passed": passed,
            "protocol_validation": protocol_validation,
            "indices": counts,
            "shards_checked": len(convergence),
            "convergence_fraction": convergence_fraction,
            "valid_correction_fraction": valid_fraction,
            "boundary_trace_error_max": trace_max,
            "correction_trace_error_max": correction_trace_max,
            "nonlinear_residual_max": max(residuals),
            "ground_truth_defect_max": defect_max,
            "fixed_boundary_per_target_task": True,
            "initial_condition_split_leakage": False,
        }
        output = write_json(
            f"artifacts/manifests/gray_scott_seed42_{args.scope}_data_gate.json", report
        )
        audit.update({"output": str(output), "passed": passed})
        if not passed:
            raise RuntimeError(f"Gray--Scott data gate failed: {report}")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
