from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import resolve_in_project, write_json


ROOT = "artifacts/data/gray_scott_r2_seed42"


def _load_rows(path: str) -> tuple[dict, dict[int, dict]]:
    index = read_index(path, verify_hashes=True)
    payloads = {}
    for row in index["shards"]:
        shard = torch.load(resolve_in_project(row["path"], must_exist=True),
                           map_location="cpu", weights_only=False)
        validate_shard(shard)
        payloads[int(shard["ids"]["case_id"])] = shard
    return index, payloads


def main() -> None:
    sets = {}
    for split in ("train", "validation"):
        _, fine = _load_rows(f"{ROOT}/source_{split}_256.index.json")
        _, coarse = _load_rows(f"{ROOT}/source_{split}_paired_128.index.json")
        sets[split] = (fine, coarse)
    all_payloads = [payload for pair in sets.values() for group in pair for payload in group.values()]
    convergence = [bool(p["meta"]["solver_converged"]) for p in all_payloads]
    valid = [float(p["valid_correction_mask"].float().mean()) for p in all_payloads]
    trace = [float(validate_shard(p)["trace_error_max"]) for p in all_payloads]
    residual = [max(float(v) for v in p["meta"]["final_residual_norms"]) for p in all_payloads]
    lift_residual = []
    correction_cosines = []
    trajectory_restriction = []
    pairing_ok = True
    for fine, coarse in sets.values():
        pairing_ok = pairing_ok and set(coarse).issubset(fine)
        for case_id, low in coarse.items():
            high = fine[case_id]
            pairing_ok = pairing_ok and all(
                low["ids"][key] == high["ids"][key]
                for key in ("case_id", "boundary_id", "initial_condition_id", "pair_id")
            )
            restricted_trajectory = F.interpolate(
                high["trajectory"], size=(128, 128), mode="bilinear", align_corners=True
            )
            trajectory_restriction.append(float(
                (restricted_trajectory - low["trajectory"]).norm()
                / low["trajectory"].norm().clamp_min(1e-8)
            ))
            restricted_correction = F.interpolate(
                high["newton_correction"], size=(128, 128),
                mode="bilinear", align_corners=True,
            )
            a, b = restricted_correction.flatten(1), low["newton_correction"].flatten(1)
            correction_cosines.extend(float(v) for v in F.cosine_similarity(a, b, dim=1))
    for payload in all_payloads:
        lift = payload["boundary_lift"]
        lap = (lift[:, 2:, 1:-1] + lift[:, :-2, 1:-1]
               + lift[:, 1:-1, 2:] + lift[:, 1:-1, :-2]
               - 4.0 * lift[:, 1:-1, 1:-1])
        lift_residual.append(float(lap.abs().max()))
    passed = (
        sum(convergence) / len(convergence) == 1.0
        and min(valid) >= 0.95
        and max(trace) <= 1e-6
        and max(residual) <= 1e-7
        and max(lift_residual) <= 2e-6
        and pairing_ok
        and len(sets["train"][0]) == 100 and len(sets["validation"][0]) == 20
        and len(sets["train"][1]) == 40 and len(sets["validation"][1]) == 10
    )
    payload = {
        "schema": "blrpng_gray_scott_r2_data_gate_v1",
        "passed": passed,
        "source_256_rollouts": 120,
        "paired_128_rollouts": 50,
        "convergence_fraction": sum(convergence) / len(convergence),
        "valid_correction_fraction_min": min(valid),
        "boundary_trace_error_max": max(trace),
        "nonlinear_residual_max": max(residual),
        "harmonic_lift_residual_max": max(lift_residual),
        "paired_ids_exact": pairing_ok,
        "mean_paired_trajectory_restriction_rel_l2": sum(trajectory_restriction) / len(trajectory_restriction),
        "mean_paired_newton_correction_cosine": sum(correction_cosines) / len(correction_cosines),
        "query_used": False,
    }
    with workspace_audit("verify_gray_scott_r2") as audit:
        output = write_json("artifacts/manifests/gray_scott_r2_seed42_data_gate.json", payload)
        audit.update({"output": str(output), "passed": passed})
    if not passed:
        raise RuntimeError(f"Gray--Scott R2 data gate failed: {payload}")
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
