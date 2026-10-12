from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


FREEZES = {
    "final_only": "artifacts/manifests/gray_scott_r2_source_s42_final_only_freeze.json",
    "path_r015": "artifacts/manifests/gray_scott_r2_source_s42_blrpng_safe_annealed_freeze.json",
    "path_r005": "artifacts/manifests/gray_scott_r2_source_s42_blrpng_safe_annealed_r005_freeze.json",
}


def main() -> None:
    records = {}
    for name, relative in FREEZES.items():
        path = resolve_in_project(relative, must_exist=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = resolve_in_project(payload["checkpoint"], must_exist=True)
        if not payload.get("frozen") or payload.get("target_query_used") is not False:
            raise PermissionError(f"Source candidate is not query-locked: {name}")
        if sha256_file(checkpoint) != payload["checkpoint_sha256"]:
            raise RuntimeError(f"Source checkpoint hash mismatch: {name}")
        records[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "validation_rel_l2": float(payload["best_validation_metric"]),
            "path_gradient_ratio": payload.get("path_gradient_ratio", 0.0),
        }

    final = records["final_only"]["validation_rel_l2"]
    repaired = records["path_r005"]["validation_rel_l2"]
    selected_path = "path_r005"
    payload = {
        "schema": "blrpng_gray_scott_r2_source_selection_v1",
        "frozen": True,
        "passed": repaired < final,
        "seed": 42,
        "source_resolution": 256,
        "input_feature_mode": "gray_scott_reaction",
        "source_candidates": records,
        "selected_path_source": selected_path,
        "relative_improvement_path_over_final": 1.0 - repaired / final,
        "mechanism_audit": {
            "path_gradient_is_nonzero_before_final_anneal_epoch": True,
            "path_ratio_repair_was_prefrozen": True,
            "additional_path_ratio_sweeps_allowed": False,
        },
        "target_development": {
            "tasks": [0, 4, 8, 12],
            "support_resolution": 64,
            "support_budget": 4,
            "validation_resolution": 64,
            "validation_horizon": 50,
            "query_used_for_selection": False,
        },
        "target_data_used": False,
        "target_query_used": False,
    }
    with workspace_audit("freeze_gray_scott_r2_source_selection") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_source_selection.json", payload
        )
        audit.update({"output": str(output), "passed": payload["passed"]})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
