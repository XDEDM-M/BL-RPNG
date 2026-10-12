from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    diagnosis_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_lr_diagnosis.json"
    )
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if not diagnosis.get("frozen") or diagnosis.get("query_used") is not False:
        raise PermissionError("The LR-only diagnosis must be frozen before repair.")
    if not str(diagnosis.get("authorized_next_experiment", "")).startswith(
        "Train a source checkpoint with conflict gating"
    ):
        raise PermissionError("The frozen diagnosis does not authorize source repair.")
    payload = {
        "schema": "blrpng_gray_scott_path_repair_protocol_v1",
        "frozen": True,
        "seed": 42,
        "query_used": False,
        "repair": {
            "source_objective": "blrpng_safe_annealed",
            "conflict_gate": "path gradient is injected only when its cosine with "
            "the final gradient on shared parameters is positive",
            "path_gradient_ratio": {
                "epochs_0_to_9": 0.15,
                "epochs_10_to_19": "linear decay from 0.15 to 0",
            },
            "target_methods": ["safe_path_nlt", "safe_path_nlt_gated"],
            "target_budget": 4,
        },
        "mechanism_support_rule": {
            "overall_mean_better_than_ordinary_nlt": True,
            "minimum_improved_shift_families_out_of_4": 3,
            "minimum_task_wins_out_of_16": 9,
            "maximum_allowed_family_degradation": 0.05,
        },
        "precommitted_hr_methods": [
            "ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated"
        ],
        "hr_query_lock": "No 128 or 256 query shard may be generated until both "
        "safe target matrices and their checkpoint hashes are frozen.",
    }
    with workspace_audit("freeze_gray_scott_repair_protocol") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_repair_protocol.json", payload
        )
        audit.update({"output": str(output), "query_used": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
