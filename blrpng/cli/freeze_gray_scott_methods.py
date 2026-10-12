from __future__ import annotations

import json

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


SOURCES = {
    "ordinary_nlt": "artifacts/checkpoints/gray_scott/gray_scott_source_s42_final_only.inference.pt",
    "path_nlt": "artifacts/checkpoints/gray_scott/gray_scott_source_s42_blrpng.inference.pt",
}


def main() -> None:
    require_prerequisite_gate("artifacts/manifests/gray_scott_seed42_lr_data_gate.json")
    source_rows = {
        key: {"path": path, "sha256": sha256_file(resolve_in_project(path, must_exist=True))}
        for key, path in SOURCES.items()
    }
    payload = {
        "schema": "blrpng_gray_scott_method_freeze_v1",
        "passed": True, "frozen": True, "seed": 42,
        "support_budget": 4,
        "selection": "per-task 64x64 validation, 10-step projected rollout relL2",
        "query_used_for_selection": False,
        "source_checkpoints": source_rows,
        "methods": {
            "ordinary_nlt": {"source": "final_only", "target_objective": "final_only"},
            "path_nlt": {"source": "blrpng", "target_objective": "final_only", "current_main": True},
            "path_nlt_gated": {"source": "blrpng", "target_objective": "blrpng_v2"},
            "path_nlt_commutator": {"source": "blrpng", "target_objective": "commutator_only"},
            "path_nlt_plus": {"source": "blrpng", "target_objective": "blrpng_plus"},
            "lr_scratch_converged": {
                "source": None, "target_objective": "final_only", "epochs_max": 100,
            },
        },
        "upgrade_rule": (
            "path_nlt remains primary unless a corrected path/commutator/defect variant "
            "wins on frozen LR validation across shift families before HR query is opened"
        ),
        "primary_query": "256x256 at horizons 20 and 50",
    }
    with workspace_audit("freeze_gray_scott_methods") as audit:
        output = write_json("artifacts/manifests/gray_scott_seed42_method_freeze.json", payload)
        audit.update({"output": str(output), "source_checkpoints": source_rows})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
