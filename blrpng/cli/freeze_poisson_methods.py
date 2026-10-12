from __future__ import annotations

import json

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


SOURCES = {
    "ordinary_nlt": "artifacts/checkpoints/poisson/poisson_source_s42_final_only.inference.pt",
    "path_nlt": "artifacts/checkpoints/poisson/poisson_source_s42_blrpng.inference.pt",
}


def main() -> None:
    require_prerequisite_gate("artifacts/manifests/poisson_negative_control_lr_data_gate.json")
    checkpoints = {
        method: {
            "path": path,
            "sha256": sha256_file(resolve_in_project(path, must_exist=True)),
        }
        for method, path in SOURCES.items()
    }
    payload = {
        "schema": "blrpng_poisson_method_freeze_v1",
        "passed": True,
        "frozen": True,
        "seed": 42,
        "primary_support_budget": 4,
        "checkpoint_selection": "per-task 64x64 validation relL2 only",
        "query_used_for_selection": False,
        "methods": {
            "ordinary_nlt": {
                "source_objective": "final_only", "target_objective": "final_only",
                "adapter": "nlt",
            },
            "path_nlt": {
                "source_objective": "blrpng", "target_objective": "final_only",
                "adapter": "nlt",
            },
            "lr_scratch": {
                "source_objective": None, "target_objective": "final_only",
                "adapter": "full", "retention": "evaluate then delete full weights",
            },
            "boundary_augmented_lr_scratch": {
                "source_objective": None, "target_objective": "final_only",
                "adapter": "full", "representation": "boundary_augmented",
                "retention": "diagnostic subset; evaluate then delete full weights",
            },
        },
        "source_checkpoints": checkpoints,
        "primary_contrast": "path_nlt versus ordinary_nlt",
        "negative_control_gate": "absolute paired improvement <= 5%",
        "primary_query": "256x256, 16 unseen forcing fields per target boundary",
    }
    with workspace_audit("freeze_poisson_methods") as audit:
        output = write_json(
            "artifacts/manifests/poisson_negative_control_method_freeze.json", payload
        )
        audit.update({"output": str(output), "source_checkpoints": checkpoints})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
