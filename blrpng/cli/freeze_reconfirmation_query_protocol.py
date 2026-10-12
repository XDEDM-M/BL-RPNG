from __future__ import annotations

import json

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def main() -> None:
    checkpoint_freeze_path = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_checkpoint_freeze.json",
        must_exist=True,
    )
    checkpoint_freeze = require_prerequisite_gate(checkpoint_freeze_path)
    data_gate_path = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_lr_data_gate.json",
        must_exist=True,
    )
    require_prerequisite_gate(data_gate_path)
    payload = {
        "schema": "blrpng_reconfirmation_query_protocol_freeze_v1",
        "passed": True,
        "seed": 42,
        "query_opened_at_freeze": False,
        "primary_method": checkpoint_freeze["selected_primary_method"],
        "primary_support_budget": 4,
        "data_efficiency_budgets": [1, 2, 4, 8],
        "primary_endpoint": "256x256, 20-step per-rollout relative L2",
        "secondary_endpoints": [
            "128x128 20-step relative L2",
            "256x256 50-step relative L2",
            "near-boundary and interior relative L2",
            "PDE residual and Fourier shell error",
        ],
        "evaluation_methods": [
            "nlt",
            "ordinary_nlt",
            "nlt_path",
            "nlt_commutator",
            "nlt_commutator_defect",
            "lr_scratch",
        ],
        "lr_scratch_protocol": {
            "representation": "lifted",
            "initialization": "deterministic random seed 42",
            "epochs": 100,
            "learning_rate": 1.0e-4,
            "scheduler": "cosine",
            "minimum_learning_rate": 1.0e-6,
            "batch_size": 4,
            "weight_decay": 1.0e-6,
            "checkpoint_selection": "64x64 LR validation only",
        },
        "statistics": {
            "replicates": 50_000,
            "hierarchy": "seed->boundary->initial_condition",
            "scratch_improvement_threshold": 0.75,
            "marginal_speedup_threshold": 10.0,
        },
        "checkpoint_freeze": str(checkpoint_freeze_path.relative_to(PROJECT_ROOT)),
        "checkpoint_freeze_sha256": sha256_file(checkpoint_freeze_path),
        "lr_data_gate": str(data_gate_path.relative_to(PROJECT_ROOT)),
        "lr_data_gate_sha256": sha256_file(data_gate_path),
        "selection_prohibition": (
            "No method, hyperparameter, or checkpoint may be re-selected from HR query results."
        ),
    }
    with workspace_audit("freeze_reconfirmation_query_protocol") as audit:
        output = write_json(
            "artifacts/manifests/allen_cahn_reconfirmation_query_protocol_freeze.json",
            payload,
        )
        audit["output"] = str(output)
    print(json.dumps({
        "output": str(output),
        "primary_method": payload["primary_method"],
        "query_opened_at_freeze": False,
    }, indent=2))


if __name__ == "__main__":
    main()
