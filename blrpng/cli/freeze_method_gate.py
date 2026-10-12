from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Allen-Cahn method before opening HR query.")
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    parser.add_argument("--oracle-gate", default="artifacts/manifests/allen_cahn_oracle_saturation_gate.json")
    args = parser.parse_args()
    config = load_config(args.config)
    oracle = require_prerequisite_gate(args.oracle_gate)
    sources = {
        "path_pretraining_checkpoint": "artifacts/checkpoints/allen_cahn_formal_a_blrpng_seed42.pt",
        "current_state": "artifacts/reports/current_state_20260820.json",
        "method_contract": "METHOD.md",
        "theory_contract": "THEORY.md",
        "blind_task_manifest": "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
    }
    hashes = {
        key: sha256_file(resolve_in_project(path, must_exist=True)) for key, path in sources.items()
    }
    payload = {
        "schema": "blrpng_method_freeze_gate_v1",
        "passed": True,
        "method": "BL-RPNG-NLT",
        "source_pretraining": "Newton path at normalized log-residual progress 0.25",
        "target_adaptation": "NLT final-only",
        "support_budgets": config["blind_protocol"]["support_budgets"],
        "primary_endpoint": "m4, 256x256, 20-step per-rollout relative L2",
        "secondary_horizon": 50,
        "checkpoint_selection": "equal-weight LR seen-boundary and unseen-family validation only",
        "hr_query_used_for_selection": False,
        "oracle_gate": str(resolve_in_project(args.oracle_gate).relative_to(PROJECT_ROOT)),
        "oracle_gate_sha256": sha256_file(resolve_in_project(args.oracle_gate)),
        "oracle_status": oracle.get("oracle_status"),
        "saturated_hr_oracle": bool(oracle.get("saturated_hr_oracle", False)),
        "authorized_stage_progression": bool(
            oracle.get("authorized_stage_progression", False)
        ),
        "hr_reference_claim": (
            "resource-capped m256 HR reference"
            if oracle.get("oracle_status") == "resource_capped_unsaturated"
            else "saturated HR oracle"
        ),
        "sources": sources,
        "source_sha256": hashes,
        "target_path_promotion_rule": (
            "NLT+path remains an ablation unless it materially wins on frozen LR validation "
            "before any HR query is opened"
        ),
    }
    with workspace_audit("freeze_method_gate") as audit:
        output = write_json(
            "artifacts/manifests/allen_cahn_method_freeze_gate.json", payload
        )
        audit["output"] = str(output)
    print(json.dumps({
        "output": str(output),
        "query_may_be_prepared": True,
        "oracle": oracle["oracle_status"],
        "saturated_hr_oracle": bool(oracle.get("saturated_hr_oracle", False)),
    }, indent=2))


if __name__ == "__main__":
    main()
