from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.gray_scott_protocol import build_gray_scott_protocol, validate_gray_scott_protocol
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the seed42 Gray--Scott R2 development protocol.")
    parser.add_argument("--config", default="configs/gray_scott_r2_development.yaml")
    args = parser.parse_args()
    diagnosis_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_confirmation256_failure_diagnosis.json",
        must_exist=True,
    )
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosis.get("stage_passed") is not False or "multiscale" not in str(
        diagnosis.get("authorized_next_experiment", "")
    ):
        raise PermissionError("The frozen Gray--Scott diagnosis does not authorize R2 repair.")
    config = load_config(args.config)
    capacity = require_capacity(int(float(config["resources"]["stage_budget_gb"]) * 2**30))
    protocol = build_gray_scott_protocol(config)
    protocol["schema"] = "blrpng_gray_scott_r2_development_protocol_v1"
    protocol["design"].update({
        "source_role": "development_not_blind",
        "paired_resolution": 128,
        "paired_train_rollouts": int(config["data"]["paired_train_rollouts"]),
        "paired_validation_rollouts": int(config["data"]["paired_validation_rollouts"]),
        "input_feature_mode": config["model"]["input_feature_mode"],
        "query_generation_authorized": False,
    })
    from blrpng.paths import sha256_json
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = sha256_json(protocol)
    validation = validate_gray_scott_protocol(protocol)
    payload = {
        **protocol,
        "parent_diagnosis": str(diagnosis_path),
        "parent_diagnosis_sha256": sha256_file(diagnosis_path),
        "validation": validation,
    }
    payload.pop("protocol_sha256", None)
    payload["protocol_sha256"] = sha256_json(payload)
    with workspace_audit("freeze_gray_scott_r2_protocol") as audit:
        output = write_json("artifacts/manifests/gray_scott_r2_seed42_protocol.json", payload)
        preflight = write_json("artifacts/manifests/gray_scott_r2_storage_preflight.json", {
            "schema": "blrpng_storage_preflight_v1", **capacity,
            "estimated_components_gib": {
                "source_256": 3.0, "paired_source_128": 0.4,
                "long_lr_validation": 0.2, "checkpoints_and_logs": 14.4,
            },
        })
        audit.update({"output": str(output), "preflight": str(preflight)})
    print(json.dumps({"protocol": str(output), "validation": validation, "capacity": capacity}, indent=2))


if __name__ == "__main__":
    main()
