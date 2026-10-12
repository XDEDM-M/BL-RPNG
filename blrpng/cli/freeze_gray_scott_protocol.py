from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.gray_scott_protocol import build_gray_scott_protocol, validate_gray_scott_protocol
from blrpng.paths import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Gray--Scott seed42 coupled-field protocol.")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    prerequisite = require_prerequisite_gate(config["experiment"]["prerequisite_gate"])
    capacity = require_capacity(int(float(config["resources"]["stage_budget_gb"]) * 2**30))
    protocol = build_gray_scott_protocol(config)
    validation = validate_gray_scott_protocol(protocol)
    with workspace_audit("freeze_gray_scott_protocol") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_protocol.json", protocol
        )
        preflight = write_json(
            "artifacts/manifests/gray_scott_seed42_storage_preflight.json",
            {"schema": "blrpng_storage_preflight_v1", **capacity},
        )
        audit.update({"output": str(output), "preflight": str(preflight)})
    print(json.dumps({
        "protocol": str(output), "protocol_sha256": protocol["protocol_sha256"],
        "validation": validation, "prerequisite_next_stage": prerequisite.get("next_stage"),
        "capacity": capacity,
    }, indent=2))


if __name__ == "__main__":
    main()
