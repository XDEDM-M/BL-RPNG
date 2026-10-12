from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.capacity import artifact_size_bytes, require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.poisson_protocol import build_poisson_protocol, validate_poisson_protocol
from blrpng.paths import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the seed42 Poisson negative-control protocol.")
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument(
        "--output", default="artifacts/manifests/poisson_negative_control_seed42_protocol.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    prerequisite = require_prerequisite_gate(config["experiment"]["prerequisite_gate"])
    # Conservative upper bound for all shards plus two compact source models.
    capacity = require_capacity(int(2.8 * 2**30))
    protocol = build_poisson_protocol(config)
    validation = validate_poisson_protocol(protocol)
    with workspace_audit("freeze_poisson_protocol") as audit:
        output = write_json(args.output, protocol)
        preflight = write_json(
            "artifacts/manifests/poisson_negative_control_storage_preflight.json",
            {
                "schema": "blrpng_storage_preflight_v1",
                **capacity,
                "artifact_bytes_at_freeze": artifact_size_bytes(),
                "stage_budget_gb": float(config["resources"]["stage_budget_gb"]),
            },
        )
        audit.update({"protocol": str(output), "preflight": str(preflight)})
    print(json.dumps({
        "protocol": str(output), "protocol_sha256": protocol["protocol_sha256"],
        "validation": validation, "prerequisite_stage": prerequisite.get("next_stage"),
        "capacity": capacity,
    }, indent=2))


if __name__ == "__main__":
    main()
