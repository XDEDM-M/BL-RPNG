from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.blind_boundaries import write_reconfirmation_split_manifest
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a locked Allen-Cahn reconfirmation split.")
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    parser.add_argument("--role", choices=("support", "validation", "query", "query_512"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    gates = config["gates"]
    require_prerequisite_gate(gates["required_oracle_manifest"])
    if args.role.startswith("query"):
        require_prerequisite_gate(gates["required_method_freeze_manifest"])
        require_prerequisite_gate(gates["required_checkpoint_freeze_manifest"])
        require_prerequisite_gate(gates["required_query_protocol_freeze_manifest"])
    counts = {
        "support": int(config["blind_protocol"]["support_initial_conditions"]),
        "validation": int(config["blind_protocol"]["validation_initial_conditions"]),
        "query": int(config["blind_protocol"]["query_initial_conditions"]),
        "query_512": int(config["blind_protocol"]["query_512_initial_conditions"]),
    }
    with workspace_audit("prepare_reconfirmation_split") as audit:
        output = write_reconfirmation_split_manifest(
            "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
            role=args.role,
            initial_conditions=counts[args.role],
        )
        audit.update({"role": args.role, "output": output})
    print(json.dumps({"role": args.role, "manifest": output, "contains_solutions": False}, indent=2))


if __name__ == "__main__":
    main()
