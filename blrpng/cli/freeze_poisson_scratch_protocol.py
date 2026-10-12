from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import write_json


def main() -> None:
    payload = {
        "schema": "blrpng_poisson_scratch_protocol_v1",
        "passed": True,
        "frozen": True,
        "seed": 42,
        "support_budget": 4,
        "representation": "lifted",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.000001,
        "epochs_max": 100,
        "minimum_epochs": 40,
        "early_stopping_patience": 15,
        "scheduler": "cosine",
        "minimum_learning_rate": 0.00001,
        "selection": "64x64 validation only",
        "query_used_to_define_or_select_scratch": False,
        "rationale": (
            "Give scratch five times the NLT epoch budget to rule out an "
            "under-converged scratch baseline; this strengthens the negative control."
        ),
        "retention": "metrics and hashes only after frozen-query evaluation",
    }
    with workspace_audit("freeze_poisson_scratch_protocol") as audit:
        output = write_json(
            "artifacts/manifests/poisson_negative_control_scratch_protocol.json",
            payload,
        )
        audit["output"] = str(output)
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
