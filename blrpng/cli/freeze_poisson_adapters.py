from __future__ import annotations

import json

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    rows = []
    for task in range(30):
        for method in ("ordinary_nlt", "path_nlt"):
            path = (
                f"artifacts/checkpoints/poisson/"
                f"poisson_task{task:02d}_m4_s42_{method}.adapter.pt"
            )
            resolved = resolve_in_project(path, must_exist=True)
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
            if payload.get("format") != "blrpng_adapter_delta_v1":
                raise ValueError(f"Invalid adapter: {resolved}")
            base = resolve_in_project(payload["base_checkpoint"], must_exist=True)
            if sha256_file(base) != payload["base_checkpoint_sha256"]:
                raise ValueError(f"Changed base checkpoint for {resolved}")
            if not payload.get("adapter_state"):
                raise ValueError(f"Empty NLT state: {resolved}")
            rows.append({
                "task": task,
                "method": method,
                "path": path,
                "sha256": sha256_file(resolved),
                "base_checkpoint": payload["base_checkpoint"],
                "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
                "best_validation_metric": float(
                    payload["metadata"]["best_validation_metric"]
                ),
                "elapsed_seconds": float(payload["metadata"]["elapsed_seconds"]),
            })
    payload = {
        "schema": "blrpng_poisson_adapter_freeze_v1",
        "passed": True,
        "frozen": True,
        "seed": 42,
        "support_budget": 4,
        "selection_data": "64x64 target validation only",
        "hr_query_used": False,
        "adapters": rows,
    }
    with workspace_audit("freeze_poisson_adapters") as audit:
        output = write_json(
            "artifacts/manifests/poisson_negative_control_seed42_adapter_freeze.json",
            payload,
        )
        audit.update({"output": str(output), "adapter_count": len(rows)})
    print(json.dumps({
        "output": str(output), "adapter_count": len(rows),
        "validation_means": {
            method: sum(row["best_validation_metric"] for row in rows if row["method"] == method) / 30
            for method in ("ordinary_nlt", "path_nlt")
        },
        "hr_query_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
