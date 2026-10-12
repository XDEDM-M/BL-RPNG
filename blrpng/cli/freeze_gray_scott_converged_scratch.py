from __future__ import annotations

import json
from statistics import mean

import torch

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    require_prerequisite_gate(
        "artifacts/manifests/gray_scott_seed42_converged_lr_summary.json"
    )
    rows = []
    hashes = []
    for task in range(16):
        path = resolve_in_project(
            f"artifacts/checkpoints/gray_scott/"
            f"gray_scott_task{task:02d}_m4_s42_lifted_scratch_e100.inference.pt",
            must_exist=True,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        history = checkpoint["history"]
        validation = [float(row["validation"]["rollout_rel_l2"]) for row in history]
        if len(history) < 60:
            raise RuntimeError(f"Scratch task {task} stopped before 60 epochs.")
        earlier, later = mean(validation[-20:-10]), mean(validation[-10:])
        rows.append({
            "task": task,
            "epochs": len(history),
            "best_epoch": min(range(len(validation)), key=validation.__getitem__),
            "best_validation_metric": float(checkpoint["best_validation_metric"]),
            "relative_late_improvement": (earlier - later) / earlier,
        })
        hashes.append(sha256_file(path))
    payload = {
        "schema": "blrpng_gray_scott_converged_scratch_freeze_v1",
        "frozen": True,
        "passed": True,
        "seed": 42,
        "query_used": False,
        "mean_validation_metric": mean(row["best_validation_metric"] for row in rows),
        "runs": rows,
        "checkpoint_sha256": hashes,
        "authorized_next_experiment": "Generate the unopened 128-grid confirmation query.",
    }
    with workspace_audit("freeze_gray_scott_converged_scratch") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_converged_scratch_freeze.json",
            payload,
        )
        audit.update({"output": str(output), "query_used": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
