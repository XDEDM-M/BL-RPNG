from __future__ import annotations

import json
from statistics import mean

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def _summary(path: str) -> dict:
    resolved = resolve_in_project(path, must_exist=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    validation = [
        float(row["validation"]["rollout_rel_l2"])
        for row in checkpoint["history"]
    ]
    earlier, later = mean(validation[-20:-10]), mean(validation[-10:])
    return {
        "path": path,
        "sha256": sha256_file(resolved),
        "best_validation_metric": float(checkpoint["best_validation_metric"]),
        "epochs": len(validation),
        "best_epoch": min(range(len(validation)), key=validation.__getitem__),
        "relative_late_improvement": (earlier - later) / earlier,
        "elapsed_seconds": float(checkpoint["elapsed_seconds"]),
    }


def main() -> None:
    supersession = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_scratch_saturation_supersession.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    if supersession.get("query_opening_authorized") is not False:
        raise PermissionError("Scratch saturation must remain query locked.")
    tasks = []
    continuation_late = []
    for task in range(16):
        prefix = (
            f"artifacts/checkpoints/gray_scott/gray_scott_task{task:02d}_"
            "m4_s42_lifted_scratch_"
        )
        e100 = _summary(prefix + "e100.inference.pt")
        e300 = _summary(prefix + "e300.inference.pt")
        selected = min((e100, e300), key=lambda row: row["best_validation_metric"])
        continuation_late.append(e300["relative_late_improvement"])
        tasks.append({
            "task": task,
            "e100": e100,
            "e300_continuation": e300,
            "selected_path": selected["path"],
            "selected_sha256": selected["sha256"],
            "selected_stage": "e100" if selected is e100 else "e300",
            "selected_validation_metric": selected["best_validation_metric"],
            "total_training_seconds": e100["elapsed_seconds"] + e300["elapsed_seconds"],
        })
    maximum_late = max(continuation_late)
    passed = maximum_late <= 0.02
    payload = {
        "schema": "blrpng_gray_scott_saturated_scratch_freeze_v1",
        "frozen": True,
        "passed": passed,
        "seed": 42,
        "query_used": False,
        "selection_rule": "minimum 64-grid validation error across e100 and e300 continuation",
        "mean_selected_validation_metric": mean(
            row["selected_validation_metric"] for row in tasks
        ),
        "maximum_continuation_late_improvement": maximum_late,
        "continuation_plateau_threshold": 0.02,
        "tasks": tasks,
        "authorized_next_experiment": (
            "Generate the unopened 128-grid confirmation query." if passed else None
        ),
    }
    with workspace_audit("freeze_gray_scott_saturated_scratch") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_saturated_scratch_freeze.json",
            payload,
        )
        audit.update({"output": str(output), "passed": passed, "query_used": False})
    if not passed:
        raise RuntimeError(f"Scratch continuation has not plateaued: {maximum_late}")
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
