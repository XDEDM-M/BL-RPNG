from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    relative = "artifacts/manifests/gray_scott_seed42_converged_scratch_freeze.json"
    source = json.loads(resolve_in_project(relative, must_exist=True).read_text(encoding="utf-8"))
    maximum = max(float(row["relative_late_improvement"]) for row in source["runs"])
    near_end = [
        int(row["task"]) for row in source["runs"]
        if int(row["best_epoch"]) >= int(row["epochs"]) - 5
    ]
    payload = {
        "schema": "blrpng_gray_scott_scratch_saturation_supersession_v1",
        "frozen": True,
        "passed": False,
        "supersedes_claim_interpretation_of": relative,
        "retained_interpretation": "completed 100-epoch matched scratch training",
        "invalid_interpretation": "converged or saturated scratch baseline",
        "maximum_relative_late_improvement": maximum,
        "tasks_with_best_epoch_in_last_five": near_end,
        "query_opening_authorized": False,
        "authorized_next_experiment": (
            "Continue each frozen e100 scratch checkpoint with low learning rate for up to "
            "200 additional epochs, selected only by the same 64-grid validation split."
        ),
    }
    with workspace_audit("supersede_gray_scott_scratch_freeze") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_scratch_saturation_supersession.json",
            payload,
        )
        audit.update({"output": str(output), "query_opening_authorized": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
