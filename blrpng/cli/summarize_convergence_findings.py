from __future__ import annotations

import json
from pathlib import Path

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def _load(path: str) -> dict:
    resolved = resolve_in_project(path, must_exist=True)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _mean(records: list[dict], method: str, task: int = 0) -> float:
    values = [
        float(row["relative_l2"]) for row in records
        if row["method"] == method and int(row["target_task_id"]) == task
    ]
    if not values:
        raise KeyError((method, task))
    return sum(values) / len(values)


def main() -> None:
    old = _load("artifacts/reports/v2_seed42_query256_h20.json")
    adapter = _load("artifacts/reports/adapter_convergence_commutator_task0_query256_h20.json")
    old_nlt = _mean(old["records"], "nlt")
    old_scratch = _mean(old["records"], "lr_scratch")
    new_nlt = _mean(adapter["records"], "nlt")
    new_scratch = _mean(adapter["records"], "scratch")
    commutator = _mean(adapter["records"], "nlt_commutator")
    weight_reports = {}
    for label, path in {
        "0.2": "artifacts/reports/task0_nlt_commutator_cw02_e100_query256_h20.json",
        "1.0": "artifacts/reports/task0_nlt_commutator_cw10_e100_query256_h20.json",
        "5.0": "artifacts/reports/task0_nlt_commutator_cw50_e100_query256_h20.json",
    }.items():
        resolved = resolve_in_project(path)
        if resolved.exists():
            payload = _load(path)
            weight_reports[label] = {
                "query256_h20_relative_l2": float(payload["metrics"]["primary_rollout_rel_l2"]),
                "source_sha256": sha256_file(resolved),
            }
    hr_rows = []
    for path in sorted(resolve_in_project("artifacts/checkpoints/v2").glob(
        "v2_task00_m64_s42_hr_oracle*conv*e100.pt"
    )):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        history = payload["history"]
        hr_rows.append({
            "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            "initialization": (
                "ordinary" if "hr_oracle_ordinary" in path.name
                else "path" if "hr_oracle_path" in path.name else "scratch"
            ),
            "best_validation_relative_l2": float(payload["best_validation_metric"]),
            "best_epoch": min(
                range(len(history)),
                key=lambda index: history[index]["validation"]["rollout_rel_l2"],
            ),
            "final_validation_relative_l2": float(history[-1]["validation"]["rollout_rel_l2"]),
            "epochs": len(history),
        })
    payload = {
        "schema": "blrpng_convergence_findings_v1",
        "status": "exploratory_current_task_current_query_not_blind",
        "task": 0, "resolution": 256, "horizon": 20,
        "adapter_query": {
            "old_20_epoch_nlt": old_nlt,
            "old_20_epoch_scratch": old_scratch,
            "validation_selected_100_epoch_nlt": new_nlt,
            "validation_selected_100_epoch_scratch": new_scratch,
            "validation_selected_100_epoch_commutator_0.05": commutator,
            "scratch_100epoch_improvement_over_20epoch": 1.0 - new_scratch / old_scratch,
            "nlt_100epoch_change_over_20epoch": 1.0 - new_nlt / old_nlt,
            "nlt_improvement_over_convergence_audited_scratch": 1.0 - new_nlt / new_scratch,
            "commutator_improvement_over_100epoch_nlt": 1.0 - commutator / new_nlt,
            "weight_sweep": weight_reports,
        },
        "hr_validation": hr_rows,
        "interpretation": [
            "The old HR specialist was materially optimization-limited.",
            "The LR scratch baseline also improves with training, but NLT retains over 75% gain on task 0.",
            "Long NLT adaptation improves LR validation but harms 256-grid transfer, indicating resolution overfitting.",
            "Commutator regularization partly mitigates, but does not yet beat the old 20-epoch NLT on current query.",
            "No result here is eligible as a fresh blind or saturated-oracle claim.",
        ],
        "sources": {
            "old_matrix": sha256_file(resolve_in_project("artifacts/reports/v2_seed42_query256_h20.json")),
            "adapter_audit": sha256_file(resolve_in_project(
                "artifacts/reports/adapter_convergence_commutator_task0_query256_h20.json"
            )),
        },
    }
    with workspace_audit("summarize_convergence_findings") as audit:
        output = write_json("artifacts/reports/convergence_findings_task0_exploratory.json", payload)
        audit["output"] = str(output)
    print(json.dumps({"output": str(output), "adapter_query": payload["adapter_query"],
                      "hr_runs": len(hr_rows)}, indent=2))


if __name__ == "__main__":
    main()

