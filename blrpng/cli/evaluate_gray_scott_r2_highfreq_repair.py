from __future__ import annotations

import json

import torch

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    outcome = require_prerequisite_gate(
        "artifacts/manifests/gray_scott_r2_highfreq_repair_outcome.json"
    )
    selection = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_target_selection.json", must_exist=True
    ).read_text(encoding="utf-8"))
    candidates = {
        "path_nlt": selection["checkpoints"]["0"]["path_nlt"],
        "path_nlt_lora": {
            "path": outcome["checkpoint"], "sha256": outcome["checkpoint_sha256"]
        },
    }
    records = []
    with workspace_audit("evaluate_gray_scott_r2_highfreq_repair") as audit:
        for resolution in (128, 256):
            index = (
                "artifacts/data/gray_scott_r2_seed42/blind_query/"
                f"task00_query_{resolution}.index.json"
            )
            for method, frozen in candidates.items():
                if sha256_file(frozen["path"]) != frozen["sha256"]:
                    raise RuntimeError("High-frequency diagnostic checkpoint hash mismatch.")
                model, _ = load_checkpoint_model(frozen["path"], device=torch.device("cuda"))
                for horizon in (20, 50):
                    metrics = projected_rollout_index(
                        model, index, device=torch.device("cuda"), horizon=horizon,
                        rollout_batch_size=4,
                    )
                    records.append({
                        "resolution": resolution,
                        "horizon": horizon,
                        "method": method,
                        "mean_relative_l2": metrics["rollout_rel_l2"],
                        "per_rollout_rel_l2": metrics["per_rollout_rel_l2"],
                    })
                del model
                torch.cuda.empty_cache()
        comparisons = {}
        for resolution in (128, 256):
            for horizon in (20, 50):
                values = {
                    row["method"]: row["mean_relative_l2"] for row in records
                    if row["resolution"] == resolution and row["horizon"] == horizon
                }
                comparisons[f"r{resolution}_h{horizon}"] = {
                    "path_nlt": values["path_nlt"],
                    "path_nlt_lora": values["path_nlt_lora"],
                    "relative_improvement": 1.0 - values["path_nlt_lora"] / values["path_nlt"],
                }
        passed = all(row["relative_improvement"] > 0 for row in comparisons.values())
        payload = {
            "schema": "blrpng_gray_scott_r2_highfreq_repair_diagnostic_v1",
            "frozen": True,
            "passed": passed,
            "exploratory_after_query": True,
            "cannot_be_used_as_blind_confirmation": True,
            "records": records,
            "comparisons": comparisons,
            "new_query_authorized": passed,
        }
        output = write_json(
            "artifacts/reports/gray_scott/r2_highfreq_repair_diagnostic.json", payload
        )
        audit.update({"output": str(output), "passed": passed})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
