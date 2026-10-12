from __future__ import annotations

import json

import torch

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    selection = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_target_selection.json", must_exist=True
    ).read_text(encoding="utf-8"))
    repair = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_highfreq_repair_outcome.json", must_exist=True
    ).read_text(encoding="utf-8"))
    methods = {
        "path_nlt": selection["checkpoints"]["0"]["path_nlt"],
        "path_nlt_lora": {"path": repair["checkpoint"], "sha256": repair["checkpoint_sha256"]},
    }
    records = []
    for resolution in (128, 256):
        index = (
            "artifacts/data/gray_scott_r2_seed42/highfreq_confirmation/"
            f"query_{resolution}.index.json"
        )
        for method, frozen in methods.items():
            if sha256_file(frozen["path"]) != frozen["sha256"]:
                raise RuntimeError("Confirmation checkpoint hash mismatch.")
            model, _ = load_checkpoint_model(frozen["path"], device=torch.device("cuda"))
            for horizon in (20, 50):
                metrics = projected_rollout_index(
                    model, index, device=torch.device("cuda"), horizon=horizon,
                    rollout_batch_size=4,
                )
                for initial_id, error in zip(
                    metrics["initial_condition_ids"], metrics["per_rollout_rel_l2"]
                ):
                    records.append({
                        "seed": 42, "boundary_id": 10000,
                        "initial_condition_id": int(initial_id),
                        "method": method, "relative_l2": float(error),
                        "resolution": resolution, "horizon": horizon,
                    })
            del model
            torch.cuda.empty_cache()
    comparisons = {}
    for resolution in (128, 256):
        for horizon in (20, 50):
            selected = [
                row for row in records
                if row["resolution"] == resolution and row["horizon"] == horizon
            ]
            comparisons[f"r{resolution}_h{horizon}"] = paired_hierarchical_bootstrap(
                selected, method="path_nlt_lora", baseline="path_nlt",
                replicates=50_000, seed=20260908 + resolution + horizon,
            )
    passed = (
        comparisons["r256_h20"]["relative_improvement"] > 0.05
        and comparisons["r256_h50"]["relative_improvement"] > 0.05
    )
    payload = {
        "schema": "blrpng_gray_scott_r2_highfreq_confirmation_v1",
        "frozen": True, "passed": passed, "records": records,
        "comparisons": comparisons,
        "promotion_rule": "Both 256-grid horizons must improve by more than 5%.",
        "repair_promoted": passed,
        "selection_used_this_query": False,
    }
    with workspace_audit("evaluate_gray_scott_r2_highfreq_confirmation") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_highfreq_confirmation.json", payload
        )
        audit.update({"output": str(output), "passed": passed})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
