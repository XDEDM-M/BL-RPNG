from __future__ import annotations

import argparse
import json
import time

import torch

from blrpng.data.capacity import require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import resolve_in_project, sha256_file, write_json


TASKS = (0, 4, 8, 12)
METHODS = ("final_nlt", "path_nlt", "final_nlt_multiscale", "path_nlt_multiscale")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Gray--Scott R2 blind query.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    gate = require_prerequisite_gate(
        "artifacts/manifests/gray_scott_r2_blind_query_data_gate.json"
    )
    if gate.get("query_used_for_selection") is not False:
        raise PermissionError("R2 blind-query data gate is not selection locked.")
    selection = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_target_selection.json", must_exist=True
    ).read_text(encoding="utf-8"))
    protocol = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_blind_query_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    if not selection.get("frozen") or protocol.get("selection_used_query") is not False:
        raise PermissionError("R2 blind-query selection lock is invalid.")
    device = torch.device(args.device)
    records = []
    model_reports = []
    with workspace_audit(f"evaluate_gray_scott_r2_query{args.resolution}") as audit:
        for task in TASKS:
            index_path = (
                "artifacts/data/gray_scott_r2_seed42/blind_query/"
                f"task{task:02d}_query_{args.resolution}.index.json"
            )
            index = read_index(index_path, verify_hashes=True)
            for method in METHODS:
                frozen = selection["checkpoints"][str(task)][method]
                if sha256_file(frozen["path"]) != frozen["sha256"]:
                    raise RuntimeError("Frozen R2 adapter hash mismatch.")
                model, _ = load_checkpoint_model(frozen["path"], device=device)
                for horizon in (20, 50):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    metrics = projected_rollout_index(
                        model, index_path, device=device, horizon=horizon,
                        rollout_batch_size=4,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    for initial_id, error in zip(
                        metrics["initial_condition_ids"], metrics["per_rollout_rel_l2"]
                    ):
                        records.append({
                            "seed": 42,
                            "target_task_id": task,
                            "boundary_id": int(index["shards"][0]["boundary_id"]),
                            "initial_condition_id": int(initial_id),
                            "boundary_regime": index["metadata"]["boundary_regime"],
                            "method": method,
                            "relative_l2": float(error),
                            "resolution": args.resolution,
                            "horizon": horizon,
                        })
                    model_reports.append({
                        "task": task,
                        "method": method,
                        "horizon": horizon,
                        "mean_relative_l2": metrics["rollout_rel_l2"],
                        "rollout_mse": metrics["rollout_mse"],
                        "evaluation_seconds": elapsed,
                        "checkpoint_sha256": frozen["sha256"],
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        output = write_json(
            f"artifacts/reports/gray_scott/r2_blind_query{args.resolution}.json",
            {
                "schema": "blrpng_gray_scott_r2_blind_evaluation_v1",
                "frozen": True,
                "resolution": args.resolution,
                "selected_method": selection["selected_method"],
                "selection_used_this_query": False,
                "records": records,
                "model_reports": model_reports,
            },
        )
        audit.update({"output": str(output), "records": len(records)})
    means = {
        f"{method}_h{horizon}": sum(
            row["relative_l2"] for row in records
            if row["method"] == method and row["horizon"] == horizon
        ) / sum(
            row["method"] == method and row["horizon"] == horizon for row in records
        )
        for method in METHODS for horizon in (20, 50)
    }
    print(json.dumps({"output": str(output), "means": means}, indent=2))


if __name__ == "__main__":
    main()
