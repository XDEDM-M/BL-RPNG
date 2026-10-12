from __future__ import annotations

import json
from statistics import mean, median

import torch

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def _families() -> dict[int, str]:
    path = resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["target_task_id"]): str(row["family"])
        for row in payload["tasks"]
    }


def _validation(payload: dict) -> tuple[int, dict]:
    history = payload.get("history", payload.get("metadata", {}).get("history", []))
    if len(history) != 100 and payload.get("format") == "blrpng_checkpoint_v1":
        raise ValueError("Scratch convergence checkpoint must contain exactly 100 epochs.")
    best_index = min(
        range(len(history)),
        key=lambda index: float(history[index]["validation"]["rollout_rel_l2"]),
    )
    return best_index, history[best_index]["validation"]


def main() -> None:
    families = _families()
    records = []
    rows = []
    best_epochs = []
    last_ten_gains = []
    for task, family in sorted(families.items()):
        scratch_path = resolve_in_project(
            "artifacts/checkpoints/reconfirmation/"
            f"reconfirmation_task{task:02d}_m4_s42_lr_scratch.inference.pt",
            must_exist=True,
        )
        nlt_path = resolve_in_project(
            "artifacts/checkpoints/reconfirmation/"
            f"reconfirmation_task{task:02d}_m4_s42_nlt.adapter.pt",
            must_exist=True,
        )
        scratch = torch.load(scratch_path, map_location="cpu", weights_only=False)
        if scratch.get("format") != "blrpng_checkpoint_v1":
            raise ValueError(f"Unexpected scratch checkpoint: {scratch_path}")
        if scratch.get("inference_only") is not True:
            raise ValueError(f"Scratch checkpoint was not compacted: {scratch_path}")
        if scratch.get("adapter", {}).get("strategy") != "full":
            raise ValueError(f"Scratch checkpoint is not a full model: {scratch_path}")
        # This verifies model construction and exact state compatibility, not
        # merely that torch can deserialize the container.
        load_checkpoint_model(scratch_path, device=torch.device("cpu"))
        nlt = torch.load(nlt_path, map_location="cpu", weights_only=False)
        if nlt.get("format") != "blrpng_adapter_delta_v1" or not nlt.get("adapter_state"):
            raise ValueError(f"Unexpected NLT adapter: {nlt_path}")
        scratch_best, scratch_validation = _validation(scratch)
        nlt_best, nlt_validation = _validation(nlt)
        best_epochs.append(scratch_best + 1)
        history = scratch["history"]
        if scratch_best == len(history) - 1:
            previous = float(history[-11]["validation"]["rollout_rel_l2"])
            current = float(history[-1]["validation"]["rollout_rel_l2"])
            last_ten_gains.append((previous - current) / max(previous, 1e-15))
        scratch_stratum = scratch_validation["strata"][0]
        nlt_stratum = nlt_validation["strata"][0]
        validation_index = read_index(
            f"artifacts/data/reconfirmation/task{task:02d}_validation_64_all.index.json"
        )
        scratch_ids = [
            int(row["initial_condition_id"]) for row in validation_index["shards"]
        ]
        nlt_ids = [int(value) for value in nlt_stratum["initial_condition_ids"]]
        if scratch_ids != nlt_ids:
            raise ValueError(f"Unpaired validation initial conditions for task {task}.")
        for initial_id, scratch_value, nlt_value in zip(
            scratch_ids,
            scratch_stratum["per_rollout_rel_l2"],
            nlt_stratum["per_rollout_rel_l2"],
        ):
            for method, value in (("lr_scratch", scratch_value), ("nlt", nlt_value)):
                records.append({
                    "seed": 42,
                    "boundary_id": task,
                    "initial_condition_id": initial_id,
                    "method": method,
                    "relative_l2": float(value),
                })
        rows.append({
            "task": task,
            "family": family,
            "scratch_checkpoint": str(scratch_path.relative_to(PROJECT_ROOT)),
            "scratch_sha256": sha256_file(scratch_path),
            "nlt_checkpoint": str(nlt_path.relative_to(PROJECT_ROOT)),
            "nlt_sha256": sha256_file(nlt_path),
            "scratch_best_epoch": scratch_best + 1,
            "nlt_best_epoch": nlt_best + 1,
            "scratch_validation_rel_l2": float(scratch_validation["rollout_rel_l2"]),
            "nlt_validation_rel_l2": float(nlt_validation["rollout_rel_l2"]),
            "scratch_training_seconds": float(scratch.get("elapsed_seconds", 0.0)),
            "hr_query_used": False,
        })
        del scratch, nlt
    comparison = paired_hierarchical_bootstrap(
        records,
        method="nlt",
        baseline="lr_scratch",
        replicates=50_000,
        seed=20260902,
    )
    convergence = {
        "median_best_epoch": median(best_epochs),
        "best_at_epoch_100": sum(value == 100 for value in best_epochs),
        "mean_last10_relative_gain_when_best_at_100": (
            mean(last_ten_gains) if last_ten_gains else None
        ),
        "max_last10_relative_gain_when_best_at_100": (
            max(last_ten_gains) if last_ten_gains else None
        ),
    }
    payload = {
        "schema": "blrpng_reconfirmation_scratch_freeze_v1",
        "seed": 42,
        "support_budget": 4,
        "epochs": 100,
        "learning_rate": 1e-4,
        "scheduler": "cosine_min_1e-6",
        "selection_data": "64x64 LR validation only",
        "query_opened_for_selection": False,
        "comparison": comparison,
        "convergence": convergence,
        "checkpoint_rows": rows,
    }
    with workspace_audit("freeze_reconfirmation_scratch") as audit:
        report = write_json(
            "artifacts/reports/allen_cahn_reconfirmation_lr_scratch_freeze.json",
            payload,
        )
        freeze = write_json(
            "artifacts/manifests/allen_cahn_reconfirmation_lr_scratch_freeze.json",
            {
                "schema": "blrpng_reconfirmation_scratch_checkpoint_freeze_v1",
                "passed": True,
                "hr_query_used": False,
                "report": str(report.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(report),
                "checkpoints": [
                    {
                        "task": row["task"],
                        "path": row["scratch_checkpoint"],
                        "sha256": row["scratch_sha256"],
                    }
                    for row in rows
                ],
            },
        )
        audit.update({"report": str(report), "freeze": str(freeze)})
    print(json.dumps({
        "report": str(report),
        "freeze": str(freeze),
        "comparison": comparison,
        "convergence": convergence,
    }, indent=2))


if __name__ == "__main__":
    main()
