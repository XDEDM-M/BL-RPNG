from __future__ import annotations

import json
import random

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_json, write_json


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated")


def main() -> None:
    query_gate_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_query128_evaluation_gate.json",
        must_exist=True,
    )
    query_gate = json.loads(query_gate_path.read_text(encoding="utf-8"))
    if query_gate.get("passed") is not False:
        raise RuntimeError("Convergence repair is only authorized after the failed blind gate.")
    best_epochs = {method: [] for method in METHODS}
    for method in METHODS:
        for task in range(16):
            path = resolve_in_project(
                f"artifacts/checkpoints/gray_scott/"
                f"gray_scott_task{task:02d}_m4_s42_{method}.adapter.pt",
                must_exist=True,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            history = payload["metadata"]["history"]
            values = [float(row["validation"]["rollout_rel_l2"]) for row in history]
            best_epochs[method].append(min(range(len(values)), key=values.__getitem__))
    if any(epoch != 19 for rows in best_epochs.values() for epoch in rows):
        raise RuntimeError("The frozen under-convergence diagnosis no longer matches artifacts.")

    original_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_protocol.json", must_exist=True
    )
    original = json.loads(original_path.read_text(encoding="utf-8"))
    rng = random.Random(20260903)
    confirmation_tasks = []
    for target in original["targets"]:
        task = int(target["target_task_id"])
        query = []
        for sample in range(8):
            pair_id = 3_000_000 + task * 1_000 + 200 + sample
            query.append({
                "case_id": pair_id,
                "pair_id": pair_id,
                "initial_condition_id": 300_000 + task * 1_000 + 200 + sample,
                "initial_seed": rng.randrange(1, 2**31 - 1),
                "role": "target_confirmation_query",
            })
        confirmation_tasks.append({
            key: target[key] for key in (
                "target_task_id", "boundary_id", "boundary_seed", "boundary_regime",
                "correlation", "amplitude", "feed", "kill"
            )
        } | {"query": query})
    confirmation = {
        "schema": "blrpng_gray_scott_confirmation_query_protocol_v1",
        "frozen": True,
        "seed": 42,
        "split_seed": 20260903,
        "parent_protocol_sha256": original["protocol_sha256"],
        "resolution_sequence": [128, 256],
        "steps": 50,
        "query_generated": False,
        "tasks": confirmation_tasks,
    }
    confirmation["protocol_sha256"] = sha256_json(confirmation)
    repair = {
        "schema": "blrpng_gray_scott_convergence_repair_v1",
        "frozen": True,
        "passed": True,
        "seed": 42,
        "opened_query_used_only_for_failure_diagnosis": True,
        "underconvergence_evidence": {
            "all_48_adapters_best_at_last_epoch": True,
            "best_epochs": best_epochs,
            "old_epochs": 20,
            "old_optimizer_steps": 200,
        },
        "training": {
            "epochs": 100,
            "minimum_epochs": 60,
            "early_stopping_patience": 15,
            "lr_drop_epoch": 90,
            "lr_drop_gamma": 0.2,
            "batch_size": 4,
            "seed": 42,
            "support_budget": 4,
            "checkpoint_selection": "64-grid target validation only",
        },
        "methods": list(METHODS),
        "confirmation_query_must_remain_unopened_until_lr_checkpoints_frozen": True,
    }
    with workspace_audit("freeze_gray_scott_convergence_repair") as audit:
        repair_output = write_json(
            "artifacts/manifests/gray_scott_seed42_convergence_repair.json", repair
        )
        query_output = write_json(
            "artifacts/manifests/gray_scott_seed42_confirmation_query_protocol.json",
            confirmation,
        )
        audit.update({
            "repair": str(repair_output), "confirmation": str(query_output),
            "confirmation_query_generated": False,
        })
    print(json.dumps({
        "repair": str(repair_output), "confirmation": str(query_output),
        "all_48_best_at_epoch_19": True,
        "confirmation_units": 16 * 8,
    }, indent=2))


if __name__ == "__main__":
    main()
