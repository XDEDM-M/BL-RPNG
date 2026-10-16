#!/usr/bin/env python3
"""Freeze all LR-selected rotation models and fresh query plans before HR generation/use."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = ROOT.parent / "NS-PIMPLE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(NS_ROOT / "scripts"))
from blrpng.paths import sha256_file, write_json  # noqa: E402

import run_ns_pimple_method_v1 as method  # noqa: E402
import generate_method_v2_confirmation as v2_generator  # noqa: E402


SOURCE_ARMS = method.SOURCE_ARMS
TARGET_ARMS = (*SOURCE_ARMS, "lifted_lr_scratch")
FREEZE_PATH = method.REPORT_ROOT / "rotation_confirmation_freeze.json"


def h5_row_index(manifest: dict, role: str, level: str) -> dict[str, dict]:
    return {
        row["pair_id"]: row
        for row in manifest.get("trajectories", [])
        if row.get("role") == role and row.get("level") == level
    }


def main() -> int:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"rotation confirmation is already frozen: {FREEZE_PATH}")
    report_path = method.REPORT_ROOT / "rotation_source_lr_diagnostic.json"
    report = method.load_json(report_path)
    prereg_path = ROOT / "docs/prereg/ns_pimple_method_v1_rotation_source_lr_diagnostic.json"
    if report.get("preregistration_sha256") != sha256_file(prereg_path):
        raise ValueError("rotation LR report does not match its frozen preregistration")
    if report.get("target_hr_query_used") is not False or report.get("sealed_hr_claim") is not False:
        raise ValueError("rotation LR report indicates HR query access")
    if set(report.get("source_arms", {})) != set(SOURCE_ARMS):
        raise ValueError("not all three rotation-augmented source arms are complete")
    if set(report.get("target_tasks", {})) != set(method.TASKS):
        raise ValueError("not all four target tasks are complete")
    for task in method.TASKS:
        if set(report["target_tasks"][task]) != set(TARGET_ARMS):
            raise ValueError(f"target comparison is incomplete for {task}")

    v0_path = NS_ROOT / "data/v0/manifest.json"
    v1_path = NS_ROOT / "data/method_v1/manifest.json"
    v2_path = NS_ROOT / "data/method_v2/manifest.json"
    v0, v1, v2 = (method.load_json(path) for path in (v0_path, v1_path, v2_path))
    v2_validation = h5_row_index(v2, "validation", "lr")
    if len(v2_validation) != 8 or any(row["status"] != "complete" for row in v2_validation.values()):
        raise ValueError("fresh V2 LR validation data are incomplete")
    if h5_row_index(v2, "query", "hr"):
        raise ValueError("V2 HR query trajectories must be generated only after this freeze")
    validation_pairs = set(v2_validation)
    expected_validation = {
        f"{task}_{ic}" for task in method.TASKS for ic in ("validation_mode25", "validation_mode52")
    }
    if validation_pairs != expected_validation:
        raise ValueError("V2 validation IC set differs from the frozen rotation protocol")

    v1_validation = h5_row_index(v1, "validation", "lr")
    v1_task_validation = {
        pair_id: row for pair_id, row in v1_validation.items()
        if any(pair_id.startswith(task + "_") for task in method.TASKS)
    }
    if len(v1_task_validation) != 16:
        raise ValueError("expected the 16 existing target LR validation trajectories")
    for task in method.TASKS:
        expected_ids = {
            pair_id for pair_id in v1_task_validation
            if pair_id.startswith(task + "_")
        } | {
            pair_id for pair_id in v2_validation if pair_id.startswith(task + "_")
        }
        if set(report["target_validation_pair_ids"].get(task, [])) != expected_ids:
            raise ValueError(f"LR-selected model validation ICs differ from the frozen set: {task}")
    validation_hashes = {}
    for row in (*v1_task_validation.values(), *v2_validation.values()):
        validation_hashes[row["pair_id"]] = {
            "level": row["level"], "role": row["role"],
            "config_sha256": row["config_sha256"], "hdf5_sha256": row["hdf5_sha256"],
        }

    v0_payload = json.loads(v0_path.read_text())
    query_plans = [row for row in v2_generator.build_rows(v0_payload) if row["role"] == "query"]
    if len(query_plans) != 8:
        raise ValueError("expected eight fresh planned HR queries")
    if any((NS_ROOT / row["hdf5"]).exists() for row in query_plans):
        raise ValueError("a fresh HR query HDF5 already exists before model freeze")

    source_checkpoints = {}
    for arm, row in report["source_arms"].items():
        path = ROOT / row["checkpoint"]
        if sha256_file(path) != row["checkpoint_sha256"]:
            raise ValueError(f"rotation source checkpoint hash mismatch: {arm}")
        source_checkpoints[arm] = {"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}
    target_checkpoints = {}
    for task in method.TASKS:
        target_checkpoints[task] = {}
        for arm in TARGET_ARMS:
            row = report["target_tasks"][task][arm]
            path = ROOT / row["checkpoint"]
            if sha256_file(path) != row["checkpoint_sha256"]:
                raise ValueError(f"rotation target checkpoint hash mismatch: {task}/{arm}")
            target_checkpoints[task][arm] = {"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}

    original_checkpoints = {}
    old_target_report = method.load_json(method.REPORT_ROOT / "target_training.json")
    for task in method.TASKS:
        original_checkpoints[task] = {}
        for arm in TARGET_ARMS:
            path = method.CHECKPOINT_ROOT / f"target_{task}_{arm}_seed42.pt"
            old_row = old_target_report["tasks"][task][arm]
            if sha256_file(path) != old_row["checkpoint_sha256"]:
                raise ValueError(f"original V1 checkpoint hash mismatch: {task}/{arm}")
            original_checkpoints[task][arm] = {
                "path": str(path.relative_to(ROOT)), "sha256": old_row["checkpoint_sha256"]
            }

    data_hashes = {
        "source_train": sha256_file(method.DATA_ROOT / "source_path_pretraining.pt"),
        "source_validation_rollouts": sha256_file(method.DATA_ROOT / "source_validation_rollouts.pt"),
        "dataset_index": sha256_file(method.DATA_ROOT / "dataset_index.json"),
        "target_support": {
            task: sha256_file(method.DATA_ROOT / f"{task}_support_final_only.pt") for task in method.TASKS
        },
    }
    freeze = {
        "schema": "blrpng_ns_pimple_rotation_confirmation_freeze_v1",
        "status": "frozen_before_fresh_hr_query_generation_and_field_access",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": sha256_file(prereg_path),
        "lr_report_sha256": sha256_file(report_path),
        "freeze_runner_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator": "scripts/evaluate_ns_pimple_rotation_confirmation.py",
        "evaluator_sha256": sha256_file(ROOT / "scripts/evaluate_ns_pimple_rotation_confirmation.py"),
        "v0_manifest_sha256": sha256_file(v0_path),
        "v1_manifest_sha256": sha256_file(v1_path),
        "v2_validation_manifest_sha256": sha256_file(v2_path),
        "v2_generator_sha256": sha256_file(NS_ROOT / "scripts/generate_method_v2_confirmation.py"),
        "rotation_runner_sha256": sha256_file(ROOT / "scripts/run_ns_pimple_rotation_source_lr_diagnostic.py"),
        "method_runner_sha256": sha256_file(ROOT / "scripts/run_ns_pimple_method_v1.py"),
        "engine_sha256": sha256_file(ROOT / "blrpng/training/engine.py"),
        "model_base_config_sha256": sha256_file(ROOT / "configs/navier_stokes_sfepy_seed42.yaml"),
        "data_hashes": data_hashes,
        "target_validation_pair_hashes": validation_hashes,
        "fresh_hr_query_plan": [
            {key: row[key] for key in ("pair_id", "task_id", "split", "role", "level", "config", "config_sha256", "hdf5")}
            for row in query_plans
        ],
        "rotation_source_checkpoints": source_checkpoints,
        "rotation_target_checkpoints": target_checkpoints,
        "original_v1_target_checkpoints": original_checkpoints,
        "target_hr_query_used_for_checkpoint_selection": False,
        "target_hr_query_fields_read": False,
        "sealed_test_opened": False,
    }
    write_json(str(FREEZE_PATH.relative_to(ROOT)), freeze)
    print(json.dumps({"status": freeze["status"], "freeze": str(FREEZE_PATH.relative_to(ROOT)),
                      "fresh_hr_queries_planned": len(query_plans),
                      "target_hr_query_fields_read": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
