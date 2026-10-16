#!/usr/bin/env python3
"""One-pass evaluation on fresh, frozen NS-PIMPLE V2 HR queries."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = ROOT.parent / "NS-PIMPLE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NS_ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ns_pimple_method_v1 as method  # noqa: E402
from blrpng.models.fno import build_model  # noqa: E402
from blrpng.paths import sha256_file, write_json  # noqa: E402


FREEZE_PATH = method.REPORT_ROOT / "rotation_confirmation_freeze.json"
ACCESS_LOG_PATH = method.REPORT_ROOT / "rotation_confirmation_query_access_log.json"
RESULT_PATH = method.REPORT_ROOT / "rotation_confirmation_query_result.json"
EVALUATOR_AMENDMENT_PATH = method.REPORT_ROOT / "rotation_confirmation_evaluator_amendment.json"
ROTATION_ARMS = (*method.SOURCE_ARMS, "lifted_lr_scratch")
ARM_PREFIXES = {
    **{arm: "rotation_" for arm in ROTATION_ARMS},
}


def manifest_row_map(manifest: dict) -> dict[str, dict]:
    return {row["pair_id"]: row for row in manifest.get("trajectories", [])}


def verify_before_query(freeze: dict, v2: dict) -> dict[str, list[dict]]:
    if freeze.get("status") != "frozen_before_fresh_hr_query_generation_and_field_access":
        raise ValueError("confirmation freeze is not in the expected pre-query state")
    if freeze.get("target_hr_query_fields_read") is not False:
        raise ValueError("freeze records that target HR fields were already accessed")
    if freeze.get("freeze_runner_sha256") != sha256_file(ROOT / "scripts/freeze_ns_pimple_rotation_confirmation.py"):
        raise ValueError("freeze runner changed after checkpoint freeze")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if freeze.get("evaluator_sha256") != evaluator_sha:
        if not EVALUATOR_AMENDMENT_PATH.is_file():
            raise ValueError("query evaluator changed after checkpoint freeze without a recorded amendment")
        amendment = method.load_json(EVALUATOR_AMENDMENT_PATH)
        if (amendment.get("freeze_sha256") != sha256_file(FREEZE_PATH)
                or amendment.get("previous_evaluator_sha256") != freeze.get("evaluator_sha256")
                or amendment.get("replacement_evaluator_sha256") != evaluator_sha
                or amendment.get("target_hr_query_fields_read_before_amendment") is not False
                or amendment.get("status") != "pre_access_evaluator_bug_fix"):
            raise ValueError("evaluator amendment does not match the frozen checkpoint and replacement evaluator")
    prereg = ROOT / "docs/prereg/ns_pimple_method_v1_rotation_source_lr_diagnostic.json"
    if freeze.get("preregistration_sha256") != sha256_file(prereg):
        raise ValueError("rotation preregistration changed after freeze")
    if freeze.get("v0_manifest_sha256") != sha256_file(NS_ROOT / "data/v0/manifest.json"):
        raise ValueError("V0 data manifest changed after freeze")
    if freeze.get("v1_manifest_sha256") != sha256_file(NS_ROOT / "data/method_v1/manifest.json"):
        raise ValueError("V1 data manifest changed after freeze")
    if freeze.get("v2_generator_sha256") != sha256_file(NS_ROOT / "scripts/generate_method_v2_confirmation.py"):
        raise ValueError("V2 generator changed after freeze")
    if freeze.get("rotation_runner_sha256") != sha256_file(ROOT / "scripts/run_ns_pimple_rotation_source_lr_diagnostic.py"):
        raise ValueError("rotation training runner changed after freeze")
    if freeze.get("method_runner_sha256") != sha256_file(ROOT / "scripts/run_ns_pimple_method_v1.py"):
        raise ValueError("method data/evaluation bridge changed after freeze")
    if freeze.get("engine_sha256") != sha256_file(ROOT / "blrpng/training/engine.py"):
        raise ValueError("training engine changed after freeze")
    if freeze.get("model_base_config_sha256") != sha256_file(ROOT / "configs/navier_stokes_sfepy_seed42.yaml"):
        raise ValueError("model base configuration changed after freeze")
    report_path = method.REPORT_ROOT / "rotation_source_lr_diagnostic.json"
    if freeze.get("lr_report_sha256") != sha256_file(report_path):
        raise ValueError("LR source/target report changed after freeze")
    report = method.load_json(report_path)
    if report.get("target_hr_query_used") is not False:
        raise ValueError("rotation LR report indicates HR query access")
    for key, path in (
        ("source_train", method.DATA_ROOT / "source_path_pretraining.pt"),
        ("source_validation_rollouts", method.DATA_ROOT / "source_validation_rollouts.pt"),
        ("dataset_index", method.DATA_ROOT / "dataset_index.json"),
    ):
        if freeze["data_hashes"].get(key) != sha256_file(path):
            raise ValueError(f"prepared source data changed after freeze: {key}")
    for task, digest in freeze["data_hashes"]["target_support"].items():
        if digest != sha256_file(method.DATA_ROOT / f"{task}_support_final_only.pt"):
            raise ValueError(f"target LR support changed after freeze: {task}")
    for arm, row in freeze["rotation_source_checkpoints"].items():
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"rotation source checkpoint changed after freeze: {arm}")
    for task, arms in freeze["rotation_target_checkpoints"].items():
        for arm, row in arms.items():
            if sha256_file(ROOT / row["path"]) != row["sha256"]:
                raise ValueError(f"rotation target checkpoint changed after freeze: {task}/{arm}")
    for task, arms in freeze["original_v1_target_checkpoints"].items():
        for arm, row in arms.items():
            if sha256_file(ROOT / row["path"]) != row["sha256"]:
                raise ValueError(f"original V1 target checkpoint changed after freeze: {task}/{arm}")

    v1 = method.load_json(NS_ROOT / "data/method_v1/manifest.json")
    v1_rows = manifest_row_map(v1)
    for pair_id, expected in freeze["target_validation_pair_hashes"].items():
        row = v1_rows.get(pair_id)
        if row is None:
            row = manifest_row_map(v2).get(pair_id)
        if row is None or row.get("config_sha256") != expected["config_sha256"] or row.get("hdf5_sha256") != expected["hdf5_sha256"]:
            raise ValueError(f"frozen LR validation row changed: {pair_id}")

    v2_rows = manifest_row_map(v2)
    planned = {row["pair_id"]: row for row in freeze["fresh_hr_query_plan"]}
    if set(planned) != {f"{task}_{ic}" for task in method.TASKS for ic in ("query_mode15", "query_mode51")}:
        raise ValueError("frozen HR query plan has unexpected ICs")
    by_task: dict[str, list[dict]] = {task: [] for task in method.TASKS}
    for pair_id, plan in planned.items():
        row = v2_rows.get(pair_id)
        if (row is None or row.get("status") != "complete" or row.get("role") != "query"
                or row.get("level") != "hr" or row.get("config_sha256") != plan["config_sha256"]
                or row.get("config") != plan["config"] or row.get("hdf5") != plan["hdf5"]
                or row.get("audit", {}).get("status") != "pass"):
            raise ValueError(f"fresh HR query does not match the frozen plan: {pair_id}")
        task = next(name for name in method.TASKS if pair_id.startswith(name + "_"))
        by_task[task].append({
            "pair_id": pair_id, "task_id": row["task_id"], "family_id": row["family_id"],
            "split": row["split"], "role": row["role"], "status": "complete",
            "training_access": "none",
            "cases": {"hr": {"hdf5": row["hdf5"], "status": "complete",
                              "hdf5_sha256": row["hdf5_sha256"]}},
        })
    if any(len(rows) != 2 for rows in by_task.values()):
        raise ValueError("fresh HR query count is not two per task")
    return by_task


def load_model(checkpoint_path: Path, model_config: dict, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = build_model(model_config, representation="lifted")
    if payload.get("adapter") == "nlt":
        base.configure_adaptation("nlt", rank=4, alpha=4.0)
    base.load_state_dict(payload["base_state_dict"], strict=True)
    model = method.TimeDependentLiftedOperator(base).to(device)
    model.eval()
    return model


def main() -> int:
    if ACCESS_LOG_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("fresh HR query evaluation has already started; refusing a second pass")
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("global rotation confirmation freeze is missing")
    freeze = method.load_json(FREEZE_PATH)
    v2_path = NS_ROOT / "data/method_v2/manifest.json"
    v2 = method.load_json(v2_path)
    query_rows = verify_before_query(freeze, v2)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device is unavailable: {device}")
    model_base = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = method.build_model_config(model_base)
    dataset = method.NSPimpleDataset(NS_ROOT)
    generated = {}
    for rows in query_rows.values():
        for row in rows:
            dataset.by_id[row["pair_id"]] = row
            generated[row["pair_id"]] = row

    access_log = {
        "schema": "blrpng_ns_pimple_rotation_confirmation_query_access_log_v1",
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fresh_query_pair_ids": {task: [row["pair_id"] for row in rows] for task, rows in query_rows.items()},
        "sealed_test_opened": True,
        "purpose": "one frozen evaluation pass; no checkpoint or model selection",
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_amendment_sha256": (sha256_file(EVALUATOR_AMENDMENT_PATH)
                                       if EVALUATOR_AMENDMENT_PATH.is_file() else None),
        "pre_evaluation_query_checksum_audit": (
            method.load_json(EVALUATOR_AMENDMENT_PATH).get("pre_evaluation_query_checksum_audit")
            if EVALUATOR_AMENDMENT_PATH.is_file() else None
        ),
    }
    write_json(str(ACCESS_LOG_PATH.relative_to(ROOT)), access_log)

    # After the durable access log, hash the sealed HDF5s and load query fields.
    for task, rows in query_rows.items():
        for row in rows:
            case_path = NS_ROOT / row["cases"]["hr"]["hdf5"]
            actual = sha256_file(case_path)
            expected = generated[row["pair_id"]]["cases"]["hr"]["hdf5_sha256"]
            if actual != expected:
                raise ValueError(f"fresh HR query checksum mismatch: {row['pair_id']}")
    prepared = {
        task: [method.prepare_case(dataset, row, "hr", training=False) for row in rows]
        for task, rows in query_rows.items()
    }
    result = {
        "schema": "blrpng_ns_pimple_rotation_confirmation_query_result_v1",
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "query_access_log_sha256": sha256_file(ACCESS_LOG_PATH),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_amendment_sha256": (sha256_file(EVALUATOR_AMENDMENT_PATH)
                                       if EVALUATOR_AMENDMENT_PATH.is_file() else None),
        "pre_evaluation_query_checksum_audit": access_log["pre_evaluation_query_checksum_audit"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device), "primary_metric": "mean 20-step native cell-centre vector velocity relative-L2",
        "query_pair_ids": {task: [case["pair_id"] for case in cases] for task, cases in prepared.items()},
        "tasks": {},
    }
    rotation_report = method.load_json(method.REPORT_ROOT / "rotation_source_lr_diagnostic.json")
    original_report = method.load_json(method.REPORT_ROOT / "target_training.json")
    arm_sources = {
        **{f"rotation_{arm}": "rotation" for arm in ROTATION_ARMS},
        **{f"original_v1_{arm}": "original_v1" for arm in ROTATION_ARMS},
    }
    for task in method.TASKS:
        result["tasks"][task] = {"split": "development" if task.startswith("F5_") else "confirmation", "arms": {}}
        for arm_name, source_kind in arm_sources.items():
            arm = arm_name.removeprefix("rotation_").removeprefix("original_v1_")
            if source_kind == "rotation":
                relative = rotation_report["target_tasks"][task][arm]["checkpoint"]
                checkpoint_path = ROOT / relative
            else:
                relative = original_report["tasks"][task][arm]["checkpoint"]
                checkpoint_path = ROOT / relative
            model = load_model(checkpoint_path, model_config, device)
            rows = [method.detailed_frozen_evaluation(model, case, device) for case in prepared[task]]
            result["tasks"][task]["arms"][arm_name] = {
                "query_pair_ids": [row["pair_id"] for row in rows],
                "mean_relative_l2": float(np.mean([row["rollout_rel_l2_mean"] for row in rows])),
                "final_step_relative_l2_mean": float(np.mean([row["rollout_rel_l2_final"] for row in rows])),
                "per_query": rows,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result["split_mean"] = {}
    for prefix, split in (("F5_", "development"), ("F6_", "confirmation")):
        tasks = [task for task in method.TASKS if task.startswith(prefix)]
        result["split_mean"][split] = {}
        for arm_name in arm_sources:
            task_values = [result["tasks"][task]["arms"][arm_name]["mean_relative_l2"] for task in tasks]
            result["split_mean"][split][arm_name] = float(np.mean(task_values))
    result["interpretation_rule"] = (
        "PIMPLE-guided validity requires rotation_selected_pimple_path to beat both "
        "rotation_final_only and rotation_matched_endpoint_control on development and confirmation means."
    )
    write_json(str(RESULT_PATH.relative_to(ROOT)), result)
    print(json.dumps({"status": "fresh_hr_confirmation_complete", "split_mean": result["split_mean"],
                      "query_access_log": str(ACCESS_LOG_PATH.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
