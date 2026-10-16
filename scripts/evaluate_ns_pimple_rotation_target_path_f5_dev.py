#!/usr/bin/env python3
"""One-pass F5 development evaluation of rotation-source target path fine-tuning."""

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


REPORT_PATH = method.REPORT_ROOT / "rotation_target_path_lr_diagnostic.json"
ACCESS_LOG_PATH = method.REPORT_ROOT / "rotation_target_path_f5_dev_access_log.json"
RESULT_PATH = method.REPORT_ROOT / "rotation_target_path_f5_dev_result.json"
TASKS = tuple(task for task in method.TASKS if task.startswith("F5_"))
ARMS = ("final_only", "selected_pimple_path", "matched_endpoint_control")


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
        raise FileExistsError("F5 development query pass has already started")
    report = method.load_json(REPORT_PATH)
    if report.get("status") != "lr_checkpoints_frozen_before_f5_hr_development_open":
        raise ValueError("target path checkpoints were not frozen before F5 HR development access")
    if report.get("f5_hr_development_opened") is not False:
        raise ValueError("target path report does not certify unopened F5 HR development data")
    prereg_path = ROOT / "docs/prereg/ns_pimple_method_v1_rotation_target_path_lr_diagnostic.json"
    if report.get("preregistration_sha256") != sha256_file(prereg_path):
        raise ValueError("target path preregistration changed after training")
    rotation_report = method.load_json(method.REPORT_ROOT / "rotation_source_lr_diagnostic.json")
    for task in TASKS:
        checkpoint = rotation_report["target_tasks"][task]["selected_pimple_path"]["checkpoint"]
        if report["baseline_final_only_from_rotation_run"][task]["checkpoint"] != checkpoint:
            raise ValueError(f"final-only baseline changed after freezing: {task}")
        for arm in ARMS[1:]:
            row = report["tasks"][task][arm]
            if sha256_file(ROOT / row["checkpoint"]) != row["checkpoint_sha256"]:
                raise ValueError(f"frozen target path checkpoint hash mismatch: {task}/{arm}")

    v2 = method.load_json(NS_ROOT / "data/method_v2/manifest.json")
    rows = {row["pair_id"]: row for row in v2["trajectories"]}
    queries = {task: [] for task in TASKS}
    for task in TASKS:
        for ic in ("query_mode15", "query_mode51"):
            pair_id = f"{task}_{ic}"
            row = rows.get(pair_id)
            if (row is None or row.get("role") != "query" or row.get("level") != "hr"
                    or row.get("status") != "complete" or row.get("audit", {}).get("status") != "pass"):
                raise ValueError(f"F5 development query is incomplete: {pair_id}")
            queries[task].append({
                "pair_id": pair_id,
                "task_id": row["task_id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "role": row["role"],
                "status": "complete",
                "training_access": "none",
                "cases": {"hr": {"hdf5": row["hdf5"], "status": "complete",
                                  "hdf5_sha256": row["hdf5_sha256"]}},
            })

    access_log = {
        "schema": "blrpng_ns_pimple_rotation_target_path_f5_dev_access_log_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_path_report_sha256": sha256_file(REPORT_PATH),
        "f5_query_pair_ids": {task: [row["pair_id"] for row in task_rows]
                              for task, task_rows in queries.items()},
        "f6_hr_query_fields_read": False,
        "purpose": "development comparison after LR-only checkpoint freeze; no F6 selection",
    }
    write_json(str(ACCESS_LOG_PATH.relative_to(ROOT)), access_log)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_base = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = method.build_model_config(model_base)
    dataset = method.NSPimpleDataset(NS_ROOT)
    prepared = {}
    for task, task_rows in queries.items():
        prepared[task] = []
        for row in task_rows:
            dataset.by_id[row["pair_id"]] = row
            case = method.prepare_case(dataset, row, "hr", training=False)
            prepared[task].append(case)

    result = {
        "schema": "blrpng_ns_pimple_rotation_target_path_f5_dev_result_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_path_report_sha256": sha256_file(REPORT_PATH),
        "access_log_sha256": sha256_file(ACCESS_LOG_PATH),
        "device": str(device),
        "primary_metric": "mean 20-step native cell-centre vector velocity relative-L2",
        "f6_hr_query_fields_read": False,
        "tasks": {},
    }
    for task in TASKS:
        result["tasks"][task] = {"arms": {}}
        checkpoint_rows = {
            "final_only": report["baseline_final_only_from_rotation_run"][task],
            "selected_pimple_path": report["tasks"][task]["selected_pimple_path"],
            "matched_endpoint_control": report["tasks"][task]["matched_endpoint_control"],
        }
        for arm, record in checkpoint_rows.items():
            path = ROOT / record["checkpoint"]
            expected = record["checkpoint_sha256"]
            if sha256_file(path) != expected:
                raise ValueError(f"checkpoint hash mismatch: {task}/{arm}")
            model = load_model(path, model_config, device)
            rows_out = [method.detailed_frozen_evaluation(model, case, device)
                        for case in prepared[task]]
            result["tasks"][task]["arms"][arm] = {
                "query_pair_ids": [row["pair_id"] for row in rows_out],
                "mean_relative_l2": float(np.mean([row["rollout_rel_l2_mean"] for row in rows_out])),
                "per_query": rows_out,
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": expected,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result["f5_mean"] = {
        arm: float(np.mean([result["tasks"][task]["arms"][arm]["mean_relative_l2"] for task in TASKS]))
        for arm in ARMS
    }
    result["relative_reduction_vs_final_only"] = {
        arm: (result["f5_mean"]["final_only"] - value) / result["f5_mean"]["final_only"]
        for arm, value in result["f5_mean"].items() if arm != "final_only"
    }
    path_mean = result["f5_mean"]["selected_pimple_path"]
    result["relative_reduction_vs_controls"] = {
        control: (result["f5_mean"][control] - path_mean) / result["f5_mean"][control]
        for control in ("final_only", "matched_endpoint_control")
    }
    result["no_task_worse_than_either_control"] = all(
        result["tasks"][task]["arms"]["selected_pimple_path"]["mean_relative_l2"]
        <= min(result["tasks"][task]["arms"][control]["mean_relative_l2"]
               for control in ("final_only", "matched_endpoint_control"))
        for task in TASKS
    )
    result["promote_to_fresh_f6_confirmation"] = (
        all(value >= 0.01 for value in result["relative_reduction_vs_controls"].values())
        and result["no_task_worse_than_either_control"]
    )
    write_json(str(RESULT_PATH.relative_to(ROOT)), result)
    print(json.dumps({
        "f5_mean": result["f5_mean"],
        "relative_reduction_vs_final_only": result["relative_reduction_vs_final_only"],
        "relative_reduction_vs_controls": result["relative_reduction_vs_controls"],
        "no_task_worse_than_either_control": result["no_task_worse_than_either_control"],
        "promote_to_fresh_f6_confirmation": result["promote_to_fresh_f6_confirmation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
