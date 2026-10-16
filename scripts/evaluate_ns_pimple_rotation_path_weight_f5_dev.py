#!/usr/bin/env python3
"""Evaluate frozen path-weight candidates on the existing F5 HR development set."""

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

REPORT_PATH = method.REPORT_ROOT / "rotation_path_weight_sensitivity.json"
ACCESS_LOG_PATH = method.REPORT_ROOT / "rotation_path_weight_sensitivity_f5_dev_access_log.json"
RESULT_PATH = method.REPORT_ROOT / "rotation_path_weight_sensitivity_f5_dev_result.json"
TASKS = tuple(task for task in method.TASKS if task.startswith("F5_"))
RATIOS = (0.20, 0.50)


def load_model(path: Path, model_config: dict, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    base = build_model(model_config, representation="lifted")
    if payload.get("adapter") == "nlt":
        base.configure_adaptation("nlt", rank=4, alpha=4.0)
    base.load_state_dict(payload["base_state_dict"], strict=True)
    model = method.TimeDependentLiftedOperator(base).to(device)
    model.eval()
    return model


def main() -> int:
    if ACCESS_LOG_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("weight-sensitivity F5 HR development pass has already started")
    report = method.load_json(REPORT_PATH)
    if report.get("status") != "lr_checkpoints_frozen_before_f5_hr_development_open":
        raise ValueError("path-weight checkpoints were not frozen before F5 HR access")
    if report.get("f5_hr_development_opened") is not False:
        raise ValueError("weight-sensitivity report indicates F5 HR was already opened")
    prereg_path = ROOT / "docs/prereg/ns_pimple_method_v1_rotation_path_weight_sensitivity.json"
    if report.get("preregistration_sha256") != sha256_file(prereg_path):
        raise ValueError("path-weight preregistration changed after training")

    rotation_report = method.load_json(method.REPORT_ROOT / "rotation_source_lr_diagnostic.json")
    query_manifest = method.load_json(NS_ROOT / "data/method_v2/manifest.json")
    manifest_rows = {row["pair_id"]: row for row in query_manifest["trajectories"]}
    queries = {task: [] for task in TASKS}
    for task in TASKS:
        for ic in ("query_mode15", "query_mode51"):
            pair_id = f"{task}_{ic}"
            row = manifest_rows.get(pair_id)
            if (row is None or row.get("role") != "query" or row.get("level") != "hr"
                    or row.get("status") != "complete" or row.get("audit", {}).get("status") != "pass"):
                raise ValueError(f"F5 development query metadata is incomplete: {pair_id}")
            queries[task].append({
                "pair_id": pair_id, "task_id": row["task_id"], "family_id": row["family_id"],
                "split": row["split"], "role": row["role"], "status": "complete",
                "training_access": "none",
                "cases": {"hr": {"hdf5": row["hdf5"], "status": "complete",
                                  "hdf5_sha256": row["hdf5_sha256"]}},
            })

    # All model checkpoints are verified before query fields are opened.
    final_rows = {}
    for task in TASKS:
        final_rows[task] = rotation_report["target_tasks"][task]["selected_pimple_path"]
        final_path = ROOT / final_rows[task]["checkpoint"]
        if sha256_file(final_path) != final_rows[task]["checkpoint_sha256"]:
            raise ValueError(f"final-only baseline hash mismatch: {task}")
        for ratio in RATIOS:
            key = f"r{int(round(ratio * 100)):03d}"
            for arm in ("selected_pimple_path", "matched_endpoint_control"):
                record = report["tasks"][task][key][arm]
                if sha256_file(ROOT / record["checkpoint"]) != record["checkpoint_sha256"]:
                    raise ValueError(f"path-weight checkpoint hash mismatch: {task}/{key}/{arm}")

    access_log = {
        "schema": "blrpng_ns_pimple_rotation_path_weight_f5_dev_access_log_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_sha256": sha256_file(REPORT_PATH),
        "f5_query_pair_ids": {task: [row["pair_id"] for row in values]
                              for task, values in queries.items()},
        "f6_hr_query_fields_read": False,
        "purpose": "development-only comparison after LR checkpoint freeze",
    }
    write_json(str(ACCESS_LOG_PATH.relative_to(ROOT)), access_log)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_base = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = method.build_model_config(model_base)
    dataset = method.NSPimpleDataset(NS_ROOT)
    prepared = {task: [] for task in TASKS}
    for task, rows in queries.items():
        for row in rows:
            dataset.by_id[row["pair_id"]] = row
            prepared[task].append(method.prepare_case(dataset, row, "hr", training=False))

    result = {
        "schema": "blrpng_ns_pimple_rotation_path_weight_f5_dev_result_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_sha256": sha256_file(REPORT_PATH),
        "access_log_sha256": sha256_file(ACCESS_LOG_PATH),
        "device": str(device),
        "primary_metric": "mean 20-step native cell-centre vector velocity relative-L2",
        "f6_hr_query_fields_read": False,
        "tasks": {},
    }
    for task in TASKS:
        result["tasks"][task] = {"arms": {}}
        candidates = {"final_only": final_rows[task]}
        for ratio in RATIOS:
            key = f"r{int(round(ratio * 100)):03d}"
            candidates[f"{key}_selected_pimple_path"] = report["tasks"][task][key]["selected_pimple_path"]
            candidates[f"{key}_matched_endpoint_control"] = report["tasks"][task][key]["matched_endpoint_control"]
        for arm, record in candidates.items():
            model = load_model(ROOT / record["checkpoint"], model_config, device)
            evaluated = [method.detailed_frozen_evaluation(model, case, device) for case in prepared[task]]
            result["tasks"][task]["arms"][arm] = {
                "query_pair_ids": [row["pair_id"] for row in evaluated],
                "mean_relative_l2": float(np.mean([row["rollout_rel_l2_mean"] for row in evaluated])),
                "per_query": evaluated,
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result["mean_by_arm"] = {
        arm: float(np.mean([result["tasks"][task]["arms"][arm]["mean_relative_l2"] for task in TASKS]))
        for arm in next(iter(result["tasks"].values()))["arms"]
    }
    result["gate_by_ratio"] = {}
    for ratio in RATIOS:
        key = f"r{int(round(ratio * 100)):03d}"
        path_arm, matched_arm = f"{key}_selected_pimple_path", f"{key}_matched_endpoint_control"
        path_mean = result["mean_by_arm"][path_arm]
        reduction = {
            "final_only": (result["mean_by_arm"]["final_only"] - path_mean)
            / result["mean_by_arm"]["final_only"],
            "matched_endpoint_control": (result["mean_by_arm"][matched_arm] - path_mean)
            / result["mean_by_arm"][matched_arm],
        }
        no_task_worse = all(
            result["tasks"][task]["arms"][path_arm]["mean_relative_l2"]
            <= min(result["tasks"][task]["arms"]["final_only"]["mean_relative_l2"],
                   result["tasks"][task]["arms"][matched_arm]["mean_relative_l2"])
            for task in TASKS
        )
        result["gate_by_ratio"][key] = {
            "relative_reduction_vs_controls": reduction,
            "no_task_worse_than_either_control": no_task_worse,
            "promote_to_fresh_f6_confirmation": all(value >= 0.01 for value in reduction.values())
            and no_task_worse,
        }
    result["promoted_ratios"] = [key for key, value in result["gate_by_ratio"].items()
                                  if value["promote_to_fresh_f6_confirmation"]]
    write_json(str(RESULT_PATH.relative_to(ROOT)), result)
    print(json.dumps({"mean_by_arm": result["mean_by_arm"],
                      "gate_by_ratio": result["gate_by_ratio"],
                      "promoted_ratios": result["promoted_ratios"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
