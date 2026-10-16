from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.data.shards import read_index
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    sha256_json,
)


def _read_sealed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = sha256_json({key: value for key, value in payload.items() if key != "sha256"})
    if payload.get("sha256") != actual:
        raise ValueError(f"Invalid self-seal: {path}")
    return payload


def _verify_lr_index(
    path: str,
    *,
    role: str,
    task: int,
    boundary_id: int,
    task_manifest_sha256: str,
    support_seed: int | None,
    expected_count: int | None,
) -> tuple[dict, set[int]]:
    index = read_index(PROJECT_ROOT / path, verify_hashes=True)
    metadata = index["metadata"]
    if metadata.get("split") != role:
        raise ValueError(f"{path} is {metadata.get('split')!r}, expected {role!r}.")
    if int(metadata.get("resolution", -1)) != 64 or int(metadata.get("fidelity", -1)) != 0:
        raise ValueError(f"{path} is not genuine 64x64 LR data.")
    if int(metadata.get("steps", -1)) != 20:
        raise ValueError(f"{path} does not use the frozen 20-step LR protocol.")
    if metadata.get("task_manifest_sha256") != task_manifest_sha256:
        raise ValueError(f"{path} is bound to a different task manifest.")
    rows = index["shards"]
    if not rows or len(rows) != len([row for row in rows if int(row["target_task_id"]) == task]):
        raise ValueError(f"{path} must contain only task {task}.")
    if any(
        int(row["boundary_id"]) != boundary_id
        or str(row.get("role")) != role
        or int(row.get("resolution", -1)) != 64
        or int(row.get("steps", -1)) != 20
        for row in rows
    ):
        raise ValueError(f"{path} shard rows disagree with the frozen task protocol.")
    initial_conditions = [int(row["initial_condition_id"]) for row in rows]
    if len(initial_conditions) != len(set(initial_conditions)):
        raise ValueError(f"{path} contains repeated initial-condition IDs within its {role} split.")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{path} contains {len(rows)} rollouts, expected {expected_count}.")
    if role == "support":
        if int(metadata.get("support_seed", -1)) != support_seed:
            raise ValueError(f"{path} uses a different support seed.")
    elif metadata.get("support_seed") is not None:
        raise ValueError(f"{path} LR validation index is not independent of support draws.")
    return index, {int(row["initial_condition_id"]) for row in rows}


def freeze_checkpoints(campaign: str) -> Path:
    root = PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign
    task_manifest = _read_sealed(root / "tasks.json")
    matrix = _read_sealed(root / "matrix.json")
    if task_manifest.get("schema") != "blrpng_phase1_confirmation_tasks_v1":
        raise ValueError("Unexpected confirmation task manifest schema.")
    if matrix.get("schema") != "blrpng_phase1_confirmation_matrix_v1":
        raise ValueError("Unexpected confirmation matrix schema.")
    if task_manifest.get("campaign_id") != campaign or matrix.get("campaign_id") != campaign:
        raise ValueError("Campaign ID mismatch in frozen manifests.")
    if matrix.get("task_manifest_sha256") != task_manifest.get("sha256"):
        raise ValueError("Confirmation matrix is not bound to its task manifest.")
    matrix_path = root / "matrix.json"
    matrix_path_sha = sha256_file(matrix_path)
    methods = matrix["methods"]
    method_configs = matrix["method_configs"]
    tasks = {int(row["target_task_id"]): row for row in task_manifest["tasks"]}
    task_ids = sorted(int(task) for task in matrix["task_ids"])
    if task_ids != sorted(tasks):
        raise ValueError("Confirmation matrix does not cover every frozen task.")

    ledger_rows = list(csv.DictReader((PROJECT_ROOT / "docs/EXPERIMENTS.csv").open(newline="", encoding="utf-8")))
    if any(row.get("round") == "CONFIRM" and row.get("status") == "RUNNING" for row in ledger_rows):
        raise ValueError("Cannot freeze confirmation checkpoints while any confirmation trial is running.")
    ledger = {}
    for row in ledger_rows:
        if row.get("round") == "CONFIRM":
            ledger.setdefault(row.get("trial_id"), []).append(row)

    checkpoint_rows = []
    checkpoint_paths = set()
    task_result_paths = set()
    for method in methods:
        config_record = method_configs[method]
        config_path = resolve_in_project(config_record["config_path"], must_exist=True)
        if sha256_file(config_path) != config_record["config_sha256"]:
            raise ValueError(f"Confirmation config changed after matrix freeze: {method}")
        spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment = spec["experiment"]
        if str(experiment.get("round")) != "CONFIRM":
            raise ValueError("Confirmation target configs must use round CONFIRM.")
        selected_trial = next((row for row in spec["trials"] if str(row["id"]) == method), None)
        if selected_trial is None:
            raise ValueError(f"Method {method!r} is not present in its frozen config.")
        if experiment.get("data_manifest") != str((root / "tasks.json").relative_to(PROJECT_ROOT)):
            raise ValueError(f"Confirmation config {method} does not use this campaign task manifest.")

        for model_seed in matrix["model_seeds"]:
            for support_seed in matrix["support_seeds"]:
                run_namespace = f"{method}_m{int(matrix['support_budget'])}_s{int(model_seed)}"
                if int(support_seed):
                    run_namespace += f"_ss{int(support_seed)}"
                run_dir = PROJECT_ROOT / "results/phase1/CONFIRM" / run_namespace
                run_config_path = run_dir / "config.json"
                run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                descriptor = run_config["descriptor"]
                if len(str(run_config.get("config_sha256", ""))) != 64:
                    raise ValueError(f"Run config digest is malformed for {run_namespace}.")
                if descriptor.get("round") != "CONFIRM":
                    raise ValueError(f"Run directory is not in the confirmation round: {run_namespace}")
                if str(descriptor.get("trial", {}).get("id")) != method:
                    raise ValueError(f"Selected trial ID mismatch in {run_namespace}")
                if descriptor.get("model_seed") != int(model_seed):
                    raise ValueError(f"Model seed mismatch in {run_namespace}")
                if descriptor.get("support_seed") != int(support_seed):
                    raise ValueError(f"Support seed mismatch in {run_namespace}")
                if descriptor.get("support_budget") != int(matrix["support_budget"]):
                    raise ValueError(f"Support budget mismatch in {run_namespace}")
                if descriptor.get("trial_spec_sha256") != config_record["config_sha256"]:
                    raise ValueError(f"Trial config hash mismatch in {run_namespace}")
                if descriptor.get("trial") != selected_trial:
                    raise ValueError(f"Selected trial specification mismatch in {run_namespace}")
                if run_config.get("config_sha256") != sha256_json(descriptor):
                    raise ValueError(f"Run config digest does not seal its descriptor: {run_namespace}")
                if descriptor.get("data_manifest_sha256") != task_manifest["sha256"]:
                    raise ValueError(f"Task manifest hash mismatch in {run_namespace}")
                source_checkpoint = descriptor.get("source_checkpoint")
                source_checkpoint_sha256 = descriptor.get("source_checkpoint_sha256")
                if source_checkpoint is None:
                    if source_checkpoint_sha256 is not None:
                        raise ValueError(f"Unexpected source checkpoint hash in {run_namespace}")
                else:
                    source_path = resolve_in_project(str(source_checkpoint), must_exist=True)
                    if not source_path.is_file() or sha256_file(source_path) != source_checkpoint_sha256:
                        raise ValueError(f"Source checkpoint hash mismatch in {run_namespace}")
                for task_id in task_ids:
                    task = tasks[task_id]
                    result_path = run_dir / f"task{task_id:02d}.json"
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    result_rel = str(result_path.relative_to(PROJECT_ROOT))
                    if result_rel in task_result_paths:
                        raise ValueError(f"Repeated task result path: {result_rel}")
                    task_result_paths.add(result_rel)
                    if int(result.get("task", -1)) != task_id:
                        raise ValueError(f"Task ID mismatch in {result_rel}")
                    if str(result.get("trial_id")) != method:
                        raise ValueError(f"Method ID mismatch in {result_rel}")
                    if int(result.get("model_seed", -1)) != int(model_seed):
                        raise ValueError(f"Model seed mismatch in {result_rel}")
                    if int(result.get("support_seed", -1)) != int(support_seed):
                        raise ValueError(f"Support seed mismatch in {result_rel}")
                    if int(result.get("support_budget", -1)) != int(matrix["support_budget"]):
                        raise ValueError(f"Support budget mismatch in {result_rel}")
                    if result.get("query_metric") is not None:
                        raise ValueError(f"Query metric was present during LR checkpoint selection: {result_rel}")
                    if result.get("config_sha256") != run_config.get("config_sha256"):
                        raise ValueError(f"Task result is bound to a different run config: {result_rel}")
                    if result.get("descriptor_sha256") != run_config.get("config_sha256"):
                        raise ValueError(f"Task result descriptor hash mismatch: {result_rel}")
                    if result.get("source_checkpoint") != source_checkpoint:
                        raise ValueError(f"Source checkpoint path mismatch in {result_rel}")
                    if result.get("source_checkpoint_sha256") != source_checkpoint_sha256:
                        raise ValueError(f"Source checkpoint hash binding mismatch in {result_rel}")
                    if result.get("family") != task["family"]:
                        raise ValueError(f"Family mismatch in {result_rel}")
                    if "LR validation" not in str(result.get("selection_data", "")):
                        raise ValueError(f"Checkpoint selection was not declared LR-only: {result_rel}")

                    history = result.get("history", [])
                    values = [float(row["validation"]["rollout_rel_l2"]) for row in history]
                    if not values or not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Invalid LR validation history in {result_rel}")
                    minimum = min(values)
                    minima = [index for index, value in enumerate(values) if value == minimum]
                    if len(minima) != 1 or not math.isclose(
                        float(result["best_validation_metric"]), minimum, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(f"LR validation history lacks a unique recorded minimum: {result_rel}")
                    if int(result.get("best_epoch", -1)) != int(history[minima[0]]["epoch"]):
                        raise ValueError(f"Recorded best epoch is not the unique LR minimum: {result_rel}")

                    expected_ledger_id = (
                        f"CONFIRM-{method}-task{task_id:02d}-seed{model_seed}-"
                        f"m{matrix['support_budget']}-support{support_seed}"
                    )
                    ledger_matches = ledger.get(expected_ledger_id, [])
                    if len(ledger_matches) != 1 or ledger_matches[0].get("status") != "DONE":
                        raise ValueError(f"Expected exactly one DONE ledger row: {expected_ledger_id}")
                    ledger_result_path = Path(str(ledger_matches[0].get("raw_output_path", "")))
                    if ledger_result_path.is_absolute():
                        ledger_result_path = ledger_result_path.resolve().relative_to(PROJECT_ROOT)
                    if str(ledger_result_path) != result_rel:
                        raise ValueError(f"Ledger/result path mismatch: {expected_ledger_id}")

                    support_index, support_ids = _verify_lr_index(
                        str(result["support_index"]), role="support", task=task_id,
                        boundary_id=int(task["boundary_id"]),
                        task_manifest_sha256=task_manifest["sha256"],
                        support_seed=int(support_seed), expected_count=int(matrix["support_budget"]),
                    )
                    validation_index, validation_ids = _verify_lr_index(
                        str(result["validation_index"]), role="validation_lr", task=task_id,
                        boundary_id=int(task["boundary_id"]),
                        task_manifest_sha256=task_manifest["sha256"],
                        support_seed=None, expected_count=int(matrix["lr_validation_rollouts"]),
                    )
                    if support_ids & validation_ids:
                        raise ValueError(f"Support and LR validation ICs overlap: {result_rel}")
                    if result.get("support_index_sha256") != support_index["sha256"]:
                        raise ValueError(f"Support index hash mismatch in {result_rel}")
                    if result.get("validation_index_sha256") != validation_index["sha256"]:
                        raise ValueError(f"LR validation index hash mismatch in {result_rel}")

                    checkpoint_rel = str(result["checkpoint"])
                    checkpoint_path = PROJECT_ROOT / checkpoint_rel
                    checkpoint_sha = sha256_file(checkpoint_path)
                    if checkpoint_sha != result.get("checkpoint_sha256"):
                        raise ValueError(f"Checkpoint hash mismatch in {result_rel}")
                    if checkpoint_rel in checkpoint_paths:
                        raise ValueError(f"Different confirmation cells reuse one checkpoint: {checkpoint_rel}")
                    checkpoint_paths.add(checkpoint_rel)
                    checkpoint_rows.append({
                        "method": method,
                        "model_seed": int(model_seed),
                        "support_seed": int(support_seed),
                        "target_task_id": task_id,
                        "family": str(task["family"]),
                        "run_config_path": str(run_config_path.relative_to(PROJECT_ROOT)),
                        "run_config_sha256": sha256_file(run_config_path),
                        "trial_config_path": config_record["config_path"],
                        "trial_config_sha256": config_record["config_sha256"],
                        "task_result_path": result_rel,
                        "task_result_sha256": sha256_file(result_path),
                        "config_sha256": str(result["config_sha256"]),
                        "best_validation_metric": float(result["best_validation_metric"]),
                        "best_epoch": int(result["best_epoch"]),
                        "support_index": str(result["support_index"]),
                        "support_index_sha256": str(result["support_index_sha256"]),
                        "validation_index": str(result["validation_index"]),
                        "validation_index_sha256": str(result["validation_index_sha256"]),
                        "checkpoint_path": checkpoint_rel,
                        "checkpoint_sha256": checkpoint_sha,
                    })

    payload = {
        "schema": "blrpng_phase1_confirmation_checkpoints_v1",
        "campaign_id": campaign,
        "task_manifest_sha256": task_manifest["sha256"],
        "matrix_sha256": matrix["sha256"],
        "matrix_path_sha256": matrix_path_sha,
        "experiment_ledger_sha256": sha256_file(PROJECT_ROOT / "docs/EXPERIMENTS.csv"),
        "checkpoints": checkpoint_rows,
        "checkpoint_count": len(checkpoint_rows),
        "query_opened": False,
        "lr_selection_only": True,
        "unique_checkpoint_paths": len(checkpoint_paths),
    }
    payload["sha256"] = sha256_json(payload)
    output = ensure_artifact_path(root / "checkpoints.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"frozen confirmation checkpoints: {output.relative_to(PROJECT_ROOT)}")
    print(f"checkpoint matrix cells: {len(checkpoint_rows)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and verify all LR-selected confirmation checkpoints before query access.")
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    freeze_checkpoints(args.campaign)


if __name__ == "__main__":
    main()
