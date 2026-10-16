from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.data.shards import read_index
from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json, write_json


def _manifest_hash(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    digest = sha256_json(unsigned)
    if payload.get("sha256") != digest:
        raise ValueError("R1 task manifest hash does not match its contents.")
    return digest


def audit(config_path: Path) -> dict:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = spec["experiment"]
    manifest_path = PROJECT_ROOT / experiment["data_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _manifest_hash(manifest)
    tasks = {int(row["target_task_id"]): row for row in manifest["tasks"]}
    selected = [int(task) for task in experiment["tasks"]]
    if selected != sorted(selected) or not selected:
        raise ValueError("R1 tasks must be a nonempty sorted list.")
    if not set(selected).issubset(tasks):
        raise ValueError("R1 config refers to task IDs absent from the frozen manifest.")
    families = [str(tasks[task]["family"]) for task in selected]
    if len(set(families)) != len(families):
        raise ValueError("R1 screen must cover distinct boundary families.")

    root = str(experiment["data_root"])
    budget = int(experiment["support_budget"])
    support_seed = int(experiment.get("support_seed", 0))
    support_template = str(experiment["support_index_template"])
    validation_template = str(experiment["validation_index_template"])
    support_rows = []
    validation_rows = []
    task_audits = {}
    for task in selected:
        support_path = PROJECT_ROOT / support_template.format(
            root=root, task=task, budget=budget, support_seed=support_seed
        )
        validation_path = PROJECT_ROOT / validation_template.format(
            root=root, task=task, budget=budget, support_seed=support_seed
        )
        support = read_index(support_path, verify_hashes=True)
        validation = read_index(validation_path, verify_hashes=True)
        if int(support["metadata"].get("support_seed", -1)) != support_seed:
            raise ValueError(f"support task {task}: support seed differs from R1 config")
        for label, index, role, resolution in (
            ("support", support, "support", 64),
            ("validation", validation, "validation_lr", 64),
        ):
            metadata = index["metadata"]
            if metadata.get("split") != role:
                raise ValueError(f"{label} task {task}: unexpected split role")
            if int(metadata.get("steps", -1)) != 20:
                raise ValueError(f"{label} task {task}: expected 20-step rollouts")
            if int(metadata.get("resolution", -1)) != resolution:
                raise ValueError(f"{label} task {task}: expected {resolution}x{resolution}")
            if int(metadata.get("fidelity", -1)) != 0:
                raise ValueError(f"{label} task {task}: expected genuine LR solver fidelity 0")
            if metadata.get("task_manifest_sha256") != manifest_hash:
                raise ValueError(f"{label} task {task}: task manifest hash mismatch")
        sr = [row for row in support["shards"] if int(row["target_task_id"]) == task]
        vr = [row for row in validation["shards"] if int(row["target_task_id"]) == task]
        if len(sr) != budget or len(vr) != 8:
            raise ValueError(f"task {task}: expected support m={budget} and 8 LR validation rollouts")
        boundary = int(tasks[task]["boundary_id"])
        if any(int(row["boundary_id"]) != boundary for row in sr + vr):
            raise ValueError(f"task {task}: boundary mapping differs from frozen task manifest")
        support_ics = {int(row["initial_condition_id"]) for row in sr}
        validation_ics = {int(row["initial_condition_id"]) for row in vr}
        if support_ics & validation_ics:
            raise ValueError(f"task {task}: support and LR validation initial conditions overlap")
        support_rows.extend(sr)
        validation_rows.extend(vr)
        task_audits[str(task)] = {
            "family": families[selected.index(task)],
            "boundary_id": boundary,
            "support_count": len(sr),
            "validation_lr_count": len(vr),
            "steps": 20,
            "support_index": str(support_path.relative_to(PROJECT_ROOT)),
            "support_index_sha256": support["sha256"],
            "validation_lr_index": str(validation_path.relative_to(PROJECT_ROOT)),
            "validation_lr_index_sha256": validation["sha256"],
            "support_validation_initial_condition_overlap": 0,
        }
    support_tasks = Counter(int(row["target_task_id"]) for row in support_rows)
    validation_tasks = Counter(int(row["target_task_id"]) for row in validation_rows)
    screen_seed = int(experiment["screen_model_seed"])
    source_checkpoints = {}
    for trial in spec["trials"]:
        trial_id = str(trial["id"])
        template = trial.get("initialization_checkpoint_template")
        checkpoint = (
            str(template).format(seed=screen_seed)
            if template is not None
            else trial.get("initialization_checkpoint", experiment.get("checkpoint"))
        )
        if checkpoint is None:
            source_checkpoints[trial_id] = None
            continue
        checkpoint_path = PROJECT_ROOT / str(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{trial_id}: source checkpoint is missing: {checkpoint_path}")
        source_checkpoints[trial_id] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint_path),
        }
    return {
        "schema": "blrpng_phase1_r1_input_audit_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "task_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "task_manifest_sha256": manifest_hash,
        "tasks": selected,
        "boundary_families": families,
        "support_budget": budget,
        "support_seed": support_seed,
        "horizon": {"steps": 20, "dt": 0.2, "physical_time": 4.0},
        "resolution": 64,
        "support_fidelity": 0,
        "selection_role": "LR validation only",
        "query_read": False,
        "source_checkpoints": source_checkpoints,
        "all_task_support_counts_match": all(support_tasks[task] == budget for task in selected),
        "all_task_validation_counts_match": all(validation_tasks[task] == 8 for task in selected),
        "per_task": task_audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen R1 LR-support and validation inputs.")
    parser.add_argument("--config", default="configs/phase1_r1.yaml")
    parser.add_argument("--output", default="artifacts/reports/phase1_r1_input_audit.json")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    result = audit(config_path)
    output = PROJECT_ROOT / args.output
    write_json(output, result)
    print(json.dumps({
        "output": str(output),
        "config_sha256": result["config_sha256"],
        "tasks": result["tasks"],
        "boundary_families": result["boundary_families"],
        "horizon": result["horizon"],
        "query_read": result["query_read"],
    }, indent=2))


if __name__ == "__main__":
    main()
