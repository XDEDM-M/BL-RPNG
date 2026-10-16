from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys_path_root = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(sys_path_root))

from blrpng.data.shards import read_index
from blrpng.paths import PROJECT_ROOT, sha256_json


def _sealed_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    actual = sha256_json(unsigned)
    if payload.get("sha256") != actual:
        raise ValueError("Frozen task manifest seal does not match its contents.")
    return actual


def _index(
    path: Path, *, role: str, task_hash: str, steps: int, resolution: int, fidelity: int
) -> dict[str, Any]:
    result = read_index(path, verify_hashes=True)
    metadata = result["metadata"]
    if metadata.get("split") != role:
        raise ValueError(f"{path}: expected role {role}, got {metadata.get('split')!r}")
    if int(metadata.get("steps", -1)) != steps:
        raise ValueError(f"{path}: expected {steps} steps, got {metadata.get('steps')!r}")
    if int(metadata.get("resolution", -1)) != resolution:
        raise ValueError(f"{path}: expected resolution {resolution}, got {metadata.get('resolution')!r}")
    if metadata.get("task_manifest_sha256") != task_hash:
        raise ValueError(f"{path}: frozen task manifest hash mismatch")
    if int(metadata.get("fidelity", -1)) != fidelity:
        raise ValueError(f"{path}: expected fidelity {fidelity}, got {metadata.get('fidelity')!r}")
    return result


def _generation_audit(role: str, campaign: str, *, steps: int, dt: float) -> dict[str, Any]:
    audit_dir = PROJECT_ROOT / "artifacts/reports/workspace_audits"
    matches = []
    for path in audit_dir.glob(f"phase1_dev_generate_{role}_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("campaign_id") == campaign and record.get("role") == role:
            matches.append((float(record.get("finished_unix", 0.0)), path, record))
    if not matches:
        raise FileNotFoundError(f"No completed SfePy generation audit for {campaign}/{role}")
    _, path, record = max(matches, key=lambda entry: entry[0])
    command = [str(item) for item in record.get("command", [])]
    expected = {
        "--steps": str(steps),
        "--dt": str(dt),
    }
    for key, value in expected.items():
        if key not in command or command[command.index(key) + 1] != value:
            raise ValueError(f"{path}: generation command does not contain {key} {value}")
    if not record.get("outside_workspace_unchanged", False):
        raise ValueError(f"{path}: workspace audit found an outside-workspace change")
    if record.get("query_used") is not False:
        raise ValueError(f"{path}: query role was used during data generation")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "started_unix": record.get("started_unix"),
        "finished_unix": record.get("finished_unix"),
        "case_count": record.get("case_count"),
        "steps": steps,
        "dt": dt,
        "worker_command": command,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    task_path = PROJECT_ROOT / args.task_manifest
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    task_hash = _sealed_hash(task_manifest)
    tasks = task_manifest["tasks"]
    task_to_boundary = {
        int(task["target_task_id"]): int(task["boundary_id"]) for task in tasks
    }
    if set(task_to_boundary) != set(range(len(tasks))):
        raise ValueError("Task IDs are not contiguous from zero.")

    root = PROJECT_ROOT / "artifacts/data/phase1_development"
    support_path = root / args.support_campaign / f"support_s{args.support_seed}_64.index.json"
    lr_path = root / args.lr_campaign / "validation_lr_64.index.json"
    hr_path = root / args.hr_campaign / "validation_hr_256.index.json"
    indices = {
        "support": _index(support_path, role="support", task_hash=task_hash, steps=args.steps, resolution=64, fidelity=0),
        "validation_lr": _index(lr_path, role="validation_lr", task_hash=task_hash, steps=args.steps, resolution=64, fidelity=0),
        "validation_hr": _index(hr_path, role="validation_hr", task_hash=task_hash, steps=args.steps, resolution=256, fidelity=1),
    }

    role_reports: dict[str, Any] = {}
    ic_sets: dict[str, set[int]] = {}
    boundary_maps: dict[str, dict[int, int]] = {}
    for role, index in indices.items():
        rows = index["shards"]
        counts = Counter(int(row["target_task_id"]) for row in rows)
        if set(counts) != set(task_to_boundary):
            raise ValueError(f"{role}: task set differs from frozen manifest")
        expected_count = args.support_rollouts if role == "support" else args.validation_rollouts
        if any(counts[task_id] != expected_count for task_id in task_to_boundary):
            raise ValueError(f"{role}: per-task rollout count differs from {expected_count}")
        current_boundaries = {
            int(row["target_task_id"]): int(row["boundary_id"]) for row in rows
        }
        if current_boundaries != task_to_boundary:
            raise ValueError(f"{role}: boundary mapping differs from frozen manifest")
        if any(int(row["steps"]) != args.steps for row in rows):
            raise ValueError(f"{role}: at least one shard has a mismatched horizon")
        ic_sets[role] = {int(row["initial_condition_id"]) for row in rows}
        boundary_maps[role] = current_boundaries
        role_reports[role] = {
            "index": str((root / (args.support_campaign if role == "support" else args.lr_campaign if role == "validation_lr" else args.hr_campaign) / Path(support_path.name if role == "support" else lr_path.name if role == "validation_lr" else hr_path.name)).relative_to(PROJECT_ROOT)),
            "index_sha256": index["sha256"],
            "verified_shard_count": len(rows),
            "per_task": {str(task): int(counts[task]) for task in sorted(counts)},
            "steps": sorted({int(row["steps"]) for row in rows}),
            "resolutions": sorted({int(row["resolution"]) for row in rows}),
            "fidelity": index["metadata"].get("fidelity"),
            "task_manifest_sha256": index["metadata"].get("task_manifest_sha256"),
        }

    for first, second in (("support", "validation_lr"), ("support", "validation_hr"), ("validation_lr", "validation_hr")):
        overlap = ic_sets[first] & ic_sets[second]
        if overlap:
            raise ValueError(f"Initial-condition overlap between {first} and {second}: {len(overlap)}")

    support_rows = indices["support"]["shards"]
    support_metadata = indices["support"]["metadata"]
    if int(support_metadata.get("support_seed", -1)) != args.support_seed:
        raise ValueError("Support seed in index metadata does not match requested seed.")
    for task_id in range(len(tasks)):
        task_rows = sorted(
            (row for row in support_rows if int(row["target_task_id"]) == task_id),
            key=lambda row: int(row["support_rank"]),
        )
        ranks = [int(row["support_rank"]) for row in task_rows]
        if ranks != list(range(args.support_rollouts)):
            raise ValueError(f"Task {task_id}: support ranks are not exactly 0..{args.support_rollouts - 1}")
        for budget in (1, 2, 4, 8, 16):
            subset_path = root / args.support_campaign / f"task{task_id:02d}_support_s{args.support_seed}_m{budget}.index.json"
            subset = read_index(subset_path, verify_hashes=True)
            expected_ids = [int(row["initial_condition_id"]) for row in task_rows[:budget]]
            actual_ids = [int(row["initial_condition_id"]) for row in subset["shards"]]
            if actual_ids != expected_ids:
                raise ValueError(f"Task {task_id}: m={budget} is not the nested support prefix")
            if subset["metadata"].get("parent_sha256") != indices["support"]["sha256"]:
                raise ValueError(f"Task {task_id}: m={budget} subset has wrong parent index")

    generation = {
        "support": _generation_audit("support", args.support_campaign, steps=args.steps, dt=args.dt),
        "validation_lr": _generation_audit("validation_lr", args.lr_campaign, steps=args.steps, dt=args.dt),
        "validation_hr": _generation_audit("validation_hr", args.hr_campaign, steps=args.steps, dt=args.dt),
    }
    return {
        "audit_schema": "blrpng_phase1_development_integrity_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_manifest": str(task_path.relative_to(PROJECT_ROOT)),
        "task_manifest_sha256": task_hash,
        "task_count": len(tasks),
        "boundary_ids": [task_to_boundary[index] for index in sorted(task_to_boundary)],
        "campaigns": {
            "support": args.support_campaign,
            "validation_lr": args.lr_campaign,
            "validation_hr": args.hr_campaign,
        },
        "expected_time_domain": {"steps": args.steps, "dt": args.dt, "physical_horizon": args.steps * args.dt},
        "roles": role_reports,
        "generation": generation,
        "cross_role_checks": {
            "same_frozen_task_manifest": True,
            "same_boundary_mapping": True,
            "aligned_horizon": True,
            "initial_condition_overlap": {
                "support_vs_validation_lr": 0,
                "support_vs_validation_hr": 0,
                "validation_lr_vs_validation_hr": 0,
            },
            "nested_support_budgets": [1, 2, 4, 8, 16],
        },
        "query_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase-1 development split hashes, horizons, and task pairing.")
    parser.add_argument("--task-manifest", default="artifacts/manifests/phase1_development/hrdev_20261012/tasks.json")
    parser.add_argument("--support-campaign", default="hrdev_20261012_t20_ss271829")
    parser.add_argument("--lr-campaign", default="hrdev_20261012_t20_ss271829")
    parser.add_argument("--hr-campaign", default="hrdev_20261012")
    parser.add_argument("--support-seed", type=int, default=271829)
    parser.add_argument("--support-rollouts", type=int, default=16)
    parser.add_argument("--validation-rollouts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output", default="artifacts/reports/phase1_hrdev_20261012_t20_integrity_audit.json")
    args = parser.parse_args()
    result = audit(args)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "task_count": result["task_count"], "roles": result["roles"], "cross_role_checks": result["cross_role_checks"]}, indent=2))


if __name__ == "__main__":
    main()
