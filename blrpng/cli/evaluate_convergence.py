from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout_index
from blrpng.paths import PROJECT_ROOT, sha256_file, write_json


HR_PATTERN = re.compile(
    r"v2_task(?P<task>\d+)_m(?P<budget>\d+)_s(?P<seed>\d+)_"
    r"(?P<method>hr_oracle(?:_ordinary|_path)?)_conv_(?P<label>scratch|ordinary|path)_"
    r"lr(?P<lr>[^_]+)_e(?P<epochs>\d+)\.pt$"
)
ADAPTER_PATTERN = re.compile(
    r"v2_task(?P<task>\d+)_m(?P<budget>\d+)_s(?P<seed>\d+)_"
    r"(?P<method>lr_scratch|nlt|ordinary_nlt|nlt_commutator)_adaptconv_"
    r"(?P<label>scratch|nlt|ordinary_nlt|nlt_commutator)_"
    r"lr(?P<lr>[^_]+)_e(?P<epochs>\d+)\.(?:adapter\.)?pt$"
)


def _metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("metadata", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-select and evaluate convergence checkpoints.")
    parser.add_argument("--kind", choices=("hr", "adapter"), required=True)
    parser.add_argument("--tasks", default="0")
    parser.add_argument("--resolution", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--horizon", type=int, choices=(20, 50), default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    wanted = {int(value) for value in args.tasks.split(",") if value}
    pattern = HR_PATTERN if args.kind == "hr" else ADAPTER_PATTERN
    candidates = []
    for path in sorted((PROJECT_ROOT / "artifacts/checkpoints/v2").glob("*conv*")):
        match = pattern.match(path.name)
        if not match or int(match.group("task")) not in wanted:
            continue
        metadata = _metadata(path)
        candidates.append({
            **match.groupdict(), "task": int(match.group("task")),
            "budget": int(match.group("budget")), "seed": int(match.group("seed")),
            "epochs": int(match.group("epochs")), "checkpoint": str(path),
            "validation_metric": float(metadata["best_validation_metric"]),
        })
    selected = []
    for task in sorted(wanted):
        labels = sorted({row["label"] for row in candidates if row["task"] == task})
        for label in labels:
            rows = [row for row in candidates if row["task"] == task and row["label"] == label]
            selected.append(min(rows, key=lambda row: row["validation_metric"]))
    split = "query_512" if args.resolution == 512 else "query_hr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    with workspace_audit("evaluate_convergence") as audit:
        for row in selected:
            index_path = (
                f"artifacts/data/tasks/task{row['task']:02d}_{split}_"
                f"{args.resolution}_all.index.json"
            )
            index_payload = read_index(index_path)
            boundary_id = int(index_payload["shards"][0]["boundary_id"])
            model, _ = load_checkpoint_model(row["checkpoint"], device=device)
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = projected_rollout_index(
                model, index_path, device=device, horizon=args.horizon
            )
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for initial_id, error in zip(
                metrics["initial_condition_ids"], metrics["per_rollout_rel_l2"]
            ):
                records.append({
                    "seed": row["seed"], "target_task_id": row["task"],
                    "boundary_id": boundary_id, "initial_condition_id": initial_id,
                    "method": row["label"], "relative_l2": error,
                    "validation_metric": row["validation_metric"],
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": sha256_file(row["checkpoint"]),
                    "evaluation_seconds_task": elapsed,
                })
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
        output = write_json(args.output, {
            "schema": "blrpng_convergence_evaluation_v1", "kind": args.kind,
            "selection_rule": "minimum frozen validation rollout error within label",
            "resolution": args.resolution, "horizon": args.horizon,
            "candidates": candidates, "selected": selected, "records": records,
        })
        audit["output"] = str(output)
    means = {
        label: sum(row["relative_l2"] for row in records if row["method"] == label)
        / sum(row["method"] == label for row in records)
        for label in sorted({row["method"] for row in records})
    }
    print(json.dumps({"output": str(output), "means": means, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
