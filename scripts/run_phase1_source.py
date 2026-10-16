from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import (
    PROJECT_ROOT,
    exclusive_artifact_lock,
    sha256_file,
    sha256_json,
    write_json,
)
from blrpng.training.engine import train


LEDGER = PROJECT_ROOT / "docs/EXPERIMENTS.csv"
FIELDS = [
    "trial_id", "status", "round", "objective", "seed", "task_ids",
    "config_sha256", "elapsed_gpu_hours", "validation_metric", "query_metric",
    "raw_output_path", "notes",
]


def append_ledger(row: dict) -> None:
    exists = LEDGER.exists() and LEDGER.stat().st_size > 0
    with LEDGER.open("a", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start one Allen-Cahn HR source model for R1.")
    parser.add_argument("--objective", choices=("path", "final"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--resume-from", help="Resume optimizer/model state from a recovery checkpoint.")
    parser.add_argument(
        "--resume-from-scratch", action="store_true",
        help="Resume a random-initialization source run without a warm-start source checkpoint.",
    )
    parser.add_argument("--epochs", type=int, help="Override the source training epoch cap.")
    parser.add_argument(
        "--attempt-suffix",
        help="Add a safe suffix for a diagnostic or separate recovery attempt.",
    )
    args = parser.parse_args()
    if args.from_scratch and (args.resume_from or args.resume_from_scratch):
        raise ValueError("--from-scratch cannot be combined with a resume option.")
    if args.resume_from_scratch and not args.resume_from:
        raise ValueError("--resume-from-scratch requires --resume-from.")
    if args.attempt_suffix and not all(
        character.isalnum() or character in "_-" for character in args.attempt_suffix
    ):
        raise ValueError("--attempt-suffix may contain only letters, numbers, '_' and '-'.")
    config_path = "configs/phase1_r1_source.yaml"
    cfg = load_config(config_path)
    objective = "blrpng" if args.objective == "path" else "final_only"
    initialization = (
        None if args.from_scratch or args.resume_from_scratch
        else cfg["source_checkpoints"][args.objective]
    )
    resume_checkpoint = args.resume_from
    training = dict(cfg["training"])
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        training["epochs"] = int(args.epochs)
    suffix = (
        "_scratch" if args.from_scratch
        else "_resume_scratch" if args.resume_from_scratch
        else "_resume" if resume_checkpoint
        else ""
    )
    if args.attempt_suffix:
        suffix += f"_{args.attempt_suffix}"
    output = f"artifacts/checkpoints/phase1/R1/source_{args.objective}_s{args.seed}{suffix}.pt"
    training_mode = (
        "from_scratch" if args.from_scratch
        else "resume_from_scratch" if args.resume_from_scratch
        else "resume" if resume_checkpoint
        else "warm_start"
    )
    descriptor = {
        "round": "R1",
        "objective": objective,
        "training_mode": training_mode,
        "attempt_suffix": args.attempt_suffix,
        "seed": args.seed,
        "source_initialization": initialization,
        "source_sha256": sha256_file(PROJECT_ROOT / initialization) if initialization else None,
        "resume_checkpoint": resume_checkpoint,
        "resume_sha256": (
            sha256_file(PROJECT_ROOT / resume_checkpoint) if resume_checkpoint else None
        ),
        "train_data": cfg["data_paths"]["train"],
        "train_data_sha256": sha256_file(PROJECT_ROOT / cfg["data_paths"]["train"]),
        "validation_data": cfg["data_paths"]["validation"],
        "validation_data_sha256": sha256_file(PROJECT_ROOT / cfg["data_paths"]["validation"]),
        "training": training,
        "config_sha256": sha256_file(PROJECT_ROOT / config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "engine_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/engine.py"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
    }
    trial_hash = sha256_json(descriptor)
    trial_id = f"R1-source-{args.objective}-{training_mode}-s{args.seed}"
    if args.attempt_suffix:
        trial_id += f"-{args.attempt_suffix}"
    if (PROJECT_ROOT / output).exists():
        raise FileExistsError(f"Refusing to overwrite an existing source checkpoint: {output}")
    row = {
        "trial_id": trial_id,
        "status": "RUNNING",
        "round": "R1",
        "objective": objective,
        "seed": args.seed,
        "task_ids": "source_A",
        "config_sha256": trial_hash,
        "raw_output_path": output,
        "notes": (
            f"HR source train/validation only; {training_mode}; no target HR query"
            + (f"; attempt_suffix={args.attempt_suffix}; diagnostic-only" if args.attempt_suffix else "")
        ),
    }
    append_ledger(row)
    print(
        f"START trial={trial_id} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')} "
        f"mode={training_mode} init={initialization} output={output}",
        flush=True,
    )
    local = {
        "experiment": {
            "name": "allen_cahn_phase1_r1_source_continuation",
            "seed": 20261012,
            "training_seed": args.seed,
            "deterministic": True,
        },
        "data": {
            "equation": "allen_cahn",
            "representation": "lifted",
            "dt": 0.2,
            "eps": 0.03,
            "gamma": 5.0,
        },
        "model": cfg["model"],
        "training": training,
        "_config_path": config_path,
        "_config_sha256": trial_hash,
    }
    started = time.perf_counter()
    try:
        with exclusive_artifact_lock("phase1_training_gpu0"):
            with workspace_audit("phase1_source_train") as audit:
                result = train(
                    local,
                    seed=args.seed,
                    objective=objective,
                    initialization_checkpoint=initialization,
                    resume_checkpoint=resume_checkpoint,
                    output_checkpoint=output,
                    train_data_path=cfg["data_paths"]["train"],
                    validation_data_paths=[cfg["data_paths"]["validation"]],
                )
                audit.update({
                    "round": "R1",
                    "trial_id": trial_id,
                    "config_sha256": trial_hash,
                    "output": output,
                    "objective": objective,
                    "query_used": False,
                })
        elapsed = time.perf_counter() - started
        checkpoint_sha = sha256_file(PROJECT_ROOT / output)
        best = min(result["history"], key=lambda item: item["validation"]["rollout_rel_l2"])
        report = {
            "descriptor": descriptor,
            "trial_id": trial_id,
            "config_sha256": trial_hash,
            "checkpoint": output,
            "checkpoint_sha256": checkpoint_sha,
            "best_validation_metric": float(result["best_validation_metric"]),
            "best_epoch": int(best["epoch"]),
            "epochs_run": len(result["history"]),
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "query_used": False,
            "history": result["history"],
        }
        write_json(f"artifacts/reports/phase1_{trial_id}.json", report)
        append_ledger({
            **row,
            "status": "DONE",
            "elapsed_gpu_hours": report["gpu_hours"],
            "validation_metric": report["best_validation_metric"],
            "raw_output_path": f"artifacts/reports/phase1_{trial_id}.json",
            "notes": f"best_epoch={report['best_epoch']}; query unused; checkpoint_sha256={checkpoint_sha}",
        })
        print(json.dumps({key: report[key] for key in (
            "trial_id", "checkpoint", "best_validation_metric", "best_epoch",
            "epochs_run", "elapsed_seconds", "gpu_hours",
        )}, indent=2))
    except Exception as exc:
        elapsed = time.perf_counter() - started
        append_ledger({
            **row,
            "status": "FAILED",
            "elapsed_gpu_hours": elapsed / 3600.0,
            "raw_output_path": output,
            "notes": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()
