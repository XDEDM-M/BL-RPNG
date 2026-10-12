from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from blrpng.config import load_config
from blrpng.data.capacity import artifact_size_bytes
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, write_json


METHODS = {
    "scratch": ("lr_scratch", "pt"),
    "nlt": ("nlt", "adapter.pt"),
    "ordinary_nlt": ("ordinary_nlt", "adapter.pt"),
    "nlt_commutator": ("nlt_commutator", "adapter.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convergence audit for LR scratch and NLT adapters.")
    parser.add_argument("--config", default="configs/allen_cahn_adapter_convergence.yaml")
    parser.add_argument("--tasks", default="0,2,7")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--learning-rates", default="0.001,0.0003,0.0001")
    parser.add_argument("--methods", default="scratch,nlt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = [int(value) for value in args.tasks.split(",") if value]
    rates = [float(value) for value in args.learning_rates.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    baseline_bytes = artifact_size_bytes()
    limit = float(config["resources"]["stage_budget_gb"]) * 2**30
    jobs = [(task, method, rate) for task in tasks for method in methods for rate in rates]
    pending, running, completed = list(jobs), [], []
    progress_path = ensure_artifact_path("artifacts/reports/adapter_convergence_progress.json")
    with workspace_audit("run_adapter_convergence") as audit:
        while pending or running:
            if artifact_size_bytes() - baseline_bytes > limit:
                raise RuntimeError("Adapter convergence audit exceeded its storage budget.")
            while pending and len(running) < min(args.max_parallel, len(gpu_ids)):
                task, label, rate = pending.pop(0)
                method, extension = METHODS[label]
                lr_tag = f"lr{rate:.0e}".replace("-", "m")
                tag = f"adaptconv_{label}_{lr_tag}_e{args.epochs}"
                expected = PROJECT_ROOT / (
                    f"artifacts/checkpoints/v2/v2_task{task:02d}_m{args.budget}_s{args.seed}_"
                    f"{method}_{tag}.{extension}"
                )
                if args.skip_existing and expected.exists():
                    completed.append({"task": task, "method": label, "lr": rate,
                                      "checkpoint": str(expected), "skipped": True})
                    continue
                gpu = next(index for index in gpu_ids
                           if index not in {row[-1] for row in running})
                log = ensure_artifact_path(
                    f"artifacts/experiments/adapter_convergence/task{task:02d}_{tag}.log"
                )
                stream = log.open("w", encoding="utf-8")
                environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = [
                    sys.executable, "-m", "blrpng.cli.train_task",
                    "--config", args.config, "--task", str(task),
                    "--budget", str(args.budget), "--method", method,
                    "--seed", str(args.seed), "--lr", str(rate),
                    "--epochs", str(args.epochs), "--run-tag", tag,
                ]
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, text=True,
                )
                running.append((process, stream, log, task, label, rate, expected, gpu))
            time.sleep(0.5)
            active = []
            for row in running:
                process, stream, log, task, label, rate, expected, gpu = row
                if process.poll() is None:
                    active.append(row); continue
                stream.close()
                if process.returncode:
                    raise RuntimeError(
                        f"Adapter convergence failed: {task}/{label}/{rate}\n"
                        + log.read_text(encoding="utf-8", errors="replace")
                    )
                completed.append({"task": task, "method": label, "lr": rate,
                                  "checkpoint": str(expected), "log": str(log)})
            running = active
            write_json(progress_path, {
                "schema": "blrpng_adapter_convergence_progress_v1",
                "completed": completed, "remaining": len(pending) + len(running),
            })
        audit["completed"] = len(completed)
    print(json.dumps({"completed": len(completed), "progress": str(progress_path)}, indent=2))


if __name__ == "__main__":
    main()
