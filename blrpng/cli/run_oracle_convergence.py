from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import ARTIFACT_ROOT, PROJECT_ROOT, ensure_artifact_path, write_json


METHODS = {
    "scratch": "hr_oracle",
    "ordinary": "hr_oracle_ordinary",
    "path": "hr_oracle_path",
}


def _artifact_gib() -> float:
    return sum(path.stat().st_size for path in ARTIFACT_ROOT.rglob("*") if path.is_file()) / 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description="Convergence audit for the HR specialist.")
    parser.add_argument("--config", default="configs/allen_cahn_oracle_convergence.yaml")
    parser.add_argument("--tasks", default="0,2,7")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--learning-rates", default="0.001,0.0003,0.0001")
    parser.add_argument("--initializations", default="scratch,ordinary,path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    epochs = int(args.epochs or config["training"]["epochs"])
    tasks = [int(value) for value in args.tasks.split(",") if value]
    rates = [float(value) for value in args.learning_rates.split(",") if value]
    initializations = [value for value in args.initializations.split(",") if value]
    unknown = sorted(set(initializations) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown initialization labels: {unknown}")
    stage_limit = float(config["resources"]["stage_budget_gb"])
    baseline = _artifact_gib()
    jobs = [(task, initialization, rate) for task in tasks for initialization in initializations for rate in rates]
    progress = ensure_artifact_path("artifacts/reports/oracle_convergence_progress.json")
    with workspace_audit("run_oracle_convergence") as audit:
        pending = list(jobs)
        running = []
        completed = []
        while pending or running:
            if _artifact_gib() - baseline > stage_limit:
                raise RuntimeError("Oracle convergence stage exceeded its storage budget.")
            while pending and len(running) < args.max_parallel:
                task, initialization, rate = pending.pop(0)
                method = METHODS[initialization]
                lr_tag = f"lr{rate:.0e}".replace("-", "m")
                tag = f"conv_{initialization}_{lr_tag}_e{epochs}"
                expected = PROJECT_ROOT / (
                    f"artifacts/checkpoints/v2/v2_task{task:02d}_m{args.budget}_s{args.seed}_"
                    f"{method}_{tag}.pt"
                )
                if args.skip_existing and expected.exists():
                    completed.append({"task": task, "initialization": initialization, "lr": rate,
                                      "checkpoint": str(expected), "skipped": True})
                    continue
                log = ensure_artifact_path(
                    f"artifacts/experiments/oracle_convergence/task{task:02d}_{tag}.log"
                )
                stream = log.open("w", encoding="utf-8")
                used = {row[-1] for row in running}
                gpu = next(index for index in range(args.max_parallel) if index not in used)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = [
                    sys.executable, "-m", "blrpng.cli.train_task",
                    "--config", args.config, "--task", str(task),
                    "--budget", str(args.budget), "--method", method,
                    "--seed", str(args.seed), "--lr", str(rate),
                    "--epochs", str(epochs), "--run-tag", tag,
                ]
                recovery = PROJECT_ROOT / (
                    f"artifacts/checkpoints/v2_task{task:02d}_m{args.budget}_s{args.seed}_"
                    f"{method}_{tag}_recovery.pt"
                )
                if recovery.exists():
                    command.extend(("--resume", str(recovery)))
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, text=True,
                )
                running.append((process, stream, log, task, initialization, rate, expected, gpu))
            time.sleep(0.5)
            still = []
            for row in running:
                process, stream, log, task, initialization, rate, expected, gpu = row
                if process.poll() is None:
                    still.append(row)
                    continue
                stream.close()
                if process.returncode:
                    raise RuntimeError(
                        f"Oracle convergence job failed: {task}/{initialization}/{rate}\n"
                        + log.read_text(encoding="utf-8", errors="replace")
                    )
                completed.append({"task": task, "initialization": initialization, "lr": rate,
                                  "checkpoint": str(expected), "log": str(log)})
            running = still
            write_json(progress, {"schema": "blrpng_oracle_convergence_progress_v1",
                                  "epochs": epochs, "completed": completed,
                                  "remaining": len(pending) + len(running)})
        audit["completed"] = len(completed)
    print(json.dumps({"completed": len(completed), "progress": str(progress)}, indent=2))


if __name__ == "__main__":
    main()
