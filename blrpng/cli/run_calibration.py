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
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, write_json


def _checkpoint(task: int, budget: int, seed: int, method: str) -> Path:
    suffix = ".adapter.pt" if method in {
        "lora", "nlt", "ordinary_lora", "ordinary_nlt", "nlt_path",
        "nlt_commutator", "nlt_commutator_defect", "blrpng_v2",
        "blrpng_plus", "blrpng_plus_d1e3", "blrpng_plus_d1e2",
        "blrpng_v2_nlt", "blrpng_plus_nlt",
    } else ".pt"
    return PROJECT_ROOT / f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{seed}_{method}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched seed42 target-task calibration matrix.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--methods", default="pathpre_final,lora,nlt,blrpng_v2")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    tasks = [
        (task, method) for task in range(int(config["data"]["target_boundaries"]))
        for method in methods
    ]
    progress_path = ensure_artifact_path("artifacts/reports/v2_calibration_progress.json")
    with workspace_audit("run_calibration") as audit:
        running = []
        completed = []
        pending = list(tasks)
        while pending or running:
            while pending and len(running) < args.max_parallel:
                task, method = pending.pop(0)
                expected = _checkpoint(task, args.budget, args.seed, method)
                if args.skip_existing and expected.exists():
                    completed.append({"task": task, "method": method, "checkpoint": str(expected), "skipped": True})
                    continue
                log = ensure_artifact_path(f"artifacts/experiments/v2_logs/task{task:02d}_{method}.log")
                stream = log.open("w", encoding="utf-8")
                used_gpus = {item[-1] for item in running}
                gpu = next(value for value in range(args.max_parallel) if value not in used_gpus)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = [
                    sys.executable, "-m", "blrpng.cli.train_task", "--config", args.config,
                    "--task", str(task), "--budget", str(args.budget),
                    "--method", method, "--seed", str(args.seed),
                ]
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True)
                running.append((process, stream, log, task, method, expected, gpu))
            time.sleep(0.25)
            still = []
            for process, stream, log, task, method, expected, gpu in running:
                if process.poll() is None:
                    still.append((process, stream, log, task, method, expected, gpu)); continue
                stream.close()
                if process.returncode:
                    raise RuntimeError(f"Calibration task failed: task={task}, method={method}\n{log.read_text(encoding='utf-8', errors='replace')}")
                if not expected.exists():
                    raise FileNotFoundError(f"Expected calibration checkpoint was not written: {expected}")
                completed.append({"task": task, "method": method, "checkpoint": str(expected), "log": str(log)})
            running = still
            write_json(progress_path, {"schema": "blrpng_calibration_progress_v1", "completed": completed, "remaining": len(pending) + len(running)})
        audit["completed"] = len(completed)
    print(json.dumps({"completed": len(completed), "progress": str(progress_path)}, indent=2))


if __name__ == "__main__":
    main()
