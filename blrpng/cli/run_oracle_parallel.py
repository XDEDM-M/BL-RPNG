from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent oracle task curves on two frozen GPUs.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = list(range(int(config["data"]["target_boundaries"])))
    groups = (tasks[::2], tasks[1::2])
    outputs = ["artifacts/reports/v2_oracle_curves_gpu0.json",
               "artifacts/reports/v2_oracle_curves_gpu1.json"]
    progress = ["artifacts/reports/v2_oracle_progress_gpu0.json",
                "artifacts/reports/v2_oracle_progress_gpu1.json"]
    processes = []
    with workspace_audit("run_oracle_parallel") as audit:
        for gpu, group in enumerate(groups):
            command = [sys.executable, "-m", "blrpng.cli.run_oracle_curve",
                       "--config", args.config, "--seed", str(args.seed),
                       "--output", outputs[gpu], "--progress-output", progress[gpu]]
            for task in group:
                command += ["--task", str(task)]
            if args.skip_existing:
                command.append("--skip-existing")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = (PROJECT_ROOT / f"artifacts/experiments/v2_logs/oracle_gpu{gpu}.log").open("w")
            processes.append((subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                                stdout=log, stderr=subprocess.STDOUT), log))
        for process, log in processes:
            return_code = process.wait()
            log.close()
            if return_code:
                raise subprocess.CalledProcessError(return_code, process.args)
        combined = {"schema": "blrpng_oracle_curve_v1", "seed": args.seed,
                    "tasks": {}, "frozen": True}
        for output in outputs:
            payload = json.loads(resolve_in_project(output, must_exist=True).read_text())
            combined["tasks"].update(payload["tasks"])
        combined["tasks"] = dict(sorted(combined["tasks"].items(), key=lambda item: int(item[0])))
        output = write_json("artifacts/reports/v2_oracle_curves.json", combined)
        audit["tasks"] = len(combined["tasks"])
    print(json.dumps({"output": str(output), "tasks": len(combined["tasks"])}, indent=2))


if __name__ == "__main__":
    main()
