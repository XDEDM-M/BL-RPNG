from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from blrpng.analysis.bootstrap import select_oracle_plateau
from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-specific HR specialist learning curves.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="artifacts/reports/v2_oracle_curves.json")
    parser.add_argument("--progress-output", default="artifacts/reports/v2_oracle_curve_progress.json")
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = args.task or list(range(int(config["data"]["target_boundaries"])))
    budgets = [int(value) for value in config["data"]["oracle_budgets"]]
    threshold = float(config["statistics"]["oracle_plateau_improvement"])
    report = {"schema": "blrpng_oracle_curve_v1", "seed": args.seed, "tasks": {}}
    audit_name = "run_oracle_curve_" + "_".join(str(task) for task in tasks)
    with workspace_audit(audit_name) as audit:
        for task in tasks:
            errors = {}
            for budget in budgets:
                checkpoint = PROJECT_ROOT / f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{args.seed}_hr_oracle.pt"
                if not (args.skip_existing and checkpoint.exists()):
                    subprocess.run([
                        sys.executable, "-m", "blrpng.cli.train_task", "--config", args.config,
                        "--task", str(task), "--budget", str(budget),
                        "--method", "hr_oracle", "--seed", str(args.seed),
                    ], cwd=PROJECT_ROOT, check=True)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                errors[budget] = float(payload["best_validation_metric"])
                plateau = select_oracle_plateau(errors, improvement_threshold=threshold)
                transitions = plateau["relative_improvements"]
                if transitions and list(transitions.values())[-1] < threshold:
                    break
            selection = select_oracle_plateau(errors, improvement_threshold=threshold)
            selected_budget = int(selection["selected_budget"])
            hashes = {}
            for budget in errors:
                checkpoint = PROJECT_ROOT / f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{args.seed}_hr_oracle.pt"
                hashes[str(budget)] = sha256_file(checkpoint)
            selection["checkpoint_sha256"] = hashes
            selection["selected_checkpoint"] = (
                f"artifacts/checkpoints/v2/v2_task{task:02d}_m{selected_budget}_s{args.seed}_hr_oracle.pt"
            )
            report["tasks"][str(task)] = selection
            write_json(args.progress_output, report)
            # The frozen curve retains metrics/hashes for every budget but only
            # the saturated specialist weights, as required by the 300 GB gate.
            for budget in errors:
                if int(budget) == selected_budget:
                    continue
                checkpoint = PROJECT_ROOT / f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{args.seed}_hr_oracle.pt"
                checkpoint.unlink(missing_ok=True)
        audit["tasks"] = len(tasks)
        output = write_json(args.output, {**report, "frozen": True})
    print(json.dumps({"output": str(output), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
