from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from blrpng.config import load_config
from blrpng.data.generate import SFEPY_PYTHON, workspace_audit
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, write_json


def _run(config: dict, *, split: str, resolution: int, case_start: int, repeat: int) -> dict:
    stem = f"solver_benchmark_{split}_{resolution}_r{repeat}"
    npz = ensure_artifact_path(f"artifacts/tmp/{stem}.npz")
    meta = ensure_artifact_path(f"artifacts/tmp/{stem}.json")
    data = config["data"]
    command = [
        str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_worker",
        "--manifest", str(resolve_in_project("artifacts/manifests/allen_cahn_v2_calibration.json", must_exist=True)),
        "--split", split, "--output-npz", str(npz), "--output-meta", str(meta),
        "--output-resolution", str(resolution), "--mesh-resolution", str(resolution + 1),
        "--steps", str(data["train_steps"]), "--fidelity", str(int(resolution > 64)),
        "--dt", str(data["dt"]), "--epsilon", str(data["eps"]), "--gamma", str(data["gamma"]),
        "--progress-threshold", str(data["residual_progress_threshold"]),
        "--case-start", str(case_start), "--limit", "1",
    ]
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    with np.load(npz, allow_pickle=False) as archive:
        solve = float(np.asarray(archive["solver_seconds"]).sum())
        assembly = float(np.asarray(archive["assembly_seconds"]).sum())
        converged = bool(np.asarray(archive["solver_converged"]).all())
    npz.unlink(missing_ok=True); meta.unlink(missing_ok=True)
    return {"repeat": repeat, "assembly_seconds": assembly, "solve_seconds": solve,
            "total_seconds": assembly + solve, "converged": converged}


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-repeat target solver timing benchmark.")
    parser.add_argument("--config", default="configs/allen_cahn_v2_calibration.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="artifacts/reports/v2_solver_cost_benchmark.json")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = {"lr64": [], "hr256": []}
    with workspace_audit("benchmark_solver_cost"):
        # Exclude import/JIT/filesystem warm-up from the repeated deployment
        # timings, while preserving the warm-up result for auditability.
        warmup = {
            "lr64": _run(config, split="support_lr", resolution=64, case_start=0, repeat=-1),
            "hr256": _run(config, split="oracle_train", resolution=256, case_start=0, repeat=-1),
        }
        for repeat in range(args.repeats):
            rows["lr64"].append(_run(config, split="support_lr", resolution=64,
                                     case_start=0, repeat=repeat))
            rows["hr256"].append(_run(config, split="oracle_train", resolution=256,
                                      case_start=0, repeat=repeat))
        summary = {}
        for name, values in rows.items():
            totals = np.asarray([row["total_seconds"] for row in values])
            summary[name] = {"mean_seconds": float(totals.mean()),
                             "std_seconds": float(totals.std(ddof=1)) if len(totals) > 1 else 0.0,
                             "cv": float(totals.std(ddof=1) / totals.mean()) if len(totals) > 1 else 0.0}
        output = write_json(args.output, {"schema": "blrpng_solver_cost_benchmark_v1",
                                          "repeats": args.repeats, "warmup": warmup,
                                          "runs": rows, "summary": summary})
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
