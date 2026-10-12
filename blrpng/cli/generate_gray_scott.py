from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from blrpng.config import load_config
from blrpng.data.generate import SFEPY_PYTHON, workspace_audit
from blrpng.data.manifest import make_cases, write_manifest
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
)


def _convert(config, manifest, split, resolution, mesh, steps, fidelity):
    stem = f"gray_scott_{split}_{resolution}"
    npz = ensure_artifact_path(f"artifacts/tmp/{stem}.npz")
    meta_path = ensure_artifact_path(f"artifacts/tmp/{stem}.json")
    output = ensure_artifact_path(f"artifacts/data/{stem}.pt")
    data = config["data"]
    command = [
        str(SFEPY_PYTHON), "-m", "blrpng.solvers.sfepy_gray_scott_worker",
        "--manifest", str(manifest), "--split", split,
        "--output-npz", str(npz), "--output-meta", str(meta_path),
        "--output-resolution", str(resolution), "--mesh-resolution", str(mesh),
        "--steps", str(steps), "--fidelity", str(fidelity),
        "--progress-threshold", str(data["residual_progress_threshold"]),
        "--dt", str(data["dt"]), "--diffusion-u", str(data["diffusion_u"]),
        "--diffusion-v", str(data["diffusion_v"]), "--feed", str(data["feed"]),
        "--kill", str(data["kill"]),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(f"Gray--Scott worker failed:\n{completed.stdout}\n{completed.stderr}")
    with np.load(npz, allow_pickle=False) as archive:
        payload = {key: torch.from_numpy(archive[key]) for key in archive.files}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "schema_version": "blrpng_dataset_v3",
            "num_time_steps": steps,
            "config_sha256": config["_config_sha256"],
            "manifest_file_sha256": sha256_file(manifest),
            "generator_python": str(SFEPY_PYTHON),
            "converter_python": sys.executable,
        }
    )
    payload["meta"] = meta
    torch.save(payload, output)
    npz.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    gate = resolve_in_project(config["experiment"]["prerequisite_gate"], must_exist=True)
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if not gate_payload.get("passed"):
        raise PermissionError("Allen--Cahn v2 oracle-parity gate has not passed.")
    data = config["data"]
    boundary_count = 1 if args.smoke else max(data["boundary_budgets"])
    initial_count = 1 if args.smoke else int(data["initial_conditions_per_boundary"])
    cases = make_cases(
        count=boundary_count * initial_count,
        split="train",
        seed=int(config["experiment"]["seed"]),
        frequencies=tuple(data["boundary_frequencies"]),
        amplitude_range=(data["boundary_amplitude"], data["boundary_amplitude"]),
        fixed_boundaries=boundary_count,
        initial_conditions_per_boundary=initial_count,
    )
    with workspace_audit("generate_gray_scott") as audit:
        manifest = write_manifest(
            "artifacts/manifests/gray_scott.json",
            cases,
            metadata={
                "config_sha256": config["_config_sha256"],
                "allen_cahn_gate": str(gate),
                "frozen": True,
            },
        )
        steps = 1 if args.smoke else int(data["num_train_time_steps"])
        coarse_resolution = 16 if args.smoke else int(data["coarse_grid"])
        coarse_mesh = 17 if args.smoke else int(data["coarse_mesh"])
        output = _convert(
            config, manifest, "train", coarse_resolution, coarse_mesh, steps, 0
        )
        audit["output"] = str(output)
    print(output)


if __name__ == "__main__":
    main()
