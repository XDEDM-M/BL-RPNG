from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, sha256_json
from blrpng.data.confirmation_query_gate import (
    QUERY_CASE_ID_BASE,
    QUERY_CASE_MANIFEST_PATH,
    QUERY_IC_SEED_OFFSET,
    QUERY_SOLVER_PROTOCOL,
    capture_query_runtime_environment,
)


MODEL_SEEDS = [0, 1, 2, 3, 42]
SUPPORT_SEEDS = [271829, 271830, 271831]
SUPPORT_BUDGET = 4


def freeze_matrix(campaign: str, method_configs: list[str]) -> Path:
    task_path = PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "tasks.json"
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    task_seal = task_manifest.get("sha256")
    if task_manifest.get("schema") != "blrpng_phase1_confirmation_tasks_v1":
        raise ValueError("Expected a frozen confirmation task manifest.")
    if task_manifest.get("campaign_id") != campaign:
        raise ValueError("Campaign ID does not match the frozen task manifest.")
    if task_seal != sha256_json({key: value for key, value in task_manifest.items() if key != "sha256"}):
        raise ValueError("Confirmation task manifest seal is invalid.")

    configs: dict[str, dict[str, str]] = {}
    for item in method_configs:
        if "=" not in item:
            raise ValueError("Each --method-config must be METHOD_ID=CONFIG_PATH.")
        method_id, config_rel = item.split("=", 1)
        if not method_id or method_id in configs:
            raise ValueError(f"Method IDs must be non-empty and unique: {method_id!r}")
        config_path = (PROJECT_ROOT / config_rel).resolve()
        config_path.relative_to(PROJECT_ROOT)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if method_id not in {str(row["id"]) for row in spec.get("trials", [])}:
            raise ValueError(f"Method {method_id!r} is not a trial in {config_rel}.")
        configs[method_id] = {
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": sha256_file(config_path),
        }
    if not configs:
        raise ValueError("At least one confirmation method must be frozen.")

    query_runtime_environment = capture_query_runtime_environment()
    if query_runtime_environment["numpy"] != np.__version__:
        raise RuntimeError(
            "The SfePy worker and training environments use different NumPy versions; "
            "do not freeze a query matrix with an ambiguous numerical runtime."
        )

    task_ids = sorted(int(row["target_task_id"]) for row in task_manifest["tasks"])
    matrix = {
        "schema": "blrpng_phase1_confirmation_matrix_v1",
        "campaign_id": campaign,
        "task_manifest_sha256": task_seal,
        "methods": list(configs),
        "method_configs": configs,
        "model_seeds": MODEL_SEEDS,
        "support_seeds": SUPPORT_SEEDS,
        "support_budget": SUPPORT_BUDGET,
        "support_rollouts": 16,
        "lr_validation_rollouts": 8,
        "query_rollouts": 8,
        "lr_steps": 20,
        "query_steps": 20,
        "query_case_manifest_path": QUERY_CASE_MANIFEST_PATH.format(campaign=campaign),
        "query_ic_seed": int(task_manifest["seed"]) + QUERY_IC_SEED_OFFSET,
        "query_case_id_base": QUERY_CASE_ID_BASE,
        "query_numpy_version": np.__version__,
        "query_runtime_environment": query_runtime_environment,
        "query_solver_protocol": QUERY_SOLVER_PROTOCOL,
        "task_ids": task_ids,
        "primary_metric": "20-step rollout relative-L2; LR-validation-selected checkpoints",
        "query_policy": "No HR query generation before every matrix checkpoint and analysis program are frozen.",
    }
    matrix["sha256"] = sha256_json(matrix)
    output = ensure_artifact_path(
        PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "matrix.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(matrix, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    rel = str(output.relative_to(PROJECT_ROOT))
    amendment = (
        f"\n## Amendment 2 — frozen confirmation matrix ({matrix['sha256']})\n\n"
        f"- Matrix manifest: `{rel}` (SHA-256 `{matrix['sha256']}`).\n"
        f"- Methods: `{', '.join(matrix['methods'])}` with config paths/hashes sealed in that manifest.\n"
        f"- Full crossed checkpoint grid: {len(matrix['methods'])} methods × 5 model seeds "
        f"`{MODEL_SEEDS}` × 3 support seeds `{SUPPORT_SEEDS}` × {len(task_ids)} tasks, at m=4.\n"
        "- Frozen data sizes are 16 LR support rollouts per task/support seed, 8 independent LR validation rollouts, and 8 independent 256×256 HR query rollouts per task; each role is 20 steps.\n"
        "- The query unlocker requires every cell in this grid and verifies each checkpoint and analysis-file hash.\n"
        f"- Confirmation HR query cases use `{matrix['query_case_manifest_path']}`, IC seed `{matrix['query_ic_seed']}`, "
        f"case ID base `{QUERY_CASE_ID_BASE}`, NumPy `{matrix['query_numpy_version']}`, and the frozen solver protocol `{QUERY_SOLVER_PROTOCOL}`.\n"
        f"- Solver runtime environment: `{query_runtime_environment}` including Python, NumPy, SciPy, SfePy, and BLAS/LAPACK build fingerprint.\n"
    )
    prereg = PROJECT_ROOT / "docs/prereg/CONFIRMATORY.md"
    with prereg.open("a", encoding="utf-8") as stream:
        stream.write(amendment)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"frozen confirmation matrix: {rel}")
    print(f"matrix sha256: {matrix['sha256']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the confirmation method × seed × support × task matrix.")
    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--method-config", action="append", required=True,
        help="Selected trial ID and config path, e.g. candidate=configs/phase1_stage2_pilot.yaml.",
    )
    args = parser.parse_args()
    freeze_matrix(args.campaign, args.method_config)


if __name__ == "__main__":
    main()
