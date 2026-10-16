from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file, sha256_json
from blrpng.data.shards import read_index
from blrpng.data.confirmation_query_gate import (
    QUERY_CASE_ID_BASE,
    QUERY_CASE_MANIFEST_PATH,
    QUERY_IC_SEED_OFFSET,
    QUERY_RUNTIME_CODE_PATHS,
    QUERY_SOLVER_PROTOCOL,
    VERIFIER_PATH,
    WORKER_PATH,
)


PREREG_PATH = "docs/prereg/CONFIRMATORY.md"
CHECKPOINT_SCHEMA = "blrpng_phase1_confirmation_checkpoints_v1"


def _assert_confirmation_query_unopened(campaign: str) -> list[str]:
    """Refuse query unlock if canonical query outputs predate the gate."""
    campaign_data = PROJECT_ROOT / "artifacts/data/phase1_confirmation" / campaign
    candidates = [
        campaign_data / "validation_hr_256.index.json",
        campaign_data / "validation_hr_256",
        PROJECT_ROOT / "artifacts/tmp" / f"phase1_confirmation_{campaign}_validation_hr_256.npz",
        PROJECT_ROOT / "artifacts/tmp" / f"phase1_confirmation_{campaign}_validation_hr_256.json",
    ]
    existing = [str(path.relative_to(PROJECT_ROOT)) for path in candidates if path.exists()]
    if existing:
        raise ValueError(
            "Refusing to unlock confirmation query because pre-gate HR query outputs exist: "
            + ", ".join(existing)
        )
    return [str(path.relative_to(PROJECT_ROOT)) for path in candidates]


def _load_sealed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("sha256")
    actual = sha256_json({key: value for key, value in payload.items() if key != "sha256"})
    if not expected or expected != actual:
        raise ValueError(f"Invalid self-seal: {path}")
    return payload


def freeze_query_gate(
    campaign: str,
    matrix_manifest: str,
    checkpoint_manifest: str,
    analysis_files: list[str],
) -> Path:
    checked_query_output_paths = _assert_confirmation_query_unopened(campaign)
    task_manifest_path = PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "tasks.json"
    task_manifest = _load_sealed(task_manifest_path)
    if task_manifest.get("schema") != "blrpng_phase1_confirmation_tasks_v1":
        raise ValueError("The final query gate requires a confirmation task manifest.")
    if task_manifest.get("campaign_id") != campaign:
        raise ValueError("Campaign ID does not match the frozen task manifest.")

    matrix_path = resolve_in_project(matrix_manifest, must_exist=True)
    expected_matrix_path = (
        PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "matrix.json"
    ).resolve()
    if matrix_path != expected_matrix_path:
        raise ValueError("The final query gate must use the canonical campaign matrix manifest.")
    matrix = _load_sealed(matrix_path)
    if matrix.get("schema") != "blrpng_phase1_confirmation_matrix_v1":
        raise ValueError("Unexpected confirmation matrix schema.")
    if matrix.get("campaign_id") != campaign:
        raise ValueError("Confirmation matrix belongs to a different campaign.")
    if matrix.get("task_manifest_sha256") != task_manifest["sha256"]:
        raise ValueError("Confirmation matrix is not bound to the frozen task manifest.")
    if matrix.get("model_seeds") != [0, 1, 2, 3, 42]:
        raise ValueError("Confirmation matrix must include the five preregistered model seeds.")
    if len(matrix.get("support_seeds", [])) < 3 or len(set(matrix["support_seeds"])) != len(matrix["support_seeds"]):
        raise ValueError("Confirmation matrix requires at least three unique support seeds.")
    expected_tasks = sorted(int(row["target_task_id"]) for row in task_manifest["tasks"])
    if matrix.get("task_ids") != expected_tasks:
        raise ValueError("Confirmation matrix must include every frozen boundary task exactly once.")
    methods = matrix.get("methods")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        raise ValueError("Confirmation matrix needs a non-empty unique method list.")
    if len(methods) > 3:
        raise ValueError("The preregistered confirmation budget allows at most three methods.")
    if int(matrix.get("support_budget", 0)) != 4:
        raise ValueError("The frozen confirmation primary support budget must be m=4.")
    frozen_data = {
        "support_rollouts": 16,
        "lr_validation_rollouts": 8,
        "query_rollouts": 8,
        "lr_steps": 20,
        "query_steps": 20,
    }
    if any(int(matrix.get(key, -1)) != value for key, value in frozen_data.items()):
        raise ValueError("Confirmation data counts or rollout horizons differ from the frozen protocol.")
    if (
        matrix.get("query_case_manifest_path")
        != QUERY_CASE_MANIFEST_PATH.format(campaign=campaign)
        or int(matrix.get("query_ic_seed", -1))
        != int(task_manifest.get("seed", -1)) + QUERY_IC_SEED_OFFSET
        or int(matrix.get("query_case_id_base", -1)) != QUERY_CASE_ID_BASE
        or matrix.get("query_numpy_version") != np.__version__
        or matrix.get("query_solver_protocol") != QUERY_SOLVER_PROTOCOL
        or not isinstance(matrix.get("query_runtime_environment"), dict)
        or matrix["query_runtime_environment"].get("numpy") != matrix.get("query_numpy_version")
    ):
        raise ValueError("Confirmation query case generation or solver protocol differs from its frozen matrix.")
    method_configs = matrix.get("method_configs", {})
    if set(method_configs) != set(methods):
        raise ValueError("Confirmation matrix method configs do not match its method list.")
    for method, row in method_configs.items():
        config_path = resolve_in_project(row["config_path"], must_exist=True)
        if sha256_file(config_path) != row.get("config_sha256"):
            raise ValueError(f"Frozen confirmation config changed: {method}")

    prereg_path = resolve_in_project(PREREG_PATH, must_exist=True)
    prereg_text = prereg_path.read_text(encoding="utf-8")
    if campaign not in prereg_text:
        raise ValueError("The frozen preregistration must name this confirmation campaign.")
    matrix_rel = str(matrix_path.relative_to(PROJECT_ROOT))
    if matrix_rel not in prereg_text or matrix["sha256"] not in prereg_text:
        raise ValueError("The preregistration must name and hash the frozen matrix manifest.")

    checkpoint_manifest_path = resolve_in_project(checkpoint_manifest, must_exist=True)
    expected_checkpoint_manifest = (
        PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "checkpoints.json"
    ).resolve()
    if checkpoint_manifest_path != expected_checkpoint_manifest:
        raise ValueError("The final query gate must use the canonical campaign checkpoint manifest.")
    checkpoints = _load_sealed(checkpoint_manifest_path)
    if checkpoints.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Checkpoint manifest schema must be {CHECKPOINT_SCHEMA!r}.")
    if checkpoints.get("campaign_id") != campaign:
        raise ValueError("Checkpoint manifest belongs to a different campaign.")
    if checkpoints.get("task_manifest_sha256") != task_manifest["sha256"]:
        raise ValueError("Checkpoint manifest is not bound to the frozen task manifest.")
    rows = checkpoints.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Checkpoint manifest must contain a non-empty checkpoints list.")
    if checkpoints.get("matrix_sha256") != matrix["sha256"]:
        raise ValueError("Checkpoint manifest does not match the frozen confirmation matrix.")
    if checkpoints.get("query_opened") is not False or checkpoints.get("lr_selection_only") is not True:
        raise ValueError("Checkpoint manifest does not certify LR-only selection with unopened query data.")
    if int(checkpoints.get("checkpoint_count", -1)) != len(rows):
        raise ValueError("Checkpoint manifest row count does not match its declared count.")
    if int(checkpoints.get("unique_checkpoint_paths", -1)) != len(rows):
        raise ValueError("Confirmation matrix cells must use distinct checkpoint files.")
    if sha256_file(PROJECT_ROOT / "docs/EXPERIMENTS.csv") != checkpoints.get("experiment_ledger_sha256"):
        raise ValueError("Experiment ledger changed after the checkpoint matrix was frozen.")
    expected = {
        (method, int(model_seed), int(support_seed), int(task_id))
        for method in methods
        for model_seed in matrix["model_seeds"]
        for support_seed in matrix["support_seeds"]
        for task_id in expected_tasks
    }
    observed = set()
    verified_checkpoints = []
    checkpoint_paths = set()
    result_paths = set()
    result_cache = {}
    index_cache = {}
    task_rows = {int(item["target_task_id"]): item for item in task_manifest["tasks"]}
    for row in rows:
        key = (
            str(row["method"]),
            int(row["model_seed"]),
            int(row["support_seed"]),
            int(row["target_task_id"]),
        )
        if key in observed:
            raise ValueError(f"Duplicate confirmation checkpoint cell: {key}")
        observed.add(key)
        if key not in expected:
            raise ValueError(f"Unexpected confirmation checkpoint cell: {key}")
        if row.get("trial_config_sha256") != method_configs[key[0]].get("config_sha256"):
            raise ValueError(f"Trial config hash mismatch in confirmation cell: {key}")
        run_config_path = resolve_in_project(row["run_config_path"], must_exist=True)
        if sha256_file(run_config_path) != row.get("run_config_sha256"):
            raise ValueError(f"Run config changed after LR checkpoint selection: {run_config_path}")
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        descriptor = run_config.get("descriptor", {})
        if (
            descriptor.get("round") != "CONFIRM"
            or str(descriptor.get("trial", {}).get("id")) != key[0]
            or int(descriptor.get("model_seed", -1)) != key[1]
            or int(descriptor.get("support_seed", -1)) != key[2]
            or int(descriptor.get("support_budget", -1)) != int(matrix["support_budget"])
            or descriptor.get("trial_spec_sha256") != method_configs[key[0]].get("config_sha256")
            or descriptor.get("data_manifest_sha256") != task_manifest["sha256"]
            or run_config.get("config_sha256") != sha256_json(descriptor)
        ):
            raise ValueError(f"Run config provenance mismatch in confirmation cell: {key}")
        source_checkpoint = descriptor.get("source_checkpoint")
        source_checkpoint_sha256 = descriptor.get("source_checkpoint_sha256")
        if source_checkpoint is None:
            if source_checkpoint_sha256 is not None:
                raise ValueError(f"Unexpected source checkpoint hash in confirmation run config: {key}")
        else:
            source_path = resolve_in_project(str(source_checkpoint), must_exist=True)
            if not source_path.is_file() or sha256_file(source_path) != source_checkpoint_sha256:
                raise ValueError(f"Adapter source checkpoint hash mismatch in confirmation cell: {key}")

        result_path = resolve_in_project(row["task_result_path"], must_exist=True)
        result_rel = str(result_path.relative_to(PROJECT_ROOT))
        if result_rel in result_paths:
            raise ValueError(f"Confirmation cells cannot reuse task-result files: {result_rel}")
        result_paths.add(result_rel)
        result_sha = sha256_file(result_path)
        if result_sha != row.get("task_result_sha256"):
            raise ValueError(f"Task-result hash mismatch: {result_rel}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        target_task = task_rows[key[3]]
        if (
            int(result.get("task", -1)) != key[3]
            or str(result.get("family")) != str(target_task["family"])
            or str(result.get("trial_id")) != key[0]
            or int(result.get("model_seed", -1)) != key[1]
            or int(result.get("support_seed", -1)) != key[2]
            or int(result.get("support_budget", -1)) != int(matrix["support_budget"])
            or result.get("query_metric") is not None
            or result.get("config_sha256") != run_config.get("config_sha256")
            or result.get("descriptor_sha256") != run_config.get("config_sha256")
            or result.get("config_sha256") != row.get("config_sha256")
            or result.get("source_checkpoint") != source_checkpoint
            or result.get("source_checkpoint_sha256") != source_checkpoint_sha256
            or result.get("checkpoint") != row.get("checkpoint_path")
            or result.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or result.get("support_index") != row.get("support_index")
            or result.get("support_index_sha256") != row.get("support_index_sha256")
            or result.get("validation_index") != row.get("validation_index")
            or result.get("validation_index_sha256") != row.get("validation_index_sha256")
            or "LR validation" not in str(result.get("selection_data", ""))
        ):
            raise ValueError(f"Task-result provenance mismatch: {result_rel}")
        history = result.get("history", [])
        values = [float(item["validation"]["rollout_rel_l2"]) for item in history]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid LR validation history: {result_rel}")
        best = min(values)
        minima = [idx for idx, value in enumerate(values) if value == best]
        if (
            len(minima) != 1
            or not math.isclose(float(result.get("best_validation_metric", float("nan"))), best, rel_tol=0.0, abs_tol=1e-12)
            or int(result.get("best_epoch", -1)) != int(history[minima[0]]["epoch"])
        ):
            raise ValueError(f"Checkpoint was not selected by a unique LR-validation minimum: {result_rel}")

        for role, path_key, hash_key in (
            ("support", "support_index", "support_index_sha256"),
            ("validation_lr", "validation_index", "validation_index_sha256"),
        ):
            index_rel = str(row[path_key])
            cache_key = (index_rel, str(row[hash_key]))
            if cache_key not in index_cache:
                index_path = resolve_in_project(index_rel, must_exist=True)
                index = read_index(index_path, verify_hashes=True)
                metadata = index["metadata"]
                if (
                    index.get("sha256") != row[hash_key]
                    or metadata.get("split") != role
                    or int(metadata.get("resolution", -1)) != 64
                    or int(metadata.get("steps", -1)) != 20
                    or int(metadata.get("fidelity", -1)) != 0
                    or metadata.get("task_manifest_sha256") != task_manifest["sha256"]
                    or int(metadata.get("target_task_id", -1)) != key[3]
                    or any(
                        int(shard["target_task_id"]) != key[3]
                        or int(shard["boundary_id"]) != int(target_task["boundary_id"])
                        or str(shard.get("role")) != role
                        for shard in index["shards"]
                    )
                ):
                    raise ValueError(f"Invalid LR {role} index: {index_rel}")
                expected_count = (
                    int(matrix["support_budget"])
                    if role == "support"
                    else int(matrix["lr_validation_rollouts"])
                )
                if len(index["shards"]) != expected_count:
                    raise ValueError(f"Wrong LR {role} rollout count in {index_rel}")
                initial_conditions = [int(item["initial_condition_id"]) for item in index["shards"]]
                if len(initial_conditions) != len(set(initial_conditions)):
                    raise ValueError(f"Repeated initial-condition IDs within LR {role} index: {index_rel}")
                index_cache[cache_key] = index
            if role == "support" and int(index_cache[cache_key]["metadata"].get("support_seed", -1)) != key[2]:
                raise ValueError(f"Support seed mismatch in {index_rel}")
            if role == "validation_lr" and index_cache[cache_key]["metadata"].get("support_seed") is not None:
                raise ValueError(f"LR validation shares a support draw: {index_rel}")

        support_index = index_cache[(str(row["support_index"]), str(row["support_index_sha256"]))]
        validation_index = index_cache[(str(row["validation_index"]), str(row["validation_index_sha256"]))]
        support_ics = {int(item["initial_condition_id"]) for item in support_index["shards"]}
        validation_ics = {int(item["initial_condition_id"]) for item in validation_index["shards"]}
        if support_ics & validation_ics:
            raise ValueError(f"LR support and validation initial conditions overlap in cell {key}")

        rel = str(row["checkpoint_path"])
        path = resolve_in_project(rel, must_exist=True)
        if rel in checkpoint_paths:
            raise ValueError(f"Multiple confirmation cells reference the same checkpoint: {rel}")
        checkpoint_paths.add(rel)
        if not path.is_file():
            raise ValueError(f"Checkpoint is not a file: {rel}")
        actual = sha256_file(path)
        if actual != row.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint hash mismatch: {rel}")
        verified_checkpoints.append({
            "checkpoint_path": rel,
            "checkpoint_sha256": actual,
            "task_result_path": result_rel,
            "task_result_sha256": result_sha,
            "source_checkpoint": source_checkpoint,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "support_index": str(row["support_index"]),
            "support_index_sha256": str(row["support_index_sha256"]),
            "validation_index": str(row["validation_index"]),
            "validation_index_sha256": str(row["validation_index_sha256"]),
        })
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"Confirmation checkpoint matrix is incomplete ({len(missing)} cells missing).")

    required_analysis = {
        "scripts/evaluate_phase1_confirmation.py",
        "scripts/analyze_phase1_confirmation.py",
    }
    normalized_analysis = sorted({str(resolve_in_project(item, must_exist=True).relative_to(PROJECT_ROOT)) for item in analysis_files})
    if not required_analysis.issubset(set(normalized_analysis)):
        missing = sorted(required_analysis - set(normalized_analysis))
        raise ValueError(f"Freeze both confirmation evaluator and analysis code before query access: {missing}")
    analysis_hashes = {rel: sha256_file(PROJECT_ROOT / rel) for rel in normalized_analysis}

    generator_path = "scripts/prepare_phase1_development.py"
    freezer_path = "scripts/freeze_phase1_confirmation_query.py"
    worker_path = WORKER_PATH
    verifier_path = VERIFIER_PATH
    gate = {
        "schema": "blrpng_phase1_confirmation_query_gate_v1",
        "campaign_id": campaign,
        "task_manifest_path": str(task_manifest_path.relative_to(PROJECT_ROOT)),
        "task_manifest_sha256": task_manifest["sha256"],
        "matrix_manifest_path": matrix_rel,
        "matrix_manifest_sha256": matrix["sha256"],
        "method_configs": method_configs,
        "preregistration_path": PREREG_PATH,
        "preregistration_sha256": sha256_file(prereg_path),
        "checkpoint_manifest_path": str(checkpoint_manifest_path.relative_to(PROJECT_ROOT)),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "checkpoints": verified_checkpoints,
        "analysis_files": analysis_hashes,
        "query_generator_path": generator_path,
        "query_generator_sha256": sha256_file(PROJECT_ROOT / generator_path),
        "gate_freezer_path": freezer_path,
        "gate_freezer_sha256": sha256_file(Path(__file__).resolve()),
        "query_worker_path": worker_path,
        "query_worker_sha256": sha256_file(PROJECT_ROOT / worker_path),
        "query_gate_verifier_path": verifier_path,
        "query_gate_verifier_sha256": sha256_file(PROJECT_ROOT / verifier_path),
        "runtime_code_hashes": {
            rel: sha256_file(PROJECT_ROOT / rel) for rel in QUERY_RUNTIME_CODE_PATHS
        },
        "query_runtime_environment": matrix["query_runtime_environment"],
        "pre_unlock_query_outputs_absent": True,
        "pre_unlock_query_output_paths_checked": checked_query_output_paths,
        "query_role_unlocked": True,
        "reason": "All confirmation checkpoints, preregistration, and query analysis code are hash-frozen.",
    }
    gate["sha256"] = sha256_json(gate)
    output = ensure_artifact_path(
        PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "query_gate.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(gate, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Unlock final confirmation HR query generation after freezing all checkpoints and analysis code.")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--analysis-file", action="append", required=True)
    args = parser.parse_args()
    output = freeze_query_gate(
        args.campaign, args.matrix_manifest, args.checkpoint_manifest, args.analysis_file
    )
    print(f"confirmation HR query gate frozen: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
