from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.config import load_config
from blrpng.data.shards import read_index
from blrpng.paths import (
    PROJECT_ROOT,
    exclusive_artifact_lock,
    sha256_file,
    sha256_json,
    write_json,
)
from blrpng.training.adapters import compact_full_checkpoint_for_inference
from blrpng.training.engine import train
from scripts.run_phase1_trials import _append_ledger


TASK_MANIFEST_SHA256 = "bb50c7e4d5b8d0900e21d7ae64235c6d955244cc3bfdca1a97ab6056596a334a"
HR_VALIDATION_MANIFEST_SHA256 = "d491452f11d9c75b62c385fd9ecf2df40b960053782e0b2c31a63a819cd21ac5"
HR_SUPPORT_M4_INDEX_SHA256 = {
    0: "cb66e634789b44cc11be7d58bd99bb442e60ffacd6362e1db0dd1c650974ef85",
    1: "2f90cc5632c5c1d6df6a798d2db88a9e094f6e52709dcaef9b4bfe513c4753e6",
    3: "efef59e67144b17f49ae7fda8784219c5ae0e4c879a9d047bdfb99ec86be576f",
    5: "3a8a35325729f3d8af76b5b6fa07c82a0fe8ae16c509f138ca86002e06f85bf2",
}


def _write_json_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_indices(
    support_path: Path,
    validation_path: Path,
    *,
    task: int,
    boundary_id: int,
    task_manifest_sha256: str,
    support_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = read_index(support_path, verify_hashes=True)
    validation = read_index(validation_path, verify_hashes=True)
    smeta, vmeta = support["metadata"], validation["metadata"]
    if (
        smeta.get("split") != "support_hr"
        or int(smeta.get("resolution", -1)) != 256
        or int(smeta.get("steps", -1)) != 20
        or int(smeta.get("fidelity", -1)) != 1
        or int(smeta.get("support_seed", -1)) != support_seed
        or smeta.get("task_manifest_sha256") != task_manifest_sha256
        or vmeta.get("split") != "validation_hr"
        or int(vmeta.get("resolution", -1)) != 256
        or int(vmeta.get("steps", -1)) != 20
        or int(vmeta.get("fidelity", -1)) != 1
        or vmeta.get("task_manifest_sha256") != task_manifest_sha256
    ):
        raise ValueError("HR scratch support/validation index metadata is not the frozen protocol.")
    support_rows = support["shards"]
    validation_rows = validation["shards"]
    if (
        len(support_rows) != 4
        or len(validation_rows) != 8
        or any(int(row["target_task_id"]) != task for row in support_rows + validation_rows)
        or any(int(row["boundary_id"]) != boundary_id for row in support_rows + validation_rows)
        or any(row.get("role") != "support_hr" for row in support_rows)
        or any(row.get("role") != "validation_hr" for row in validation_rows)
        or any(int(row.get("resolution", -1)) != 256 for row in support_rows + validation_rows)
        or any(int(row.get("steps", -1)) != 20 for row in support_rows + validation_rows)
        or any(int(row.get("support_seed", -1)) != support_seed for row in support_rows)
    ):
        raise ValueError("HR scratch indices contain unexpected rollout rows.")
    support_ids = {int(row["initial_condition_id"]) for row in support_rows}
    validation_ids = {int(row["initial_condition_id"]) for row in validation_rows}
    if support_ids & validation_ids or len(support_ids) != 4 or len(validation_ids) != 8:
        raise ValueError("HR scratch support and HR-dev validation IDs are not disjoint.")
    if any(not row.get("sha256") for row in support_rows + validation_rows):
        raise ValueError("HR scratch indices must contain verifiable rollout shards.")
    return support, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-seed HR scratch development controls.")
    parser.add_argument("--config", default="configs/phase1_hr_scratch_dev.yaml")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        choices=(0, 1),
        help="Override the physical GPU assignment for disjoint task scheduling.",
    )
    parser.add_argument(
        "--only-task",
        type=int,
        action="append",
        help="Run only selected frozen target tasks; may be repeated.",
    )
    args = parser.parse_args()

    lock_gpu = 1 if args.physical_gpu is None else int(args.physical_gpu)
    gpu_lock = exclusive_artifact_lock(f"phase1_training_gpu{lock_gpu}")
    gpu_lock.__enter__()
    atexit.register(gpu_lock.__exit__, None, None, None)
    if not torch.cuda.is_available():
        raise RuntimeError("The HR scratch control requires its assigned GPU.")
    torch.cuda.set_device(0)

    config_path = (PROJECT_ROOT / args.config).resolve()
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = spec["experiment"]
    training_spec = spec["training"]
    physical_gpu = int(
        training_spec["device_physical_gpu"]
        if args.physical_gpu is None
        else args.physical_gpu
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices != str(physical_gpu):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES={visible_devices!r} does not match physical GPU{physical_gpu}."
        )
    if (
        int(experiment["model_seed"]) != 0
        or int(experiment["support_seed"]) != 271829
        or int(experiment["support_budget"]) != 4
        or set(experiment["tasks"]) != {0, 1, 3, 5}
        or len(experiment["tasks"]) != 4
        or training_spec["initialization_checkpoint"] is not None
    ):
        raise ValueError("HR scratch config changed the frozen seed, support, tasks, or initialization.")
    execution_tasks = (
        [int(task) for task in experiment["tasks"]]
        if args.only_task is None
        else [int(task) for task in args.only_task]
    )
    frozen_tasks = {int(task) for task in experiment["tasks"]}
    if (
        not execution_tasks
        or len(execution_tasks) != len(set(execution_tasks))
        or not set(execution_tasks).issubset(frozen_tasks)
    ):
        raise ValueError("--only-task must select unique tasks from the frozen task set.")

    task_manifest_path = PROJECT_ROOT / experiment["task_manifest"]
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    task_manifest_hash = sha256_json({
        key: value for key, value in task_manifest.items() if key != "sha256"
    })
    if (
        task_manifest.get("sha256") != task_manifest_hash
        or task_manifest_hash != TASK_MANIFEST_SHA256
    ):
        raise ValueError("Frozen development task manifest changed.")
    task_lookup = {int(row["target_task_id"]): row for row in task_manifest["tasks"]}
    base_config_path = str(experiment["base_training_config"])
    base = load_config(base_config_path)
    if base["data"].get("representation") != "lifted":
        raise ValueError("The direct HR scratch control must retain the lifted representation.")
    generation_report_path = PROJECT_ROOT / (
        "artifacts/reports/phase1_hr_scratch_data_generation_"
        "hrdev_20261015_hrsupport_s271829.json"
    )
    generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
    hr_support_parent = read_index(
        PROJECT_ROOT / "artifacts/data/phase1_development/"
        "hrdev_20261015_hrsupport_s271829/support_hr_s271829_256.index.json",
        verify_hashes=True,
    )
    if (
        generation_report.get("query_used") is not False
        or generation_report.get("all_converged") is not True
        or generation_report.get("hr_support_index_sha256") != hr_support_parent["sha256"]
        or generation_report.get("hr_validation_manifest_sha256")
        != HR_VALIDATION_MANIFEST_SHA256
    ):
        raise ValueError("HR support generation report failed provenance or convergence checks.")
    per_task_data_index_sha256 = {}
    for task in experiment["tasks"]:
        task = int(task)
        support_path = PROJECT_ROOT / experiment["support_index_template"].format(task=task)
        validation_path = PROJECT_ROOT / experiment["validation_index_template"].format(task=task)
        support_index = read_index(support_path, verify_hashes=True)
        validation_index = read_index(validation_path, verify_hashes=True)
        per_task_data_index_sha256[str(task)] = {
            "support": support_index["sha256"],
            "validation": validation_index["sha256"],
            "source_m4_support": HR_SUPPORT_M4_INDEX_SHA256[task],
        }

    descriptor = {
        "schema": "blrpng_phase1_hr_scratch_dev_run_v1",
        "config": str(Path(args.config)),
        "config_sha256": sha256_file(config_path),
        "base_training_config": base_config_path,
        "base_training_config_sha256": sha256_file(PROJECT_ROOT / base_config_path),
        "task_manifest_sha256": task_manifest_hash,
        "hr_validation_manifest_sha256": HR_VALIDATION_MANIFEST_SHA256,
        "hr_support_generation_report_sha256": sha256_file(generation_report_path),
        "hr_support_parent_index_sha256": hr_support_parent["sha256"],
        "per_task_data_index_sha256": per_task_data_index_sha256,
        "model_seed": int(experiment["model_seed"]),
        "support_seed": int(experiment["support_seed"]),
        "support_budget": int(experiment["support_budget"]),
        "tasks": list(experiment["tasks"]),
        "initialization_checkpoint": None,
        "objective": str(training_spec["objective"]),
        "training": {
            key: value for key, value in training_spec.items()
            if key != "device_physical_gpu"
        },
        "training_engine_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/engine.py"),
        "adapter_module_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/adapters.py"),
        "selection_data": experiment["selection_data"],
        "query_read": False,
    }
    descriptor_sha = sha256_json(descriptor)
    run_namespace = "hr_scratch_from_zero_hr256_m4_s0_ss271829"
    run_dir = PROJECT_ROOT / "results/phase1/HRSCRATCHDEV" / run_namespace
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing.get("descriptor_sha256") != descriptor_sha or existing.get("descriptor") != descriptor:
            raise FileExistsError("Existing HR scratch run namespace has a different descriptor.")
    else:
        _write_json_durable(run_config_path, {
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha,
        })
    task_results_path = run_dir / "task_results.json"
    task_results = {}
    if task_results_path.exists():
        existing = json.loads(task_results_path.read_text(encoding="utf-8"))
        if existing.get("descriptor_sha256") != descriptor_sha:
            raise ValueError("Existing HR scratch task results have stale configuration provenance.")
        task_results = dict(existing.get("tasks", {}))

    device = torch.device("cuda", 0)
    print(
        f"ASSIGN physical_gpu={physical_gpu} cuda_visible_devices={visible_devices} "
        f"device={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    for task in execution_tasks:
        task = int(task)
        family = str(task_lookup[task]["family"])
        boundary_id = int(task_lookup[task]["boundary_id"])
        support = PROJECT_ROOT / experiment["support_index_template"].format(task=task)
        validation = PROJECT_ROOT / experiment["validation_index_template"].format(task=task)
        support_index, validation_index = _validate_indices(
            support,
            validation,
            task=task,
            boundary_id=boundary_id,
            task_manifest_sha256=task_manifest_hash,
            support_seed=int(experiment["support_seed"]),
        )
        expected_hash = HR_SUPPORT_M4_INDEX_SHA256[task]
        if not expected_hash or support_index["metadata"].get("source_m4_indices") is None:
            raise ValueError("HR index has no frozen m=4 support provenance.")
        source_map = {
            int(item["task"]): item["sha256"]
            for item in support_index["metadata"]["source_m4_indices"]
        }
        if source_map.get(task) != expected_hash:
            raise ValueError("HR support index points to a different frozen m=4 source draw.")

        reference_path = PROJECT_ROOT / (
            "artifacts/reports/phase1_hrdev_evaluations/"
            f"pilot_pathsource_comm256_task{task:02d}_s0_ss271829.json"
        )
        reference_report = json.loads(reference_path.read_text(encoding="utf-8"))
        if (
            int(reference_report.get("task", -1)) != task
            or int(reference_report.get("model_seed", -1)) != 0
            or int(reference_report.get("support_seed", -1)) != 271829
            or reference_report.get("query_read") is not False
            or reference_report.get("training_or_within_task_checkpoint_selection_used_hr") is not False
        ):
            raise ValueError("Matched path-source HR-dev reference report failed provenance checks.")
        reference_metric = float(reference_report["metrics"]["rollout_rel_l2"])

        result_path = run_dir / f"task{task:02d}.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                saved.get("config_sha256") == descriptor_sha
                and saved.get("support_index_sha256") == support_index["sha256"]
                and saved.get("validation_index_sha256") == validation_index["sha256"]
                and saved.get("checkpoint_sha256") == sha256_file(PROJECT_ROOT / saved["checkpoint"])
            ):
                task_results[str(task)] = saved
                continue
            raise ValueError(f"Existing HR scratch result failed provenance checks: {result_path}")

        local = json.loads(json.dumps(base))
        local["experiment"]["name"] = "phase1_hr_scratch_from_zero_hr256"
        local["experiment"]["training_seed"] = int(experiment["model_seed"])
        local["data"]["source_blrpng_checkpoint"] = None
        local["training"].update({
            "epochs": int(training_spec["epochs"]),
            "batch_size": int(training_spec["batch_size"]),
            "lr": float(training_spec["lr"]),
            "lr_scheduler": str(training_spec["lr_scheduler"]),
            "minimum_lr": float(training_spec["minimum_lr"]),
            "weight_decay": float(training_spec["weight_decay"]),
            "adapter": {**local["training"].get("adapter", {}), "strategy": "full"},
        })
        local["_config_path"] = str(config_path.relative_to(PROJECT_ROOT))
        local["_config_sha256"] = descriptor_sha
        checkpoint = PROJECT_ROOT / (
            f"artifacts/checkpoints/phase1/HRSCRATCHDEV/{run_namespace}/"
            f"hr_scratch_from_zero_hr256_m4_task{task:02d}_full.pt"
        )
        ledger_row = {
            "trial_id": f"HRSCRATCHDEV-hr_scratch_from_zero_hr256_m4-task{task:02d}-seed0-support271829",
            "status": "RUNNING",
            "round": "HRSCRATCHDEV",
            "objective": "final_only",
            "seed": 0,
            "task_ids": task,
            "config_sha256": descriptor_sha,
            "raw_output_path": str(result_path),
            "notes": f"family={family}; HR256 scratch from random initialization; physical GPU{physical_gpu}; HR-dev validation; no query",
        }
        _append_ledger(ledger_row)
        print(f"START HR scratch task={task:02d} family={family} seed=0", flush=True)
        started = time.perf_counter()
        progress_path = PROJECT_ROOT / f"artifacts/reports/progress/{checkpoint.stem}.json"
        if progress_path.exists():
            raise FileExistsError(f"Preserving existing HR scratch progress artifact: {progress_path}")
        progress_stop = threading.Event()
        progress_snapshots: list[dict[str, Any]] = []

        def capture_progress() -> None:
            last_epoch = -1
            while not progress_stop.wait(0.25):
                try:
                    snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    continue
                epoch = int(snapshot.get("epoch", -1))
                if epoch > last_epoch:
                    progress_snapshots.append({
                        "epoch": epoch,
                        "elapsed_seconds": float(snapshot["elapsed_seconds"]),
                        "validation_rollout_rel_l2": float(snapshot["validation_rollout_rel_l2"]),
                        "best_validation_metric": float(snapshot["best_validation_metric"]),
                    })
                    last_epoch = epoch

        progress_thread = threading.Thread(target=capture_progress, daemon=True)
        progress_thread.start()
        try:
            torch.cuda.synchronize(device)
            result = train(
                local,
                seed=int(experiment["model_seed"]),
                objective=str(training_spec["objective"]),
                initialization_checkpoint=None,
                output_checkpoint=checkpoint,
                train_data_path=support,
                validation_data_paths=[validation],
            )
            torch.cuda.synchronize(device)
            progress_stop.set()
            progress_thread.join()
            try:
                last_snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
                if not progress_snapshots or int(last_snapshot["epoch"]) > progress_snapshots[-1]["epoch"]:
                    progress_snapshots.append({
                        "epoch": int(last_snapshot["epoch"]),
                        "elapsed_seconds": float(last_snapshot["elapsed_seconds"]),
                        "validation_rollout_rel_l2": float(last_snapshot["validation_rollout_rel_l2"]),
                        "best_validation_metric": float(last_snapshot["best_validation_metric"]),
                    })
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
            train_call_wall_seconds = time.perf_counter() - started
            engine_elapsed_seconds = (
                float(progress_snapshots[-1]["elapsed_seconds"])
                if progress_snapshots else train_call_wall_seconds
            )
            setup_and_checkpoint_overhead_seconds = max(
                train_call_wall_seconds - engine_elapsed_seconds, 0.0
            )
            selected_checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            elapsed = time.perf_counter() - started
            history = result["history"]
            best_epoch = min(
                history,
                key=lambda item: float(item["validation"]["rollout_rel_l2"]),
            )
            if abs(
                float(best_epoch["validation"]["rollout_rel_l2"])
                - float(result["best_validation_metric"])
            ) > 1e-12:
                raise RuntimeError("HR scratch best checkpoint metric differs from its history.")
            target_snapshot = next((
                item for item in progress_snapshots
                if float(item["best_validation_metric"]) <= reference_metric
            ), None)
            report = {
                "schema": "blrpng_phase1_hr_scratch_dev_task_v1",
                "task": task,
                "family": family,
                "boundary_id": boundary_id,
                "trial_id": "hr_scratch_from_zero_hr256_m4",
                "objective": str(training_spec["objective"]),
                "model_seed": 0,
                "support_seed": int(experiment["support_seed"]),
                "support_budget": 4,
                "config_sha256": descriptor_sha,
                "task_manifest_sha256": task_manifest_hash,
                "base_training_config": base_config_path,
                "base_training_config_sha256": descriptor["base_training_config_sha256"],
                "source_checkpoint": None,
                "initialization": "random",
                "support_index": str(support.relative_to(PROJECT_ROOT)),
                "support_index_sha256": support_index["sha256"],
                "source_m4_index_sha256": source_map[task],
                "validation_index": str(validation.relative_to(PROJECT_ROOT)),
                "validation_index_sha256": validation_index["sha256"],
                "support_rollouts": len(support_index["shards"]),
                "validation_rollouts": len(validation_index["shards"]),
                "resolution": 256,
                "fidelity": 1,
                "rollout_steps": 20,
                "best_validation_metric": float(result["best_validation_metric"]),
                "best_epoch": int(best_epoch["epoch"]),
                "matched_pathsource_reference_metric": reference_metric,
                "pathsource_hrdev_reference_report": str(reference_path.relative_to(PROJECT_ROOT)),
                "pathsource_hrdev_reference_report_sha256": sha256_file(reference_path),
                "time_to_match_pathsource_hrdev_seconds": (
                    None if target_snapshot is None else (
                        float(target_snapshot["elapsed_seconds"])
                        + setup_and_checkpoint_overhead_seconds
                    )
                ),
                "first_matching_epoch": (
                    None if target_snapshot is None else int(target_snapshot["epoch"])
                ),
                "epochs_run": len(history),
                "elapsed_seconds": elapsed,
                "engine_elapsed_seconds": engine_elapsed_seconds,
                "setup_and_checkpoint_overhead_seconds": setup_and_checkpoint_overhead_seconds,
                "gpu_hours": elapsed / 3600.0,
                "gpu": f"physical_gpu{physical_gpu}",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "runner_source_sha256": sha256_file(PROJECT_ROOT / "scripts/run_phase1_hr_scratch_dev.py"),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "checkpoint": str(selected_checkpoint.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(selected_checkpoint),
                "training_or_checkpoint_selection_used_hr": True,
                "selection_data": experiment["selection_data"],
                "query_metric": None,
                "query_read": False,
                "epoch_progress_snapshots": progress_snapshots,
                "history": history,
            }
            _write_json_durable(result_path, report)
            task_results[str(task)] = report
            _append_ledger({
                **ledger_row,
                "status": "DONE",
                "elapsed_gpu_hours": report["gpu_hours"],
                "validation_metric": report["best_validation_metric"],
                "raw_output_path": str(result_path),
                "notes": f"family={family}; best_epoch={report['best_epoch']}; HR-dev selected; query=false; physical GPU{physical_gpu}",
            })
            print(
                f"DONE HR scratch task={task:02d} hr_val={report['best_validation_metric']:.8f} "
                f"epoch={report['best_epoch']} gpu_h={report['gpu_hours']:.5f}",
                flush=True,
            )
        except Exception as exc:
            progress_stop.set()
            progress_thread.join()
            elapsed = time.perf_counter() - started
            _append_ledger({
                **ledger_row,
                "status": "FAILED",
                "elapsed_gpu_hours": elapsed / 3600.0,
                "raw_output_path": str(run_dir / f"task{task:02d}.failure.txt"),
                "notes": f"family={family}; {type(exc).__name__}: {exc}; physical GPU{physical_gpu}",
            })
            (run_dir / f"task{task:02d}.failure.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        if args.only_task is None:
            _write_json_durable(task_results_path, {
                "descriptor": descriptor,
                "descriptor_sha256": descriptor_sha,
                "tasks": task_results,
            })

    if all(str(task) in task_results for task in experiment["tasks"]):
        values = [float(task_results[str(task)]["best_validation_metric"]) for task in experiment["tasks"]]
        summary = {
            "schema": "blrpng_phase1_hr_scratch_dev_summary_v1",
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha,
            "tasks": list(experiment["tasks"]),
            "task_validation_metrics": {
                str(task): float(task_results[str(task)]["best_validation_metric"])
                for task in experiment["tasks"]
            },
            "mean_hr_validation_rollout_rel_l2": sum(values) / len(values),
            "total_gpu_hours": sum(float(task_results[str(task)]["gpu_hours"]) for task in experiment["tasks"]),
            "query_opened_or_loaded": False,
            "selection_used_hr_development": True,
        }
        _write_json_durable(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(
            "Partial HR scratch dispatch complete; canonical summary waits for all frozen tasks.",
            flush=True,
        )


if __name__ == "__main__":
    main()
