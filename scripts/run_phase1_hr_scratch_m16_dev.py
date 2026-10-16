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
HR_SUPPORT_M16_INDEX_SHA256 = {
    0: "6106a9a2cb4be40806c2ed40393f1a0e4b70617ec06a3decd7cdc4e12fbe656a",
    1: "9285f576fb3d69a9c376e1edd117621435e45bdf4a41e0a11d4ef880b88948d7",
    3: "6c7cdf3065dcfccfd1349fc341bc32ca97234966aae0e5030038380b12777fa6",
    5: "11c105287192db6bbe056d1194611d7e394cda7887703f162768337fdfeef5d0",
}


def _write_json_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_indices(support_path: Path, validation_path: Path, *, task: int, boundary_id: int):
    support = read_index(support_path, verify_hashes=True)
    validation = read_index(validation_path, verify_hashes=True)
    smeta, vmeta = support["metadata"], validation["metadata"]
    if (
        support["sha256"] == ""
        or int(smeta.get("target_task_id", -1)) != task
        or smeta.get("split") != "support_hr"
        or int(smeta.get("resolution", -1)) != 256
        or int(smeta.get("steps", -1)) != 20
        or int(smeta.get("fidelity", -1)) != 1
        or int(smeta.get("support_seed", -1)) != 271829
        or smeta.get("task_manifest_sha256") != TASK_MANIFEST_SHA256
        or int(vmeta.get("target_task_id", -1)) != task
        or vmeta.get("split") != "validation_hr"
        or int(vmeta.get("resolution", -1)) != 256
        or int(vmeta.get("steps", -1)) != 20
        or int(vmeta.get("fidelity", -1)) != 1
        or vmeta.get("task_manifest_sha256") != TASK_MANIFEST_SHA256
    ):
        raise ValueError("HR m=16 support/validation index metadata changed from the frozen protocol.")
    support_rows, validation_rows = support["shards"], validation["shards"]
    if (
        len(support_rows) != 16
        or len(validation_rows) != 8
        or any(int(row["target_task_id"]) != task for row in support_rows + validation_rows)
        or any(int(row["boundary_id"]) != boundary_id for row in support_rows + validation_rows)
        or any(row.get("role") != "support_hr" for row in support_rows)
        or any(row.get("role") != "validation_hr" for row in validation_rows)
        or any(int(row.get("resolution", -1)) != 256 for row in support_rows + validation_rows)
        or any(int(row.get("steps", -1)) != 20 for row in support_rows + validation_rows)
        or any(int(row.get("support_seed", -1)) != 271829 for row in support_rows)
    ):
        raise ValueError("HR m=16 indices have unexpected rows.")
    support_ranks = sorted(int(row["support_rank"]) for row in support_rows)
    if support_ranks != list(range(16)):
        raise ValueError("HR m=16 support must preserve nested ranks 0..15.")
    support_ids = {int(row["initial_condition_id"]) for row in support_rows}
    validation_ids = {int(row["initial_condition_id"]) for row in validation_rows}
    if support_ids & validation_ids or len(support_ids) != 16 or len(validation_ids) != 8:
        raise ValueError("HR support and HR-dev validation cases overlap or repeat.")
    return support, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nested m=16 HR scratch development controls.")
    parser.add_argument("--config", default="configs/phase1_hr_scratch_m16_dev.yaml")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment, training_spec = spec["experiment"], spec["training"]
    physical_gpu = int(training_spec["device_physical_gpu"])
    if physical_gpu not in (0, 1):
        raise ValueError("Only the two assigned physical GPUs are supported.")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices != str(physical_gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES does not match the configured physical GPU.")
    lock = exclusive_artifact_lock(f"phase1_training_gpu{physical_gpu}")
    lock.__enter__()
    atexit.register(lock.__exit__, None, None, None)
    if not torch.cuda.is_available():
        raise RuntimeError("HR m=16 scratch training requires CUDA.")
    torch.cuda.set_device(0)

    if (
        int(experiment["model_seed"]) != 0
        or int(experiment["support_seed"]) != 271829
        or int(experiment["support_budget"]) != 16
        or set(experiment["tasks"]) != {0, 1, 3, 5}
        or training_spec["initialization_checkpoint"] is not None
        or int(training_spec["epochs"]) != 500
    ):
        raise ValueError("The HR m=16 scratch protocol changed from its frozen configuration.")

    task_manifest_path = PROJECT_ROOT / experiment["task_manifest"]
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    task_hash = sha256_json({key: value for key, value in task_manifest.items() if key != "sha256"})
    if task_manifest.get("sha256") != TASK_MANIFEST_SHA256 or task_hash != TASK_MANIFEST_SHA256:
        raise ValueError("Frozen HR-dev task manifest changed.")
    validation_manifest_path = PROJECT_ROOT / (
        "artifacts/manifests/phase1_development/hrdev_20261012/validation_hr.json"
    )
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    validation_manifest_hash = sha256_json(
        {key: value for key, value in validation_manifest.items() if key != "sha256"}
    )
    if (
        validation_manifest.get("sha256") != HR_VALIDATION_MANIFEST_SHA256
        or validation_manifest_hash != HR_VALIDATION_MANIFEST_SHA256
    ):
        raise ValueError("Frozen HR-dev validation manifest changed.")
    generation_path = PROJECT_ROOT / (
        "artifacts/reports/phase1_hr_scratch_data_generation_"
        "hrdev_20261015_hrsupport_m16_s271829.json"
    )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if (
        generation.get("all_converged") is not True
        or generation.get("query_used") is not False
        or int(generation.get("support_budget", -1)) != 16
        or int(generation.get("rollouts", -1)) != 64
    ):
        raise ValueError("HR m=16 support generation provenance/convergence failed.")

    task_lookup = {int(row["target_task_id"]): row for row in task_manifest["tasks"]}
    base_config_path = str(experiment["base_training_config"])
    base = load_config(base_config_path)
    if base["data"].get("representation") != "lifted":
        raise ValueError("HR scratch control must retain the lifted representation.")
    config_sha = sha256_file(config_path)
    run_namespace = "hr_scratch_from_zero_hr256_m16_s0_ss271829"
    run_dir = PROJECT_ROOT / "results/phase1/HRSCRATCHM16" / run_namespace
    run_dir.mkdir(parents=True, exist_ok=True)
    task_results_path = run_dir / "task_results.json"
    support_template = experiment["support_index_template"]
    validation_template = experiment["validation_index_template"]
    per_task_indices = {}
    for task in experiment["tasks"]:
        task = int(task)
        support = read_index(
            PROJECT_ROOT / support_template.format(task=task), verify_hashes=True
        )
        validation = read_index(
            PROJECT_ROOT / validation_template.format(task=task), verify_hashes=True
        )
        if support["sha256"] != HR_SUPPORT_M16_INDEX_SHA256[task]:
            raise ValueError(f"Frozen HR m=16 support index changed for task {task}.")
        per_task_indices[str(task)] = {
            "support": support["sha256"], "validation": validation["sha256"]
        }
    descriptor = {
        "schema": "blrpng_phase1_hr_scratch_m16_dev_run_v1",
        "config": str(Path(args.config)),
        "config_sha256": config_sha,
        "base_training_config": base_config_path,
        "base_training_config_sha256": sha256_file(PROJECT_ROOT / base_config_path),
        "task_manifest_sha256": task_hash,
        "hr_validation_manifest_sha256": validation_manifest_hash,
        "hr_support_generation_report_sha256": sha256_file(generation_path),
        "per_task_data_index_sha256": per_task_indices,
        "model_seed": 0,
        "support_seed": 271829,
        "support_budget": 16,
        "tasks": list(experiment["tasks"]),
        "initialization_checkpoint": None,
        "objective": str(training_spec["objective"]),
        "training": {key: value for key, value in training_spec.items() if key != "device_physical_gpu"},
        "training_engine_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/engine.py"),
        "adapter_module_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/adapters.py"),
        "selection_data": experiment["selection_data"],
        "query_read": False,
    }
    descriptor_sha = sha256_json(descriptor)
    run_config_path = run_dir / "config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing.get("descriptor_sha256") != descriptor_sha:
            raise FileExistsError("Existing HR m=16 run namespace has different provenance.")
    else:
        _write_json_durable(run_config_path, {
            "descriptor": descriptor, "descriptor_sha256": descriptor_sha,
        })
    task_results = {}
    if task_results_path.exists():
        existing = json.loads(task_results_path.read_text(encoding="utf-8"))
        if existing.get("descriptor_sha256") != descriptor_sha:
            raise ValueError("Existing HR m=16 results have stale configuration provenance.")
        task_results = dict(existing.get("tasks", {}))

    device = torch.device("cuda", 0)
    print(
        f"ASSIGN physical_gpu={physical_gpu} cuda_visible_devices={visible_devices} "
        f"device={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    for task in experiment["tasks"]:
        task = int(task)
        if str(task) in task_results:
            report = task_results[str(task)]
            if report.get("config_sha256") == descriptor_sha and report.get("query_read") is False:
                continue
            raise ValueError(f"Existing task {task} report failed provenance checks.")
        family = str(task_lookup[task]["family"])
        boundary_id = int(task_lookup[task]["boundary_id"])
        support_path = PROJECT_ROOT / support_template.format(task=task)
        validation_path = PROJECT_ROOT / validation_template.format(task=task)
        support_index, validation_index = _validate_indices(
            support_path, validation_path, task=task, boundary_id=boundary_id
        )
        reference_path = PROJECT_ROOT / (
            "artifacts/reports/phase1_hrdev_evaluations/"
            f"pilot_pathsource_comm256_task{task:02d}_s0_ss271829.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if (
            int(reference.get("task", -1)) != task
            or int(reference.get("model_seed", -1)) != 0
            or int(reference.get("support_seed", -1)) != 271829
            or reference.get("query_read") is not False
        ):
            raise ValueError("Matched path-source HR-dev reference failed provenance checks.")
        reference_metric = float(reference["metrics"]["rollout_rel_l2"])
        checkpoint = PROJECT_ROOT / (
            f"artifacts/checkpoints/phase1/HRSCRATCHM16/{run_namespace}/"
            f"hr_scratch_from_zero_hr256_m16_task{task:02d}_full.pt"
        )
        result_path = run_dir / f"task{task:02d}.json"
        ledger_row = {
            "trial_id": f"HRSCRATCHM16-HR256-m16-task{task:02d}-seed0-support271829",
            "status": "RUNNING",
            "round": "HRSCRATCHM16",
            "objective": "final_only",
            "seed": 0,
            "task_ids": task,
            "config_sha256": descriptor_sha,
            "raw_output_path": str(result_path),
            "notes": f"family={family}; HR256 scratch random init; nested m=16; HR-dev selection; query=false; GPU{physical_gpu}",
        }
        _append_ledger(ledger_row)
        print(f"START HR scratch m=16 task={task:02d} family={family} seed=0", flush=True)
        started = time.perf_counter()
        progress_path = PROJECT_ROOT / f"artifacts/reports/progress/{checkpoint.stem}.json"
        if progress_path.exists():
            raise FileExistsError(f"Preserving existing progress artifact: {progress_path}")
        stop = threading.Event()
        snapshots: list[dict[str, Any]] = []

        def capture() -> None:
            last_epoch = -1
            while not stop.wait(0.25):
                try:
                    item = json.loads(progress_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    continue
                epoch = int(item.get("epoch", -1))
                if epoch > last_epoch:
                    snapshots.append({
                        "epoch": epoch,
                        "elapsed_seconds": float(item["elapsed_seconds"]),
                        "validation_rollout_rel_l2": float(item["validation_rollout_rel_l2"]),
                        "best_validation_metric": float(item["best_validation_metric"]),
                    })
                    last_epoch = epoch

        thread = threading.Thread(target=capture, daemon=True)
        thread.start()
        try:
            local = json.loads(json.dumps(base))
            local["experiment"]["name"] = "phase1_hr_scratch_from_zero_hr256_m16"
            local["experiment"]["training_seed"] = 0
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
            local["_config_path"] = str(Path(args.config))
            local["_config_sha256"] = descriptor_sha
            torch.cuda.synchronize(device)
            result = train(
                local,
                seed=0,
                objective="final_only",
                initialization_checkpoint=None,
                output_checkpoint=checkpoint,
                train_data_path=support_path,
                validation_data_paths=[validation_path],
            )
            torch.cuda.synchronize(device)
            stop.set()
            thread.join()
            try:
                last = json.loads(progress_path.read_text(encoding="utf-8"))
                if not snapshots or int(last["epoch"]) > snapshots[-1]["epoch"]:
                    snapshots.append({
                        "epoch": int(last["epoch"]),
                        "elapsed_seconds": float(last["elapsed_seconds"]),
                        "validation_rollout_rel_l2": float(last["validation_rollout_rel_l2"]),
                        "best_validation_metric": float(last["best_validation_metric"]),
                    })
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
            engine_elapsed = float(
                snapshots[-1]["elapsed_seconds"] if snapshots else time.perf_counter() - started
            )
            elapsed = time.perf_counter() - started
            overhead = max(elapsed - engine_elapsed, 0.0)
            best_epoch = min(
                result["history"],
                key=lambda row: float(row["validation"]["rollout_rel_l2"]),
            )
            target = next((row for row in snapshots if row["best_validation_metric"] <= reference_metric), None)
            selected_checkpoint = compact_full_checkpoint_for_inference(result["checkpoint"])
            report = {
                "schema": "blrpng_phase1_hr_scratch_m16_dev_task_v1",
                "task": task,
                "family": family,
                "boundary_id": boundary_id,
                "trial_id": "hr_scratch_from_zero_hr256_m16",
                "objective": "final_only",
                "model_seed": 0,
                "support_seed": 271829,
                "support_budget": 16,
                "config_sha256": descriptor_sha,
                "task_manifest_sha256": task_hash,
                "base_training_config": base_config_path,
                "base_training_config_sha256": descriptor["base_training_config_sha256"],
                "source_checkpoint": None,
                "initialization": "random",
                "support_index": str(support_path.relative_to(PROJECT_ROOT)),
                "support_index_sha256": support_index["sha256"],
                "source_m16_index_sha256": HR_SUPPORT_M16_INDEX_SHA256[task],
                "validation_index": str(validation_path.relative_to(PROJECT_ROOT)),
                "validation_index_sha256": validation_index["sha256"],
                "support_rollouts": 16,
                "validation_rollouts": 8,
                "resolution": 256,
                "fidelity": 1,
                "rollout_steps": 20,
                "best_validation_metric": float(result["best_validation_metric"]),
                "best_epoch": int(best_epoch["epoch"]),
                "matched_pathsource_reference_metric": reference_metric,
                "pathsource_hrdev_reference_report": str(reference_path.relative_to(PROJECT_ROOT)),
                "pathsource_hrdev_reference_report_sha256": sha256_file(reference_path),
                "time_to_match_pathsource_hrdev_seconds": (
                    None if target is None else float(target["elapsed_seconds"]) + overhead
                ),
                "first_matching_epoch": None if target is None else int(target["epoch"]),
                "epochs_run": len(result["history"]),
                "elapsed_seconds": elapsed,
                "engine_elapsed_seconds": engine_elapsed,
                "setup_and_checkpoint_overhead_seconds": overhead,
                "gpu_hours": elapsed / 3600.0,
                "gpu": f"physical_gpu{physical_gpu}",
                "cuda_visible_devices": visible_devices,
                "runner_source_sha256": sha256_file(PROJECT_ROOT / "scripts/run_phase1_hr_scratch_m16_dev.py"),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "checkpoint": str(selected_checkpoint.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(selected_checkpoint),
                "training_or_checkpoint_selection_used_hr": True,
                "selection_data": experiment["selection_data"],
                "query_metric": None,
                "query_read": False,
                "epoch_progress_snapshots": snapshots,
                "history": result["history"],
            }
            _write_json_durable(result_path, report)
            task_results[str(task)] = report
            _append_ledger({
                **ledger_row,
                "status": "DONE",
                "elapsed_gpu_hours": report["gpu_hours"],
                "validation_metric": report["best_validation_metric"],
                "raw_output_path": str(result_path),
                "notes": f"family={family}; m=16; best_epoch={report['best_epoch']}; HR-dev selected; query=false; GPU{physical_gpu}",
            })
            _write_json_durable(task_results_path, {
                "descriptor": descriptor,
                "descriptor_sha256": descriptor_sha,
                "tasks": task_results,
            })
            print(
                f"DONE HR scratch m=16 task={task:02d} hr_val={report['best_validation_metric']:.8f} "
                f"epoch={report['best_epoch']} gpu_h={report['gpu_hours']:.5f}",
                flush=True,
            )
        except Exception as exc:
            stop.set()
            thread.join()
            elapsed = time.perf_counter() - started
            _append_ledger({
                **ledger_row,
                "status": "FAILED",
                "elapsed_gpu_hours": elapsed / 3600.0,
                "raw_output_path": str(run_dir / f"task{task:02d}.failure.txt"),
                "notes": f"{type(exc).__name__}: {exc}; physical GPU{physical_gpu}",
            })
            (run_dir / f"task{task:02d}.failure.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if all(str(int(task)) in task_results for task in experiment["tasks"]):
        values = [float(task_results[str(int(task))]["best_validation_metric"]) for task in experiment["tasks"]]
        summary = {
            "schema": "blrpng_phase1_hr_scratch_m16_dev_summary_v1",
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha,
            "tasks": list(experiment["tasks"]),
            "task_validation_metrics": {
                str(int(task)): float(task_results[str(int(task))]["best_validation_metric"])
                for task in experiment["tasks"]
            },
            "mean_hr_validation_rollout_rel_l2": sum(values) / len(values),
            "total_gpu_hours": sum(float(task_results[str(int(task))]["gpu_hours"]) for task in experiment["tasks"]),
            "support_generation_wall_seconds": float(generation["generation_wall_seconds"]),
            "support_generation_solver_seconds": float(generation["solver_seconds_sum"]),
            "query_opened_or_loaded": False,
        }
        _write_json_durable(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
