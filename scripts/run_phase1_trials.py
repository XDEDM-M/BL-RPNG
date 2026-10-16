from __future__ import annotations

import argparse
import atexit
import csv
import fcntl
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blrpng.config import load_config
from blrpng.data.shards import read_index
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    exclusive_artifact_lock,
    sha256_file,
    sha256_json,
)
from blrpng.training.adapters import (
    compact_full_checkpoint_for_inference,
    compact_task_checkpoint,
)
from blrpng.training.engine import train


LEDGER = PROJECT_ROOT / "docs/EXPERIMENTS.csv"
TASK_FAMILIES = (
    "high_frequency", "amplitude_ood", "matern_gp", "localized_bump",
    "mixed_fourier", "piecewise_smooth",
)


def _validate_target_lr_indices(
    support_path: str,
    validation_path: str,
    *,
    task: int,
    family: str,
    boundary_id: int,
    support_seed: int,
    support_budget: int,
    task_manifest_sha256: str,
    expected_validation_rollouts: int | None = None,
) -> tuple[dict, dict]:
    support = read_index(PROJECT_ROOT / support_path, verify_hashes=True)
    validation = read_index(PROJECT_ROOT / validation_path, verify_hashes=True)
    for index, expected_role in ((support, "support"), (validation, "validation_lr")):
        metadata = index["metadata"]
        if metadata.get("split") != expected_role:
            raise ValueError(
                f"Target adaptation requires {expected_role} data; got {metadata.get('split')!r}."
            )
        if int(metadata.get("resolution", -1)) != 64 or int(metadata.get("fidelity", -1)) != 0:
            raise ValueError("Target adaptation selection inputs must be genuine 64x64 LR data.")
        if int(metadata.get("steps", -1)) != 20:
            raise ValueError("Target adaptation support and validation must use the frozen 20-step rollout.")
        if metadata.get("task_manifest_sha256") != task_manifest_sha256:
            raise ValueError("Target LR index is bound to a different frozen task manifest.")
        if int(metadata.get("target_task_id", -1)) != task:
            raise ValueError("Target LR index contains a different target task.")
        if metadata.get("evaluation_role") == "query" or metadata.get("split") == "validation_hr":
            raise PermissionError("HR query data cannot be used for target checkpoint selection.")
        rows = [row for row in index["shards"] if int(row["target_task_id"]) == task]
        if not rows or len(rows) != len(index["shards"]):
            raise ValueError("Target LR index must contain only the requested task and at least one rollout.")
        if any(
            int(row["boundary_id"]) != boundary_id
            or str(row.get("role")) != expected_role
            or int(row.get("resolution", -1)) != 64
            or int(row.get("steps", -1)) != 20
            for row in rows
        ):
            raise ValueError("Target LR shard rows disagree with the frozen task role or resolution.")
        if any(row.get("family") not in (None, family) for row in rows):
            raise ValueError("Target LR index family metadata disagrees with the frozen task.")
    support_meta = support["metadata"]
    validation_meta = validation["metadata"]
    if int(support_meta.get("support_seed", -1)) != support_seed:
        raise ValueError("Support index uses a different support-draw seed.")
    if validation_meta.get("support_seed") is not None:
        raise ValueError("LR validation must be independent of the support-draw seed.")
    if len(support["shards"]) != support_budget:
        raise ValueError("Support index rollout count does not match m.")
    if (
        expected_validation_rollouts is not None
        and len(validation["shards"]) != expected_validation_rollouts
    ):
        raise ValueError(
            f"LR validation index must contain {expected_validation_rollouts} rollouts."
        )
    for index, expected_role in ((support, "support"), (validation, "validation_lr")):
        initial_conditions = [int(row["initial_condition_id"]) for row in index["shards"]]
        if len(initial_conditions) != len(set(initial_conditions)):
            raise ValueError(f"{expected_role} index repeats an initial-condition ID.")
    support_ids = {int(row["initial_condition_id"]) for row in support["shards"]}
    validation_ids = {int(row["initial_condition_id"]) for row in validation["shards"]}
    if support_ids & validation_ids:
        raise ValueError("LR support and LR validation initial conditions must be independent.")
    return support, validation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
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
    if path.read_text(encoding="utf-8") != encoded:
        raise OSError(f"JSON artifact failed read-back verification: {path}")
    if json.loads(encoded) != payload:
        raise ValueError(f"JSON artifact did not round-trip: {path}")


def _append_ledger(row: dict) -> None:
    fields = [
        "trial_id", "status", "round", "objective", "seed", "task_ids",
        "config_sha256", "elapsed_gpu_hours", "validation_metric", "query_metric",
        "raw_output_path", "notes",
    ]
    exists = LEDGER.exists() and LEDGER.stat().st_size > 0
    with LEDGER.open("a", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _training_config(
    base: dict,
    trial: dict,
    seed: int,
    config_sha: str,
    *,
    config_path: str,
    round_name: str,
) -> dict:
    local = json.loads(json.dumps(base))
    local["experiment"]["name"] = f"phase1_{round_name.lower()}_adaptation"
    local["experiment"]["training_seed"] = int(seed)
    local["training"].update({
        "epochs": int(trial["epochs"]),
        "lr": float(trial["lr"]),
        "lr_scheduler": str(trial["scheduler"]),
        "adapter": {
            **local["training"].get("adapter", {}),
            "strategy": str(trial.get("adapter_strategy", "nlt")),
        },
    })
    for source, target in (
        ("weight_decay", "weight_decay"),
        ("path_layer", "path_layer"),
        ("path_gradient_ratio", "path_gradient_ratio"),
        ("path_anneal_gamma", "path_anneal_gamma"),
        ("path_anneal_start", "path_anneal_start"),
        ("path_anneal_end", "path_anneal_end"),
        ("auxiliary_target", "auxiliary_target"),
        ("commutator_rollout_steps", "commutator_rollout_steps"),
        ("rollout_horizon", "rollout_horizon"),
        ("rollout_curriculum", "rollout_curriculum"),
        ("calibration_resolution", "calibration_resolution"),
        ("commutator_weight", "commutator_weight"),
        ("lr_drop_epoch", "lr_drop_epoch"),
        ("lr_drop_gamma", "lr_drop_gamma"),
        ("minimum_lr", "minimum_lr"),
        ("ema_decay", "ema_decay"),
        ("gradient_clip_norm", "gradient_clip_norm"),
        ("early_stopping_patience", "early_stopping_patience"),
        ("minimum_epochs", "minimum_epochs"),
        ("validation_batch_size", "validation_batch_size"),
    ):
        if source in trial:
            local["training"][target] = trial[source]
    local["_config_path"] = config_path
    local["_config_sha256"] = config_sha
    return local


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase-1 LR-validation trials.")
    parser.add_argument("--config", default="configs/phase1_r1.yaml")
    parser.add_argument("--trial", action="append", help="Trial ID; repeat to select a subset.")
    parser.add_argument(
        "--trial-shard", nargs=2, type=int, metavar=("INDEX", "COUNT"),
        help="Run this zero-based modulo shard of the configured trials.",
    )
    parser.add_argument("--task", action="append", type=int, help="Task ID; repeat to select a subset.")
    parser.add_argument("--seed", type=int, help="Override the frozen screen model seed.")
    parser.add_argument("--support-seed", type=int, help="Override support redraw seed, if the manifest provides one.")
    parser.add_argument("--budget", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--smoke", action="store_true", help="Run one 2-epoch non-ranking pipeline smoke.")
    parser.add_argument("--attempt-suffix", help="Fresh immutable namespace for an intentional new-code retry.")
    parser.add_argument("--force", action="store_true", help="Rerun completed task configurations.")
    args = parser.parse_args()
    # Serialize all Phase-1 training runners across Codex processes/worktrees.
    # GPU selection and dispatch remain the caller's responsibility; this lock
    # makes accidental concurrent use of the campaign GPU fail closed.
    training_lock = exclusive_artifact_lock("phase1_training_gpu0")
    training_lock.__enter__()
    atexit.register(training_lock.__exit__, None, None, None)
    if args.attempt_suffix and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.attempt_suffix) is None:
        raise ValueError("Attempt suffix must use lowercase letters, digits, '_' or '-'.")

    spec_path = (PROJECT_ROOT / args.config).resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    experiment = spec["experiment"]
    round_name = str(experiment.get("round", "R1"))
    # Research stage names such as S2PILOT are valid campaign namespaces too;
    # keep the namespace constrained while allowing the frozen Stage-2 card.
    if re.fullmatch(r"(?:R[0-9][A-Z0-9]*|S2PILOT|CONFIRM)", round_name) is None:
        raise ValueError(f"Invalid phase round name: {round_name!r}")
    seed = int(args.seed if args.seed is not None else experiment["screen_model_seed"])
    budget = int(args.budget if args.budget is not None else experiment["support_budget"])
    support_seed = int(
        args.support_seed
        if args.support_seed is not None
        else experiment.get("support_seed", 0)
    )
    task_ids = list(args.task) if args.task else [int(value) for value in experiment["tasks"]]
    data_root = str(experiment.get("data_root", "artifacts/data/reconfirmation"))
    support_template = str(experiment.get(
        "support_index_template",
        "{root}/task{task:02d}_support_64_m{budget}.index.json",
    ))
    validation_template = str(experiment.get(
        "validation_index_template",
        "{root}/task{task:02d}_validation_64_all.index.json",
    ))
    selection_data = str(experiment.get(
        "selection_data", "64x64 target LR validation only; no HR query loaded"
    ))
    trials = spec["trials"]
    if args.trial and args.trial_shard:
        raise ValueError("Use --trial or --trial-shard, not both.")
    if args.trial:
        wanted = set(args.trial)
        trials = [item for item in trials if item["id"] in wanted]
        missing = wanted - {item["id"] for item in trials}
        if missing:
            raise ValueError(f"Unknown trial ID(s): {sorted(missing)}")
    if args.trial_shard:
        shard_index, shard_count = args.trial_shard
        if shard_count <= 0 or not 0 <= shard_index < shard_count:
            raise ValueError("--trial-shard requires 0 <= INDEX < positive COUNT.")
        trials = [
            item for index, item in enumerate(trials)
            if index % shard_count == shard_index
        ]
    if args.smoke:
        trials = trials[:1]
        task_ids = task_ids[:1]
    if not trials or not task_ids:
        raise ValueError("At least one trial and one task are required.")

    base = load_config(experiment["base_training_config"])
    manifest_path = PROJECT_ROOT / experiment["data_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sha256" in manifest:
        unsigned_manifest = {key: value for key, value in manifest.items() if key != "sha256"}
        if manifest["sha256"] != sha256_json(unsigned_manifest):
            raise ValueError("Task manifest hash mismatch.")
    # The immutable manifest records that query was unopened at freeze time;
    # historical reports show that it was opened later. This runner constructs
    # only LR support/validation paths and never imports the HR evaluation code.
    task_lookup = {int(row["target_task_id"]): row for row in manifest["tasks"]}
    unknown = sorted(set(task_ids) - set(task_lookup))
    if unknown:
        raise ValueError(f"Unknown target task IDs: {unknown}")

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    spec_sha = sha256_file(spec_path)
    for trial in trials:
        trial = dict(trial)
        trial_id = str(trial["id"])
        if args.attempt_suffix:
            trial_id += f"-{args.attempt_suffix}"
        # Keep the initialization path explicit per trial. This supports a
        # same-budget ordinary-transfer baseline as well as the path source
        # candidate promoted after R1 source validation.
        checkpoint_template = trial.get("initialization_checkpoint_template")
        if checkpoint_template is not None:
            configured_checkpoint = str(checkpoint_template).format(seed=seed)
        elif "initialization_checkpoint" in trial:
            configured_checkpoint = trial["initialization_checkpoint"]
        elif experiment.get("checkpoint_template") is not None:
            configured_checkpoint = str(experiment["checkpoint_template"]).format(seed=seed)
        else:
            configured_checkpoint = experiment["checkpoint"]
        source_checkpoint = str(configured_checkpoint) if configured_checkpoint else None
        source_sha = (
            sha256_file(PROJECT_ROOT / source_checkpoint) if source_checkpoint else None
        )
        if args.smoke:
            trial["epochs"] = 2
            trial_id += "-smoke"
        descriptor = {
            "round": round_name,
            "attempt_suffix": args.attempt_suffix,
            "trial": trial,
            "model_seed": seed,
            "support_budget": budget,
            "support_seed": support_seed,
            "task_ids": task_ids,
            "trial_spec_sha256": spec_sha,
            "source_checkpoint": source_checkpoint,
            "source_checkpoint_sha256": source_sha,
            "data_manifest": experiment["data_manifest"],
            "base_training_config": experiment["base_training_config"],
            "base_training_config_sha256": sha256_file(PROJECT_ROOT / experiment["base_training_config"]),
            "data_manifest_sha256": sha256_file(manifest_path),
            "data_root": data_root,
            "support_index_template": support_template,
            "validation_index_template": validation_template,
            "selection_data": selection_data,
            "smoke_only": bool(args.smoke),
            "git_commit": git_commit,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "rollout_windows_sha256": sha256_file(PROJECT_ROOT / "blrpng/data/rollout_windows.py"),
            "rollout_supervision_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/rollout_supervision.py"),
            "engine_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/engine.py"),
            "adapter_module_sha256": sha256_file(PROJECT_ROOT / "blrpng/training/adapters.py"),
        }
        config_sha = sha256_json(descriptor)
        run_namespace = f"{trial_id}_m{budget}_s{seed}"
        if support_seed:
            run_namespace += f"_ss{support_seed}"
        run_dir = PROJECT_ROOT / "results" / "phase1" / round_name / run_namespace
        config_path = run_dir / "config.json"
        results_path = run_dir / "task_results.json"
        completed = {}
        if not args.force:
            if config_path.exists():
                try:
                    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Unreadable existing run config: {config_path}") from exc
                if (
                    saved_config.get("config_sha256") != config_sha
                    or saved_config.get("descriptor") != descriptor
                ):
                    raise ValueError(
                        f"Existing run config does not match the requested descriptor: {run_dir}. "
                        "Use a new namespace or --force to replace it."
                    )
            if results_path.exists():
                try:
                    completed = json.loads(results_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Unreadable existing task results: {results_path}") from exc
                if (
                    completed.get("config_sha256") != config_sha
                    or completed.get("descriptor") != descriptor
                ):
                    raise ValueError(
                        f"Existing task results do not match the requested descriptor: {results_path}. "
                        "Refusing stale-result reuse."
                    )
        task_results = dict(completed.get("tasks", {}))
        for task_key, saved in task_results.items():
            try:
                task = int(task_key)
                expected_support = support_template.format(
                    root=data_root, task=task, budget=budget,
                    support_seed=support_seed,
                )
                expected_validation = validation_template.format(
                    root=data_root, task=task, budget=budget,
                    support_seed=support_seed,
                )
                valid_saved = (
                    task in task_ids
                    and int(saved.get("task", -1)) == task
                    and str(saved.get("trial_id")) == trial_id
                    and int(saved.get("model_seed", -1)) == seed
                    and int(saved.get("support_budget", -1)) == budget
                    and int(saved.get("support_seed", -1)) == support_seed
                    and saved.get("config_sha256") == config_sha
                    and saved.get("support_index") == expected_support
                    and saved.get("validation_index") == expected_validation
                    and saved.get("support_index_sha256") == sha256_file(PROJECT_ROOT / expected_support)
                    and saved.get("validation_index_sha256") == sha256_file(PROJECT_ROOT / expected_validation)
                    and saved.get("source_checkpoint_sha256") == source_sha
                    and saved.get("descriptor_sha256") == config_sha
                    and saved.get("checkpoint")
                    and saved.get("checkpoint_sha256") == sha256_file(PROJECT_ROOT / saved["checkpoint"])
                )
            except (KeyError, OSError, TypeError, ValueError):
                valid_saved = False
            if not valid_saved:
                raise ValueError(
                    f"Existing task result {task_key} is stale or fails provenance checks in {results_path}. "
                    "Refusing reuse; use --force only for an intentional full rerun."
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(config_path, {
            "descriptor": descriptor,
            "config_sha256": config_sha,
        })
        for task in task_ids:
            task_key = str(task)
            if task_key in task_results and not args.force:
                continue
            family = str(task_lookup[task]["family"])
            support = support_template.format(
                root=data_root, task=task, budget=budget,
                support_seed=support_seed,
            )
            validation = validation_template.format(
                root=data_root, task=task, budget=budget,
                support_seed=support_seed,
            )
            target_task = task_lookup[task]
            support_index, validation_index = _validate_target_lr_indices(
                support,
                validation,
                task=task,
                family=str(target_task["family"]),
                boundary_id=int(target_task["boundary_id"]),
                support_seed=support_seed,
                support_budget=budget,
                task_manifest_sha256=str(manifest.get("sha256", "")),
                expected_validation_rollouts=(
                    8 if round_name in {"S2PILOT", "CONFIRM"} else None
                ),
            )
            stem = f"phase1_{run_namespace}_task{task:02d}"
            checkpoint = f"artifacts/checkpoints/phase1/{round_name}/{run_namespace}/{stem}.pt"
            local = _training_config(
                base,
                trial,
                seed,
                config_sha,
                config_path=args.config,
                round_name=round_name,
            )
            if "validation_batch_size" in experiment:
                local["training"]["validation_batch_size"] = int(
                    experiment["validation_batch_size"]
                )
            objective = str(trial["objective"])
            initialization_key = "source_blrpng_checkpoint"
            local["data"][initialization_key] = source_checkpoint
            started = time.perf_counter()
            started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            row = {
                "trial_id": f"{round_name}-{trial_id}-task{task:02d}-seed{seed}-m{budget}-support{support_seed}",
                "status": "RUNNING",
                "round": round_name,
                "objective": objective,
                "seed": seed,
                "task_ids": task,
                "config_sha256": config_sha,
                "raw_output_path": str(run_dir / f"task{task:02d}.json"),
                "notes": f"family={family}; selection={selection_data}; started={started_iso}",
            }
            _append_ledger(row)
            print(f"START trial={trial_id} task={task:02d} family={family} seed={seed}", flush=True)
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with torch.cuda.device(torch.cuda.current_device()) if torch.cuda.is_available() else _null_context():
                    result = train(
                        local,
                        seed=seed,
                        objective=objective,
                        initialization_checkpoint=source_checkpoint,
                        output_checkpoint=checkpoint,
                        train_data_path=support,
                        validation_data_paths=[validation],
                    )
                if local["training"]["adapter"]["strategy"] in {
                    "lora", "nlt", "nlt_mix", "nlt_lora"
                }:
                    if not source_checkpoint:
                        raise ValueError("Adapter transfer trials require an initialization checkpoint.")
                    compact = compact_task_checkpoint(
                        result["checkpoint"],
                        base_checkpoint=source_checkpoint,
                    )
                elif local["training"]["adapter"]["strategy"] == "full":
                    compact = compact_full_checkpoint_for_inference(result["checkpoint"])
                else:
                    raise ValueError(
                        "R1 runner supports NLT transfer adapters or full-model scratch only."
                    )
                elapsed = time.perf_counter() - started
                history = result["history"]
                best_epoch = min(
                    history,
                    key=lambda item: float(item["validation"]["rollout_rel_l2"]),
                )
                task_result = {
                    "task": task,
                    "family": family,
                    "trial_id": trial_id,
                    "objective": objective,
                    "model_seed": seed,
                    "support_budget": budget,
                    "support_seed": support_seed,
                    "config_sha256": config_sha,
                    "descriptor_sha256": config_sha,
                    "base_training_config": descriptor["base_training_config"],
                    "base_training_config_sha256": descriptor["base_training_config_sha256"],
                    "source_checkpoint": source_checkpoint,
                    "source_checkpoint_sha256": source_sha,
                    "support_index": support,
                    "support_index_sha256": support_index["sha256"],
                    "validation_index": validation,
                    "validation_index_sha256": validation_index["sha256"],
                    "best_validation_metric": float(result["best_validation_metric"]),
                    "best_epoch": int(best_epoch["epoch"]),
                    "epochs_run": len(history),
                    "elapsed_seconds": elapsed,
                    "gpu_hours": elapsed / 3600.0 if torch.cuda.is_available() else 0.0,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "checkpoint": str(compact.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256_file(compact),
                    "selection_data": selection_data,
                    "query_metric": None,
                    "history": history,
                }
                _write_json(run_dir / f"task{task:02d}.json", task_result)
                task_results[task_key] = task_result
                _append_ledger({
                    **row,
                    "status": "DONE",
                    "elapsed_gpu_hours": task_result["gpu_hours"],
                    "validation_metric": task_result["best_validation_metric"],
                    "raw_output_path": str(run_dir / f"task{task:02d}.json"),
                    "notes": f"family={family}; best_epoch={task_result['best_epoch']}; no query accessed",
                })
                print(
                    f"DONE trial={trial_id} task={task:02d} "
                    f"lr_val={task_result['best_validation_metric']:.8f} "
                    f"epoch={task_result['best_epoch']} sec={elapsed:.1f}",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                _append_ledger({
                    **row,
                    "status": "FAILED",
                    "elapsed_gpu_hours": elapsed / 3600.0,
                    "raw_output_path": str(run_dir / f"task{task:02d}.failure.txt"),
                    "notes": f"family={family}; {type(exc).__name__}: {exc}",
                })
                failure = run_dir / f"task{task:02d}.failure.txt"
                failure.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                _write_json(results_path, {
                    "descriptor": descriptor,
                    "config_sha256": config_sha,
                    "tasks": task_results,
                })
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _write_json(results_path, {
                "descriptor": descriptor,
                "config_sha256": config_sha,
                "tasks": task_results,
            })

        values = [float(task_results[str(task)]["best_validation_metric"]) for task in task_ids]
        summary = {
            "descriptor": descriptor,
            "config_sha256": config_sha,
            "trial_id": trial_id,
            "task_count": len(values),
            "mean_task_lr_validation_rollout_rel_l2": sum(values) / len(values),
            "worst_task_lr_validation_rollout_rel_l2": max(values),
            "task_validation_metrics": {
                str(task): float(task_results[str(task)]["best_validation_metric"])
                for task in task_ids
            },
            "selection_data": selection_data,
            "query_opened_or_loaded": False,
        }
        _write_json(run_dir / "summary.json", summary)
        print(
            f"TRIAL_COMPLETE {trial_id} mean_lr_val="
            f"{summary['mean_task_lr_validation_rollout_rel_l2']:.8f}",
            flush=True,
        )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    main()
