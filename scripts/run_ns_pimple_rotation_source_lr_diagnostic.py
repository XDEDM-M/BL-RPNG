#!/usr/bin/env python3
"""Rotation-augmented NS source pretraining with LR-only target selection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import yaml

import run_ns_pimple_method_v1 as method
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.training.engine import _train_batch
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256


ROOT = method.ROOT
PREREG = ROOT / "docs/prereg/ns_pimple_method_v1_rotation_source_lr_diagnostic.json"
REPORT_PATH = method.REPORT_ROOT / "rotation_source_lr_diagnostic.json"
OUTPUT_ROOT = "artifacts/checkpoints/ns_pimple_method_v1/rotation_source_lr_diagnostic"
SOURCE_ARMS = method.SOURCE_ARMS


def _rotate_vector_fields(value: torch.Tensor, pairs: list[tuple[int, int]], k: int) -> None:
    for first, second in pairs:
        a = value[:, first].clone()
        b = value[:, second].clone()
        if k == 1:  # physical clockwise rotation
            value[:, first], value[:, second] = b, -a
        elif k == 2:
            value[:, first], value[:, second] = -a, -b
        elif k == 3:  # physical counterclockwise rotation
            value[:, first], value[:, second] = -b, a


def rotate_grouped_batch(batch: dict[str, torch.Tensor], rotations: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rotate each transition while preserving vector components and coordinates."""
    output = {
        key: value.clone() if key == "field_scales" else torch.empty_like(value)
        for key, value in batch.items()
    }
    for k in range(4):
        selected = torch.nonzero(rotations == k, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        pairs: dict[str, list[tuple[int, int]]] = {
            "x": [(0, 1), (4, 5), (6, 7), (8, 9)],
            "y": [(0, 1)],
            "boundary_lift": [(0, 1)],
            "newton_correction": [(0, 1)],
            "endpoint_increment": [(0, 1)],
        }
        rotated: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key == "field_scales":
                continue
            rows = value.index_select(0, selected)
            if k and rows.ndim >= 4:
                rows = torch.rot90(rows, k=k, dims=(-2, -1)).contiguous()
                if key in pairs:
                    rows = rows.clone()
                    _rotate_vector_fields(rows, pairs[key], k)
                if key == "x":
                    rows = rows.clone()
                    x_old = rows[:, 2].clone()
                    y_old = rows[:, 3].clone()
                    if k == 1:
                        rows[:, 2], rows[:, 3] = y_old, 1.0 - x_old
                    elif k == 2:
                        rows[:, 2], rows[:, 3] = 1.0 - x_old, 1.0 - y_old
                    else:
                        rows[:, 2], rows[:, 3] = 1.0 - y_old, x_old
            rotated[key] = rows
        for key, rows in rotated.items():
            output[key].index_copy_(0, selected, rows)
    return output


def _source_objective(arm: str) -> tuple[str, str | None]:
    return {
        "selected_pimple_path": ("blrpng_safe", "newton_correction"),
        "final_only": ("final_only", None),
        "matched_endpoint_control": ("blrpng_safe", "endpoint_direction_path_norm"),
    }[arm]


def _training_config(base_config: dict, auxiliary: str | None) -> dict:
    training = dict(base_config["training"])
    training.update({
        "final_loss": "joint_relative_l2",
        "rollout_horizon": 1,
        "vorticity_weight": 0.0,
        "divergence_weight": 0.0,
        "path_layer": 1,
        "path_gradient_ratio": 0.05,
        "path_log_magnitude_weight": 0.1,
        "path_epsilon": 1e-8,
    })
    if auxiliary:
        training["auxiliary_target"] = auxiliary
    return training


def _batch(data: dict[str, torch.Tensor], indices: list[int], device: torch.device,
           trajectory_rotations: torch.Tensor) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    batch = {
        key: value if key == "field_scales" else value.index_select(0, index)
        for key, value in data.items()
    }
    rotations = trajectory_rotations.index_select(0, torch.div(index, 20, rounding_mode="floor"))
    rotated = rotate_grouped_batch(batch, rotations)
    return {key: value.to(device, non_blocking=True) for key, value in rotated.items()}


def train_source_arm(arm: str, data: dict[str, torch.Tensor], validation_cases: list[dict],
                     model_config: dict, base_config: dict, *, device: torch.device) -> dict:
    set_seed(42, deterministic=True)
    count = int(data["x"].shape[0])
    if count != 560 or count % 20:
        raise ValueError(f"expected 28 ordered 20-step source trajectories, found {count}")
    trajectories = count // 20
    model = method.TimeDependentLiftedOperator(
        method.build_model(model_config, representation="lifted")
    ).to(device)
    objective, auxiliary = _source_objective(arm)
    training = _training_config(base_config, auxiliary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    batch_size, max_epochs, min_epochs, patience = 8, 30, 30, 12
    steps_per_epoch = math.ceil(count / batch_size)
    order_generator = torch.Generator(device="cpu").manual_seed(271828)
    rotation_generator = torch.Generator(device="cpu").manual_seed(271829)
    best_score, best_epoch, stale = float("inf"), 0, 0
    history: list[dict] = []
    checkpoint_path = ensure_artifact_path(f"{OUTPUT_ROOT}/source_{arm}_seed42.pt")
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite rotation source checkpoint: {checkpoint_path}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(count, generator=order_generator)
        trajectory_rotations = torch.randint(0, 4, (trajectories,), generator=rotation_generator)
        losses, last_metrics = [], {}
        for start in range(0, count, batch_size):
            indices = permutation[start:start + batch_size].tolist()
            batch = _batch(data, indices, device, trajectory_rotations)
            optimizer.zero_grad(set_to_none=True)
            loss, last_metrics = _train_batch(
                model=model, batch=batch, objective=objective, training_config=training,
                epoch_progress=(epoch - 1) / max(max_epochs - 1, 1),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite rotated-source loss: {arm}/{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_score, validation_rows = method.mean_rollout_error(model, validation_cases, device)
        row = {
            "epoch": epoch,
            "optimizer_updates": epoch * steps_per_epoch,
            "train_loss_mean": float(np.mean(losses)),
            "source_validation_rollout_rel_l2": validation_score,
            "rotations_this_epoch": torch.bincount(trajectory_rotations, minlength=4).tolist(),
            "path_loss_last_batch": float(last_metrics.get("path_loss", 0.0)),
            "path_gradient_ratio_last_batch": float(last_metrics.get("gradient_ratio", 0.0)),
        }
        history.append(row)
        if validation_score < best_score:
            best_score, best_epoch, stale = validation_score, epoch, 0
            digest = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_rotation_source_checkpoint_v1",
                "arm": arm, "seed": 42, "epoch": epoch,
                "base_state_dict": model.base.state_dict(), "model_config": model_config,
                "model_state_sha256": digest,
                "rotation_augmentation": "trajectory-consistent random 0/90/180/270 degree rotations",
                "rotation_seed": 271829,
                "source_validation_only_selection": True,
                "target_hr_query_used": False,
            }, checkpoint_path)
            best_rows = validation_rows
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"rot-source {arm} epoch={epoch} train={row['train_loss_mean']:.5g} "
                  f"source-val={validation_score:.5g} best={best_score:.5g}@{best_epoch}", flush=True)
        if epoch >= min_epochs and stale >= patience:
            break
    return {
        "arm": arm, "best_epoch": best_epoch, "epochs_run": len(history),
        "optimizer_updates": len(history) * steps_per_epoch,
        "best_source_validation_rollout_rel_l2": best_score,
        "validation_per_rollout": best_rows,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_state_sha256": digest,
        "target_hr_query_used": False,
        "training_history": history,
    }


def train_target(task: str, arm: str, source: dict | None, support: dict[str, torch.Tensor],
                 validation_cases: list[dict], model_config: dict, *, device: torch.device) -> dict:
    set_seed(42 + method.TASKS.index(task), deterministic=True)
    base = method.build_model(model_config, representation="lifted")
    if arm == "lifted_lr_scratch":
        adapter_name = "none_scratch"
        adapter_info = {"trainable_parameters": sum(parameter.numel() for parameter in base.parameters())}
        learning_rate, weight_decay = 3e-4, 1e-4
        max_epochs, min_epochs, patience, validation_interval = 500, 50, 25, 5
        source_hash = None
    else:
        base.load_state_dict(source["base_state_dict"], strict=True)
        adapter_info = base.configure_adaptation("nlt", rank=4, alpha=4.0)
        adapter_name = "nlt"
        learning_rate, weight_decay = 1e-3, 1e-6
        max_epochs, min_epochs, patience, validation_interval = 100, 30, 12, 1
        source_hash = source["model_state_sha256"]
    model = method.TimeDependentLiftedOperator(base).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    count, batch_size = int(support["x"].shape[0]), 4
    steps_per_epoch = math.ceil(count / batch_size)
    generator = torch.Generator(device="cpu").manual_seed(271828)
    best_score, best_epoch, stale = float("inf"), 0, 0
    history = []
    checkpoint_path = ensure_artifact_path(
        f"{OUTPUT_ROOT}/target_{task}_{arm}_seed42.pt"
    )
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite rotation target checkpoint: {checkpoint_path}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(count, generator=generator)
        losses = []
        for start in range(0, count, batch_size):
            ids = permutation[start:start + batch_size].tolist()
            batch = method._batch(support, ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _train_batch(
                model=model, batch=batch, objective="final_only",
                training_config={"final_loss": "joint_relative_l2", "rollout_horizon": 1},
                epoch_progress=1.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % validation_interval and epoch != max_epochs:
            continue
        model.eval()
        score, validation_rows = method.mean_rollout_error(model, validation_cases, device)
        history.append({"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
                        "lr_validation_rollout_rel_l2": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            digest = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_rotation_target_checkpoint_v1",
                "task": task, "source_arm": arm, "epoch": epoch, "adapter": adapter_name,
                "base_state_dict": model.base.state_dict(), "model_config": model_config,
                "model_state_sha256": digest,
                "source_checkpoint_sha256": source_hash,
                "selection_data": "LR validation only", "target_hr_query_used": False,
            }, checkpoint_path)
            best_rows = validation_rows
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"rot-target {task} {arm} epoch={epoch} LR-val={score:.5g} "
                  f"best={best_score:.5g}@{best_epoch}", flush=True)
        if epoch >= min_epochs and stale >= patience:
            break
    return {
        "task": task, "source_arm": arm, "target_objective": "final_only",
        "adapter": adapter_name,
        "best_epoch": best_epoch, "epochs_run": history[-1]["epoch"],
        "optimizer_updates": history[-1]["epoch"] * steps_per_epoch,
        "best_lr_validation_rollout_rel_l2": best_score,
        "validation_per_rollout": best_rows,
        "trainable_parameters": int(adapter_info["trainable_parameters"]),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "target_hr_query_used": False, "training_history": history,
    }


def prepare_validation(model_config: dict) -> tuple[dict[str, list[dict]], dict[str, list[str]], dict]:
    """Combine original LR validation and fresh V2 LR validation; never load V2 HR query fields."""
    original = {
        task: method.load_torch(method.DATA_ROOT / f"{task}_validation_rollouts.pt")
        for task in method.TASKS
    }
    validation_ids = {
        task: list(method.load_json(method.DATA_ROOT / "dataset_index.json")["target_tasks"][task][
            "validation_pair_ids"
        ]) for task in method.TASKS
    }
    v2_manifest_path = method.NS_ROOT / "data/method_v2/manifest.json"
    if not v2_manifest_path.is_file():
        raise FileNotFoundError("fresh method V2 LR validation data must be generated before rotation training")
    v2 = method.load_json(v2_manifest_path)
    dataset = method.NSPimpleDataset(method.NS_ROOT)
    fresh_counts = {task: 0 for task in method.TASKS}
    for generated in v2["trajectories"]:
        if generated.get("role") != "validation" or generated.get("level") != "lr":
            continue
        task = next((name for name in method.TASKS if generated["pair_id"].startswith(name + "_")), None)
        if task is None:
            continue
        row = {
            "pair_id": generated["pair_id"], "task_id": generated["task_id"],
            "family_id": generated["family_id"], "split": generated["split"],
            "role": generated["role"], "status": "complete", "training_access": "none",
            "cases": {"lr": {"hdf5": generated["hdf5"], "status": "complete",
                              "hdf5_sha256": generated["hdf5_sha256"]}},
        }
        dataset.by_id[row["pair_id"]] = row
        case = method.prepare_case(dataset, row, "lr", training=False)
        original[task].append(method.rollout_payload(case))
        validation_ids[task].append(row["pair_id"])
        fresh_counts[task] += 1
    if any(count != 2 for count in fresh_counts.values()):
        raise RuntimeError(f"expected exactly two fresh V2 LR validation cases per task: {fresh_counts}")
    return original, validation_ids, v2


def evaluate_original_v1(task: str, arm: str, validation_cases: list[dict], model_config: dict,
                         *, device: torch.device) -> float:
    path = method.CHECKPOINT_ROOT / f"target_{task}_{arm}_seed42.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("query_used") is not False:
        raise RuntimeError(f"V1 target checkpoint audit failed for {task}/{arm}")
    base = method.build_model(model_config, representation="lifted")
    if payload.get("adapter") == "nlt":
        base.configure_adaptation("nlt", rank=4, alpha=4.0)
    base.load_state_dict(payload["base_state_dict"], strict=True)
    model = method.TimeDependentLiftedOperator(base).to(device)
    model.eval()
    score, _ = method.mean_rollout_error(model, validation_cases, device)
    return score


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device is unavailable: {device}")
    model_base = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = method.build_model_config(model_base)
    source_data = method.load_torch(method.DATA_ROOT / "source_path_pretraining.pt")
    source_validation = method.load_torch(method.DATA_ROOT / "source_validation_rollouts.pt")
    support = {task: method.load_torch(method.DATA_ROOT / f"{task}_support_final_only.pt")
               for task in method.TASKS}
    target_validation, validation_ids, v2_manifest = prepare_validation(model_config)
    source_path = method.REPORT_ROOT / "source_training.json"
    report = {
        "schema": "blrpng_ns_pimple_rotation_source_lr_diagnostic_v1",
        "status": "exploratory_lr_only_followup",
        "device": str(device), "target_hr_query_used": False, "sealed_hr_claim": False,
        "preregistration_sha256": sha256_file(PREREG),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "method_runner_sha256": sha256_file(ROOT / "scripts/run_ns_pimple_method_v1.py"),
        "engine_sha256": sha256_file(ROOT / "blrpng/training/engine.py"),
        "dataset_index_sha256": sha256_file(method.DATA_ROOT / "dataset_index.json"),
        "method_v2_validation_manifest_sha256": sha256_file(method.NS_ROOT / "data/method_v2/manifest.json"),
        "method_v2_generator_sha256": sha256_file(method.NS_ROOT / "scripts/generate_method_v2_confirmation.py"),
        "target_validation_pair_ids": validation_ids,
        "method_v2_manifest_dataset_id": v2_manifest["dataset_id"],
        "source_data_sha256": sha256_file(method.DATA_ROOT / "source_path_pretraining.pt"),
        "source_validation_sha256": sha256_file(method.DATA_ROOT / "source_validation_rollouts.pt"),
        "original_source_checkpoint_sha256": {
            arm: record["checkpoint_sha256"]
            for arm, record in method.load_json(source_path)["arms"].items()
        },
        "source_arms": {}, "target_tasks": {},
    }
    if REPORT_PATH.exists():
        prior = method.load_json(REPORT_PATH)
        if (prior.get("schema") != report["schema"]
                or prior.get("preregistration_sha256") != report["preregistration_sha256"]
                or prior.get("source_data_sha256") != report["source_data_sha256"]):
            raise ValueError("existing rotation report has a different protocol or source data")
        code_hashes = list(prior.get("runner_sha256_history", [prior.get("runner_sha256")]))
        if report["runner_sha256"] not in code_hashes:
            code_hashes.append(report["runner_sha256"])
        report = prior
        report["runner_sha256_history"] = code_hashes
        report["runner_sha256"] = sha256_file(Path(__file__).resolve())
    write_json("artifacts/reports/ns_pimple_method_v1/rotation_source_lr_diagnostic.json", report)
    sources = {}
    for arm in SOURCE_ARMS:
        report["source_arms"].setdefault(arm, None)
        if report["source_arms"][arm] is None:
            record = train_source_arm(arm, source_data, source_validation, model_config,
                                     model_base, device=device)
            report["source_arms"][arm] = record
            write_json("artifacts/reports/ns_pimple_method_v1/rotation_source_lr_diagnostic.json", report)
        payload = torch.load(ROOT / report["source_arms"][arm]["checkpoint"],
                             map_location="cpu", weights_only=False)
        if (payload.get("arm") != arm
                or sha256_file(ROOT / report["source_arms"][arm]["checkpoint"])
                != report["source_arms"][arm]["checkpoint_sha256"]):
            raise ValueError(f"rotation source checkpoint audit failed for {arm}")
        sources[arm] = payload
    for task in method.TASKS:
        report["target_tasks"].setdefault(task, {})
        for arm in (*SOURCE_ARMS, "lifted_lr_scratch"):
            if arm in report["target_tasks"][task]:
                prior = report["target_tasks"][task][arm]
                checkpoint = ROOT / prior["checkpoint"]
                if not checkpoint.is_file() or sha256_file(checkpoint) != prior["checkpoint_sha256"]:
                    raise ValueError(f"saved rotation target checkpoint hash mismatch: {checkpoint}")
                continue
            record = train_target(task, arm, sources.get(arm), support[task], target_validation[task],
                                  model_config, device=device)
            report["target_tasks"][task][arm] = record
            write_json("artifacts/reports/ns_pimple_method_v1/rotation_source_lr_diagnostic.json", report)
    split_means = {}
    for prefix, split in (("F5_", "development"), ("F6_", "confirmation")):
        split_means[split] = {}
        tasks = [task for task in method.TASKS if task.startswith(prefix)]
        for arm in (*SOURCE_ARMS, "lifted_lr_scratch"):
            rotated = [report["target_tasks"][task][arm]["best_lr_validation_rollout_rel_l2"]
                       for task in tasks]
            original = [evaluate_original_v1(task, arm, target_validation[task], model_config,
                                             device=device) for task in tasks]
            split_means[split][arm] = {
                "rotation_augmented": float(np.mean(rotated)),
                "original_v1": float(np.mean(original)),
                "change": float(np.mean(rotated) - np.mean(original)),
            }
    report["split_mean_lr_validation_rollout_rel_l2"] = split_means
    write_json("artifacts/reports/ns_pimple_method_v1/rotation_source_lr_diagnostic.json", report)
    print(split_means)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
