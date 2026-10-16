#!/usr/bin/env python3
"""Allen--Cahn-style source pretraining, LR NLT adaptation, and sealed NS queries."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = ROOT.parent / "NS-PIMPLE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NS_ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ns_pimple_dataset.reader import NSPimpleDataset  # noqa: E402
from blrpng.models.fno import build_model  # noqa: E402
from blrpng.paths import ensure_artifact_path, sha256_file, write_json  # noqa: E402
from blrpng.training.engine import _train_batch  # noqa: E402
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256  # noqa: E402
from run_ns_pimple_v0_pilot import (  # noqa: E402
    CONDITION_COUNT,
    FIELD_COUNT,
    TimeDependentLiftedOperator,
    _batch,
    _training_config,
    evaluate_rollout,
    prepare_case,
)


CONFIG_PATH = ROOT / "configs/ns_pimple_method_v1.yaml"
PROTOCOL_PATH = ROOT / "docs/prereg/ns_pimple_method_v1_amended.json"
DATA_ROOT = ROOT / "artifacts/data/ns_pimple_method_v1"
CHECKPOINT_ROOT = ROOT / "artifacts/checkpoints/ns_pimple_method_v1"
REPORT_ROOT = ROOT / "artifacts/reports/ns_pimple_method_v1"
SOURCE_ARMS = ("selected_pimple_path", "final_only", "matched_endpoint_control")
TARGET_ARMS = (*SOURCE_ARMS, "lifted_lr_scratch")
TASKS = (
    "F5_dual_sin1_top080_bottom030",
    "F5_dual_sin1_top100_bottom050",
    "F6_left_sin1_a100",
    "F6_left_sin1_a075",
)
EXCLUDED_SOURCE_TASK = "F4_poly24_top/a060"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def final_only_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = ("x", "y", "boundary_lift", "boundary_mask", "field_scales")
    return {key: tensors[key].cpu() for key in keys}


def merge_tensors(cases: list[dict[str, Any]], *, include_path: bool) -> dict[str, torch.Tensor]:
    if not cases:
        raise ValueError("cannot build an empty training dataset")
    keys = (
        ("x", "y", "boundary_lift", "boundary_mask", "newton_correction", "endpoint_increment",
         "valid_correction_mask", "field_scales")
        if include_path else ("x", "y", "boundary_lift", "boundary_mask", "field_scales")
    )
    merged = {}
    for key in keys:
        if key == "field_scales":
            merged[key] = cases[0]["tensors"][key].cpu()
        else:
            merged[key] = torch.cat([case["tensors"][key].cpu() for case in cases], dim=0)
    return merged


def rollout_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in ("pair_id", "u_cells", "previous_cells", "lifts", "grid", "time", "nu", "x_centres", "y_centres")
    }


def with_external_rows(dataset: NSPimpleDataset, v1: dict) -> list[dict]:
    rows = []
    for generated in v1["trajectories"]:
        if generated.get("status") != "complete":
            raise RuntimeError(f"method V1 data generation is incomplete: {generated['pair_id']}")
        row = {
            "pair_id": generated["pair_id"], "task_id": generated["task_id"],
            "family_id": generated["family_id"], "split": generated["split"],
            "role": generated["role"], "status": "complete", "training_access": "none",
            "cases": {generated["level"]: {
                "hdf5": generated["hdf5"], "status": "complete",
                "hdf5_sha256": generated["hdf5_sha256"],
            }},
        }
        dataset.by_id[row["pair_id"]] = row
        rows.append(row)
    return rows


def with_source_pretraining_rows(dataset: NSPimpleDataset, source_manifest: dict) -> list[dict]:
    rows = []
    for generated in source_manifest["trajectories"]:
        if generated.get("status") != "complete":
            raise RuntimeError(f"source pretraining V1 generation is incomplete: {generated['pair_id']}")
        row = {
            "pair_id": generated["pair_id"], "task_id": generated["task_id"],
            "family_id": generated["family_id"], "split": generated["split"],
            "role": generated["role"], "status": "complete", "training_access": "none",
            "cases": {"hr": {"hdf5": generated["hdf5"], "status": "complete",
                              "hdf5_sha256": generated["hdf5_sha256"]}},
        }
        dataset.by_id[row["pair_id"]] = row
        rows.append(row)
    return rows


def build_model_config(base: dict) -> dict:
    model = dict(base["model_search"]["candidates"]["large"])
    return {
        **model,
        "in_channels": 2 * FIELD_COUNT + 2 + CONDITION_COUNT,
        "out_channels": FIELD_COUNT,
        "condition_channels": CONDITION_COUNT,
        "probe_mode": "per_layer",
        "field_scales": [1.0, 1.0],
        "gauge_channels": [],
    }


def select_rows(dataset: NSPimpleDataset, generated: list[dict]) -> dict[str, Any]:
    v0_rows = dataset.pairs()
    source_support = [row for row in v0_rows if row["split"] == "source" and row["role"] == "support"
                      and row["task_id"] != EXCLUDED_SOURCE_TASK]
    source_tasks = {row["task_id"] for row in source_support}
    source_validation = [row for row in v0_rows if row["split"] == "source" and row["role"] == "query"
                         and row["task_id"] in source_tasks]
    target: dict[str, dict[str, list[dict]]] = {}
    for task in TASKS:
        task_id = next(row["task_id"] for row in v0_rows if row["pair_id"] == f"{task}_ic_a")
        split = "dev" if task.startswith("F5_") else "sealed_test"
        base_support = next(row for row in v0_rows if row["task_id"] == task_id and row["role"] == "support")
        new_support = [row for row in generated if row["task_id"] == task_id and row["role"] == "support"]
        validation = [row for row in generated if row["task_id"] == task_id and row["role"] == "validation"]
        base_queries = [row for row in v0_rows if row["task_id"] == task_id and row["role"] == "query"]
        new_queries = [row for row in generated if row["task_id"] == task_id and row["role"] == "query"]
        support = [base_support, *new_support]
        queries = [*base_queries, *new_queries]
        if (len(support), len(validation), len(queries)) != (4, 4, 4):
            raise RuntimeError(f"{task}: expected m=4 support, 4 LR validation, 4 HR query; got "
                               f"{len(support)}/{len(validation)}/{len(queries)}")
        target[task] = {"support": support, "validation": validation, "query": queries,
                        "task_id": task_id, "split": split}
    if len(source_support) != 7 or len(source_validation) != 14:
        raise RuntimeError(f"expected 7 source HR train and 14 source HR validation pairs; got "
                           f"{len(source_support)}/{len(source_validation)}")
    return {"source_train": source_support, "source_validation": source_validation, "target": target}


def prepare_all_data(*, model_config: dict) -> tuple[dict, dict, dict]:
    dataset = NSPimpleDataset(NS_ROOT)
    v1_manifest_path = NS_ROOT / "data/method_v1/manifest.json"
    source_manifest_path = NS_ROOT / "data/source_pretraining_v1/manifest.json"
    v1 = load_json(v1_manifest_path)
    source_manifest = load_json(source_manifest_path)
    dataset_gate_path = NS_ROOT / "runs/ns-method-review-20261015/dataset_audit.json"
    reader_gate_path = NS_ROOT / "runs/ns-method-review-20261015/interface_validation.json"
    method_v1_gate_path = NS_ROOT / "reports/method_v1_dataset_audit.json"
    source_gate_path = NS_ROOT / "reports/source_pretraining_v1_dataset_audit.json"
    dataset_gate = load_json(dataset_gate_path)
    reader_gate = load_json(reader_gate_path)
    method_v1_gate = load_json(method_v1_gate_path)
    source_gate = load_json(source_gate_path)
    if (dataset_gate.get("status") != "pass" or reader_gate.get("status") != "pass"
            or method_v1_gate.get("status") != "pass" or source_gate.get("status") != "pass"):
        raise RuntimeError("the frozen V0 dataset, reader, or V1 data gates are no longer passing")
    generated = with_external_rows(dataset, v1)
    rows = select_rows(dataset, generated)
    source_generated = with_source_pretraining_rows(dataset, source_manifest)
    rows["source_train"].extend(source_generated)
    prepared_source = [prepare_case(dataset, row, "hr", training=False) for row in rows["source_train"]]
    source_validation = [prepare_case(dataset, row, "hr", training=False) for row in rows["source_validation"]]
    target_prepared: dict[str, dict[str, Any]] = {}
    for task, roles in rows["target"].items():
        target_prepared[task] = {
            "support": [prepare_case(dataset, row, "lr", training=False) for row in roles["support"]],
            "validation": [prepare_case(dataset, row, "lr", training=False) for row in roles["validation"]],
            # Do not dereference, checksum, or parse HR target-query HDF5 here.
            "query_rows": roles["query"],
            "task_id": roles["task_id"],
            "split": roles["split"],
        }

    bridge_cases = [*prepared_source]
    for task_data in target_prepared.values():
        bridge_cases.extend(task_data["support"])
    bridge_checks = {
        "all_source_selected_path_labels_finite": all(case["path_labels_finite"] for case in prepared_source),
        "all_source_native_cell_centres_preserved": all(case["native_cell_centres_preserved"] for case in prepared_source),
        "all_source_lift_residual_below_1e-10": max(case["finite_volume_lift_residual_max"] for case in prepared_source) <= 1e-10,
        "all_source_first_hook_gate_passes": min(case["first_D_median"] for case in prepared_source) >= 0.05
        and min(case["first_S_median"] for case in prepared_source) >= 0.01,
        "all_target_support_lift_residual_below_1e-10": max(case["finite_volume_lift_residual_max"]
                                                            for task in target_prepared.values() for case in task["support"]) <= 1e-10,
        "all_target_support_native_cell_centres_preserved": all(case["native_cell_centres_preserved"]
                                                                 for task in target_prepared.values() for case in task["support"]),
    }
    if not all(bridge_checks.values()):
        raise RuntimeError(f"pretraining/adaptation data bridge gate failed: {bridge_checks}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    source_path = merge_tensors(prepared_source, include_path=True)
    source_final = final_only_tensors(source_path)
    source_val = merge_tensors(source_validation, include_path=False)
    save_torch(DATA_ROOT / "source_path_pretraining.pt", source_path)
    save_torch(DATA_ROOT / "source_final_only_pretraining.pt", source_final)
    save_torch(DATA_ROOT / "source_validation_final_only.pt", source_val)
    save_torch(DATA_ROOT / "source_validation_rollouts.pt", [rollout_payload(case) for case in source_validation])
    target_index = {}
    for task, cases in target_prepared.items():
        support_tensors = merge_tensors(cases["support"], include_path=False)
        validation_tensors = merge_tensors(cases["validation"], include_path=False)
        save_torch(DATA_ROOT / f"{task}_support_final_only.pt", support_tensors)
        save_torch(DATA_ROOT / f"{task}_validation_final_only.pt", validation_tensors)
        save_torch(DATA_ROOT / f"{task}_validation_rollouts.pt", [rollout_payload(case) for case in cases["validation"]])
        target_index[task] = {
            "task_id": cases["task_id"], "split": cases["split"],
            "support_pair_ids": [case["pair_id"] for case in cases["support"]],
            "validation_pair_ids": [case["pair_id"] for case in cases["validation"]],
            "query_pair_ids_sealed": [row["pair_id"] for row in cases["query_rows"]],
            "support_transition_count": int(support_tensors["x"].shape[0]),
            "validation_transition_count": int(validation_tensors["x"].shape[0]),
            "support_data": f"artifacts/data/ns_pimple_method_v1/{task}_support_final_only.pt",
            "validation_data": f"artifacts/data/ns_pimple_method_v1/{task}_validation_final_only.pt",
        }
    index = {
        "schema": "blrpng_ns_pimple_method_v1_dataset_index",
        "created_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "preparation_source_sha256": sha256_file(ROOT / "scripts/run_ns_pimple_v0_pilot.py"),
        "model_base_config_sha256": sha256_file(ROOT / "configs/navier_stokes_sfepy_seed42.yaml"),
        "v0_manifest_sha256": sha256_file(NS_ROOT / "data/v0/manifest.json"),
        "v1_manifest_sha256": sha256_file(v1_manifest_path),
        "v1_generator_source_sha256": sha256_file(NS_ROOT / "scripts/generate_method_v1_data.py"),
        "dataset_gate_sha256": sha256_file(dataset_gate_path),
        "reader_gate_sha256": sha256_file(reader_gate_path),
        "method_v1_dataset_gate_sha256": sha256_file(method_v1_gate_path),
        "source_pretraining_manifest_sha256": sha256_file(source_manifest_path),
        "source_pretraining_generator_sha256": sha256_file(NS_ROOT / "scripts/generate_source_pretraining_v1.py"),
        "source_pretraining_audit_source_sha256": sha256_file(NS_ROOT / "scripts/audit_source_pretraining_v1.py"),
        "source_pretraining_dataset_gate_sha256": sha256_file(source_gate_path),
        "path_screen_sha256": sha256_file(ROOT / "artifacts/reports/ns_pimple_v0_pilot/selected_iteration_screen.json"),
        "source_train_pair_ids": [row["pair_id"] for row in rows["source_train"]],
        "source_validation_pair_ids": [row["pair_id"] for row in rows["source_validation"]],
        "source_transition_count": int(source_path["x"].shape[0]),
        "source_selected_path_label_count": int(source_path["newton_correction"].shape[0]),
        "source_path_labels_per_physical_transition": 1,
        "target_tasks": target_index,
        "bridge_gate": {"status": "pass", "checks": bridge_checks,
                        "maximum_source_lift_residual": max(case["finite_volume_lift_residual_max"] for case in prepared_source),
                        "minimum_source_first_hook_D_median": min(case["first_D_median"] for case in prepared_source),
                        "minimum_source_first_hook_S_median": min(case["first_S_median"] for case in prepared_source)},
        "target_hr_queries_read": False,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in DATA_ROOT.glob("*.pt")
        },
    }
    write_json("artifacts/data/ns_pimple_method_v1/dataset_index.json", index)
    return index, {"source_path": source_path, "source_validation": [rollout_payload(case) for case in source_validation]}, target_prepared


def mean_rollout_error(model: TimeDependentLiftedOperator, cases: list[dict], device: torch.device) -> tuple[float, list[dict]]:
    results = [evaluate_rollout(model, case, device) for case in cases]
    return float(np.mean([row["native_cell_center_rollout_rel_l2_mean"] for row in results])), results


@torch.no_grad()
def detailed_frozen_evaluation(model: TimeDependentLiftedOperator, case: dict, device: torch.device) -> dict:
    """Report teacher-forced and autoregressive errors to localize rollout failures."""
    fields = FIELD_COUNT
    height, width = case["u_cells"].shape[-2:]
    grid = torch.from_numpy(case["grid"][None]).to(device)
    boundary_mask = torch.zeros((1, fields, height, width), dtype=torch.bool, device=device)
    one_step_errors: list[float] = []
    rollout_errors: list[float] = []
    one_step_field_errors: list[list[float]] = []
    rollout_field_errors: list[list[float]] = []
    rollout_norm_ratios: list[float] = []
    divergence_pred: list[float] = []
    divergence_ref: list[float] = []

    def infer(state: np.ndarray, history: np.ndarray, step: int) -> np.ndarray:
        dt = float(case["time"][step + 1] - case["time"][step])
        condition = np.concatenate((
            history.astype(np.float32), case["lifts"][step], case["lifts"][step + 1],
            np.full((1, height, width), case["nu"], dtype=np.float32),
            np.full((1, height, width), dt, dtype=np.float32),
        ), axis=0)
        x_np = np.concatenate((state.astype(np.float32), case["grid"], condition), axis=0)
        x = torch.from_numpy(x_np[None]).to(device)
        lift = torch.from_numpy(case["lifts"][step][None]).to(device)
        prediction, _ = model(x, lift, boundary_mask)
        return prediction[0].detach().cpu().numpy()

    def divergence_rms(value: np.ndarray) -> float:
        du_dx = np.gradient(value[0], case["x_centres"], axis=1, edge_order=2)
        dv_dy = np.gradient(value[1], case["y_centres"], axis=0, edge_order=2)
        return float(np.sqrt(np.mean((du_dx + dv_dy) ** 2)))

    model.eval()
    current = case["u_cells"][0].astype(np.float32)
    history = case["previous_cells"][0].astype(np.float32)
    for step in range(len(case["time"]) - 1):
        target = case["u_cells"][step + 1]
        # Teacher-forced evaluation uses the true current and BDF history states.
        one_step = infer(case["u_cells"][step], case["previous_cells"][step], step)
        one_step_errors.append(float(np.linalg.norm(one_step - target) / max(np.linalg.norm(target), 1e-12)))
        one_step_field_errors.append([
            float(np.linalg.norm(one_step[channel] - target[channel]) / max(np.linalg.norm(target[channel]), 1e-12))
            for channel in range(fields)
        ])
        # Rollout evaluation feeds back the model's own prior prediction.
        prediction = infer(current, history, step)
        rollout_errors.append(float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1e-12)))
        rollout_field_errors.append([
            float(np.linalg.norm(prediction[channel] - target[channel]) / max(np.linalg.norm(target[channel]), 1e-12))
            for channel in range(fields)
        ])
        rollout_norm_ratios.append(float(np.linalg.norm(prediction) / max(np.linalg.norm(target), 1e-12)))
        divergence_pred.append(divergence_rms(prediction))
        divergence_ref.append(divergence_rms(target))
        history, current = current, prediction
    return {
        "pair_id": case["pair_id"],
        "resolution": [int(height), int(width)],
        "steps": len(rollout_errors),
        "teacher_forced_rel_l2_mean": float(np.mean(one_step_errors)),
        "teacher_forced_rel_l2_final": float(one_step_errors[-1]),
        "teacher_forced_per_step_rel_l2": one_step_errors,
        "teacher_forced_per_field_rel_l2_mean": np.mean(np.asarray(one_step_field_errors), axis=0).tolist(),
        "rollout_rel_l2_mean": float(np.mean(rollout_errors)),
        "rollout_rel_l2_final": float(rollout_errors[-1]),
        "rollout_per_step_rel_l2": rollout_errors,
        "rollout_per_field_rel_l2_mean": np.mean(np.asarray(rollout_field_errors), axis=0).tolist(),
        "prediction_to_reference_norm_ratio_mean": float(np.mean(rollout_norm_ratios)),
        "cell_center_divergence_rms_pred_mean": float(np.mean(divergence_pred)),
        "cell_center_divergence_rms_reference_mean": float(np.mean(divergence_ref)),
    }


def train_source_arm(
    arm: str, train_data: dict[str, torch.Tensor], validation_cases: list[dict], model_config: dict,
    base_config: dict, *, seed: int, device: torch.device, batch_size: int,
) -> dict:
    set_seed(seed, deterministic=True)
    base = build_model(model_config, representation="lifted")
    model = TimeDependentLiftedOperator(base).to(device)
    objective_map = {
        # `_train_batch` still names this generic correction tensor `newton_correction`;
        # the payload here is the screened PIMPLE outer correction, not a Newton update.
        "selected_pimple_path": ("blrpng_safe", "newton_correction"),
        "final_only": ("final_only", None),
        "matched_endpoint_control": ("blrpng_safe", "endpoint_direction_path_norm"),
    }
    objective, auxiliary = objective_map[arm]
    training = _training_config(base_config, auxiliary)
    training.update({"path_gradient_ratio": 0.05, "path_layer": 1,
                     "path_log_magnitude_weight": 0.1, "path_epsilon": 1e-8,
                     "final_loss": "joint_relative_l2", "rollout_horizon": 1,
                     "vorticity_weight": 0.0, "divergence_weight": 0.0})
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    sample_count = int(train_data["x"].shape[0])
    steps_per_epoch = math.ceil(sample_count / batch_size)
    max_epochs, min_epochs, patience = 60, 30, 12
    generator = torch.Generator(device="cpu").manual_seed(271828)
    checkpoint_path = ensure_artifact_path(f"artifacts/checkpoints/ns_pimple_method_v1/source_{arm}_seed{seed}.pt")
    history: list[dict] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(sample_count, generator=generator)
        train_losses = []
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size].tolist()
            batch = _batch(train_data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _train_batch(model=model, batch=batch, objective=objective,
                                         training_config=training, epoch_progress=(epoch - 1) / max(max_epochs - 1, 1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {arm} source loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_score, _ = mean_rollout_error(model, validation_cases, device)
        row = {"epoch": epoch, "optimizer_updates": epoch * steps_per_epoch,
               "train_loss_mean": float(np.mean(train_losses)), "validation_rollout_rel_l2": validation_score,
               "path_loss_last": float(metrics.get("path_loss", 0.0)),
               "gradient_ratio_last": float(metrics.get("gradient_ratio", 0.0))}
        history.append(row)
        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            stale_epochs = 0
            model_hash = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_method_v1_source_checkpoint",
                "arm": arm, "seed": seed, "epoch": epoch,
                "base_state_dict": model.base.state_dict(), "model_config": model_config,
                "model_state_sha256": model_hash, "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "query_used": False,
            }, checkpoint_path)
        else:
            stale_epochs += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"source {arm} epoch={epoch} train={row['train_loss_mean']:.6g} "
                  f"source-val={validation_score:.6g} best={best_score:.6g}@{best_epoch}", flush=True)
        if epoch >= min_epochs and stale_epochs >= patience:
            break
    if best_epoch == 0:
        raise RuntimeError(f"source arm failed to select a checkpoint: {arm}")
    checkpoint_sha = sha256_file(checkpoint_path)
    return {"arm": arm, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha, "model_state_sha256": model_hash,
            "best_epoch": best_epoch, "epochs_run": len(history), "best_source_validation_rollout_rel_l2": best_score,
            "optimizer_updates": len(history) * steps_per_epoch, "training_history": history,
            "training_pair_count": sample_count // 20, "training_transition_count": sample_count,
            "validation_pair_count": len(validation_cases), "query_used": False}


def train_target(
    task: str, arm: str, source_checkpoint: dict | None,
    support: dict[str, torch.Tensor], validation_cases: list[dict], model_config: dict,
    *, seed: int, device: torch.device,
) -> dict:
    set_seed(seed, deterministic=True)
    base = build_model(model_config, representation="lifted")
    if source_checkpoint is not None:
        base.load_state_dict(source_checkpoint["base_state_dict"], strict=True)
        adapter_info = base.configure_adaptation("nlt", rank=4, alpha=4.0)
        max_epochs, min_epochs, patience, validation_interval = 100, 30, 12, 1
        learning_rate, weight_decay = 1e-3, 1e-6
        batch_size = 4
        objective = "final_only"
        trainable_parameters = int(adapter_info["trainable_parameters"])
    else:
        max_epochs, min_epochs, patience, validation_interval = 500, 50, 25, 5
        learning_rate, weight_decay = 3e-4, 1e-4
        batch_size = 4
        objective = "final_only"
        trainable_parameters = sum(parameter.numel() for parameter in base.parameters())
    model = TimeDependentLiftedOperator(base).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    sample_count = int(support["x"].shape[0])
    steps_per_epoch = math.ceil(sample_count / batch_size)
    generator = torch.Generator(device="cpu").manual_seed(271828)
    checkpoint_path = ensure_artifact_path(
        f"artifacts/checkpoints/ns_pimple_method_v1/target_{task}_{arm}_seed{seed}.pt"
    )
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(sample_count, generator=generator)
        train_losses = []
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size].tolist()
            batch = _batch(support, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _train_batch(model=model, batch=batch, objective=objective,
                                   training_config={"final_loss": "joint_relative_l2", "rollout_horizon": 1},
                                   epoch_progress=1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite target loss for {task}/{arm} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        should_validate = epoch % validation_interval == 0 or epoch == max_epochs
        if not should_validate:
            continue
        model.eval()
        validation_score, validation_rows = mean_rollout_error(model, validation_cases, device)
        row = {"epoch": epoch, "optimizer_updates": epoch * steps_per_epoch,
               "train_loss_mean": float(np.mean(train_losses)), "validation_rollout_rel_l2": validation_score}
        history.append(row)
        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            stale = 0
            model_hash = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_method_v1_target_checkpoint",
                "task": task, "arm": arm, "seed": seed, "epoch": epoch,
                "adapter": "nlt" if source_checkpoint is not None else "none_scratch",
                "source_arm": source_checkpoint["arm"] if source_checkpoint is not None else None,
                "base_state_dict": model.base.state_dict(), "model_config": model_config,
                "model_state_sha256": model_hash, "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "query_used": False,
            }, checkpoint_path)
            best_validation_rows = validation_rows
        else:
            stale += 1
        if epoch % max(validation_interval * 5, 5) == 0 or epoch == validation_interval:
            print(f"target {task} {arm} epoch={epoch} train={row['train_loss_mean']:.6g} "
                  f"LR-val={validation_score:.6g} best={best_score:.6g}@{best_epoch}", flush=True)
        if epoch >= min_epochs and stale >= patience:
            break
    if best_epoch == 0:
        raise RuntimeError(f"target arm failed to select a checkpoint: {task}/{arm}")
    return {"task": task, "arm": arm, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path), "model_state_sha256": model_hash,
            "best_epoch": best_epoch, "epochs_run": history[-1]["epoch"],
            "optimizer_updates": history[-1]["epoch"] * steps_per_epoch,
            "best_lr_validation_rollout_rel_l2": best_score,
            "validation_per_rollout": best_validation_rows,
            "support_trajectory_count": 4, "support_transition_count": sample_count,
            "validation_trajectory_count": len(validation_cases),
            "trainable_parameters": trainable_parameters, "target_path_used": False,
            "training_history": history}


def phase_prepare(model_config: dict) -> tuple[dict, dict, dict]:
    index, _, target = prepare_all_data(model_config=model_config)
    print(json.dumps({"status": "prepared", "source_transitions": index["source_transition_count"],
                      "source_path_labels": index["source_selected_path_label_count"],
                      "target_tasks": {task: {"support": len(data["support"]), "validation": len(data["validation"]),
                                              "sealed_query_rows": len(data["query_rows"])}
                                       for task, data in target.items()},
                      "target_hr_queries_read": False}, indent=2))
    return index, {}, target


def phase_source(model_config: dict, base_config: dict, seed: int, device: torch.device, batch_size: int) -> dict:
    train_data = load_torch(DATA_ROOT / "source_path_pretraining.pt")
    validation_cases = load_torch(DATA_ROOT / "source_validation_rollouts.pt")
    report = {"schema": "blrpng_ns_pimple_method_v1_source_training", "protocol_sha256": sha256_file(PROTOCOL_PATH),
              "device": str(device), "seed": seed, "target_hr_queries_read": False, "arms": {}}
    for arm in SOURCE_ARMS:
        path = CHECKPOINT_ROOT / f"source_{arm}_seed{seed}.pt"
        if path.exists():
            # A complete source checkpoint can be safely reused only if it belongs to this protocol.
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("protocol_sha256") != sha256_file(PROTOCOL_PATH) or saved.get("arm") != arm:
                raise ValueError(f"existing source checkpoint has a different protocol: {path}")
            prior = REPORT_ROOT / "source_training.json"
            if prior.exists() and arm in load_json(prior).get("arms", {}):
                report["arms"][arm] = load_json(prior)["arms"][arm]
                continue
        report["arms"][arm] = train_source_arm(arm, train_data, validation_cases, model_config,
                                                base_config, seed=seed, device=device, batch_size=batch_size)
        write_json("artifacts/reports/ns_pimple_method_v1/source_training.json", report)
    write_json("artifacts/reports/ns_pimple_method_v1/source_training.json", report)
    return report


def phase_target(model_config: dict, seed: int, device: torch.device) -> dict:
    source_report = load_json(REPORT_ROOT / "source_training.json")
    source_models = {}
    for arm in SOURCE_ARMS:
        payload = torch.load(ROOT / source_report["arms"][arm]["checkpoint"], map_location="cpu", weights_only=False)
        source_models[arm] = payload
    report_path = REPORT_ROOT / "target_training.json"
    if report_path.exists():
        report = load_json(report_path)
    else:
        report = {"schema": "blrpng_ns_pimple_method_v1_target_training", "protocol_sha256": sha256_file(PROTOCOL_PATH),
                  "seed": seed, "target_hr_queries_read": False, "tasks": {}}
    dataset = NSPimpleDataset(NS_ROOT)
    v1 = load_json(NS_ROOT / "data/method_v1/manifest.json")
    generated = with_external_rows(dataset, v1)
    selected = select_rows(dataset, generated)
    for task in TASKS:
        report["tasks"].setdefault(task, {})
        support = load_torch(DATA_ROOT / f"{task}_support_final_only.pt")
        validation_cases = load_torch(DATA_ROOT / f"{task}_validation_rollouts.pt")
        for arm in TARGET_ARMS:
            if arm in report["tasks"][task]:
                continue
            checkpoint_payload = None if arm == "lifted_lr_scratch" else source_models[arm]
            result = train_target(task, arm, checkpoint_payload, support, validation_cases,
                                  model_config, seed=seed, device=device)
            result.update({"task_id": selected["target"][task]["task_id"],
                           "split": selected["target"][task]["split"],
                           "support_pair_ids": [row["pair_id"] for row in selected["target"][task]["support"]],
                           "validation_pair_ids": [row["pair_id"] for row in selected["target"][task]["validation"]],
                           "sealed_query_pair_ids": [row["pair_id"] for row in selected["target"][task]["query"]]})
            report["tasks"][task][arm] = result
            write_json("artifacts/reports/ns_pimple_method_v1/target_training.json", report)
    write_json("artifacts/reports/ns_pimple_method_v1/target_training.json", report)

    # This global freeze binds every source and target checkpoint before target HR is opened.
    freeze = {
        "schema": "blrpng_ns_pimple_method_v1_global_checkpoint_freeze",
        "created_utc": utc_now(), "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "model_base_config_sha256": sha256_file(ROOT / "configs/navier_stokes_sfepy_seed42.yaml"),
        "dataset_index_sha256": sha256_file(DATA_ROOT / "dataset_index.json"),
        "preparation_source_sha256": sha256_file(ROOT / "scripts/run_ns_pimple_v0_pilot.py"),
        "v0_manifest_sha256": sha256_file(NS_ROOT / "data/v0/manifest.json"),
        "v1_manifest_sha256": sha256_file(NS_ROOT / "data/method_v1/manifest.json"),
        "v1_generator_source_sha256": sha256_file(NS_ROOT / "scripts/generate_method_v1_data.py"),
        "method_v1_dataset_gate_sha256": sha256_file(NS_ROOT / "reports/method_v1_dataset_audit.json"),
        "source_pretraining_manifest_sha256": sha256_file(NS_ROOT / "data/source_pretraining_v1/manifest.json"),
        "source_pretraining_generator_sha256": sha256_file(NS_ROOT / "scripts/generate_source_pretraining_v1.py"),
        "source_pretraining_audit_source_sha256": sha256_file(NS_ROOT / "scripts/audit_source_pretraining_v1.py"),
        "source_pretraining_dataset_gate_sha256": sha256_file(NS_ROOT / "reports/source_pretraining_v1_dataset_audit.json"),
        "path_screen_sha256": sha256_file(ROOT / "artifacts/reports/ns_pimple_v0_pilot/selected_iteration_screen.json"),
        "source_checkpoints": {arm: report_row["checkpoint_sha256"]
                               for arm, report_row in source_report["arms"].items()},
        "target_checkpoints": {task: {arm: row["checkpoint_sha256"] for arm, row in arms.items()}
                               for task, arms in report["tasks"].items()},
        "evaluator": "run_ns_pimple_method_v1.py::evaluate_frozen_queries",
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "target_hr_query_used_for_checkpoint_selection": False,
        "sealed_test_query_used_for_checkpoint_selection": False,
        "target_hr_query_read": False,
    }
    freeze_path = REPORT_ROOT / "global_checkpoint_freeze.json"
    write_json("artifacts/reports/ns_pimple_method_v1/global_checkpoint_freeze.json", freeze)
    return {"training": report, "freeze": freeze, "freeze_path": str(freeze_path.relative_to(ROOT))}


def verify_freeze(freeze: dict) -> None:
    if freeze.get("target_hr_query_read") is not False:
        raise ValueError("target HR query has already been opened; this evaluator only handles the first frozen read")
    if freeze.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ValueError("protocol hash changed after checkpoint freeze")
    if freeze.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("experiment config changed after checkpoint freeze")
    if freeze.get("model_base_config_sha256") != sha256_file(ROOT / "configs/navier_stokes_sfepy_seed42.yaml"):
        raise ValueError("base model config changed after checkpoint freeze")
    if freeze.get("dataset_index_sha256") != sha256_file(DATA_ROOT / "dataset_index.json"):
        raise ValueError("prepared dataset index changed after checkpoint freeze")
    if freeze.get("preparation_source_sha256") != sha256_file(ROOT / "scripts/run_ns_pimple_v0_pilot.py"):
        raise ValueError("data preparation source changed after checkpoint freeze")
    if freeze.get("path_screen_sha256") != sha256_file(ROOT / "artifacts/reports/ns_pimple_v0_pilot/selected_iteration_screen.json"):
        raise ValueError("path-selection gate report changed after checkpoint freeze")
    if freeze.get("evaluator_source_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("evaluator source changed after checkpoint freeze")
    if freeze.get("v0_manifest_sha256") != sha256_file(NS_ROOT / "data/v0/manifest.json"):
        raise ValueError("V0 manifest changed after checkpoint freeze")
    if freeze.get("v1_manifest_sha256") != sha256_file(NS_ROOT / "data/method_v1/manifest.json"):
        raise ValueError("V1 data-role manifest changed after checkpoint freeze")
    if freeze.get("v1_generator_source_sha256") != sha256_file(NS_ROOT / "scripts/generate_method_v1_data.py"):
        raise ValueError("V1 data generator changed after checkpoint freeze")
    if freeze.get("method_v1_dataset_gate_sha256") != sha256_file(NS_ROOT / "reports/method_v1_dataset_audit.json"):
        raise ValueError("V1 dataset gate report changed after checkpoint freeze")
    if freeze.get("source_pretraining_manifest_sha256") != sha256_file(NS_ROOT / "data/source_pretraining_v1/manifest.json"):
        raise ValueError("source-pretraining V1 manifest changed after checkpoint freeze")
    if freeze.get("source_pretraining_generator_sha256") != sha256_file(NS_ROOT / "scripts/generate_source_pretraining_v1.py"):
        raise ValueError("source-pretraining V1 generator changed after checkpoint freeze")
    if freeze.get("source_pretraining_audit_source_sha256") != sha256_file(NS_ROOT / "scripts/audit_source_pretraining_v1.py"):
        raise ValueError("source-pretraining V1 audit code changed after checkpoint freeze")
    if freeze.get("source_pretraining_dataset_gate_sha256") != sha256_file(NS_ROOT / "reports/source_pretraining_v1_dataset_audit.json"):
        raise ValueError("source-pretraining V1 data gate changed after checkpoint freeze")
    dataset_index = load_json(DATA_ROOT / "dataset_index.json")
    for relative_path, artifact in dataset_index.get("artifacts", {}).items():
        artifact_path = DATA_ROOT / relative_path
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"prepared data tensor changed after freeze: {relative_path}")
    if dataset_index.get("target_hr_queries_read") is not False:
        raise ValueError("dataset index indicates target HR queries were read before freeze")
    for arm, digest in freeze["source_checkpoints"].items():
        source_path = CHECKPOINT_ROOT / f"source_{arm}_seed42.pt"
        if sha256_file(source_path) != digest:
            raise ValueError(f"source checkpoint changed after freeze: {arm}")
    for task, arms in freeze["target_checkpoints"].items():
        for arm, digest in arms.items():
            checkpoint = CHECKPOINT_ROOT / f"target_{task}_{arm}_seed42.pt"
            if sha256_file(checkpoint) != digest:
                raise ValueError(f"target checkpoint changed after freeze: {task}/{arm}")


def evaluate_frozen_queries(model_config: dict, device: torch.device) -> dict:
    freeze_path = REPORT_ROOT / "global_checkpoint_freeze.json"
    freeze = load_json(freeze_path)
    verify_freeze(freeze)
    # The durable log is written immediately before loading the first HR target query.
    access_log_path = REPORT_ROOT / "query_access_log.json"
    if access_log_path.exists():
        raise FileExistsError(f"query access log already exists; refusing a second target query read: {access_log_path}")
    dataset = NSPimpleDataset(NS_ROOT)
    v1 = load_json(NS_ROOT / "data/method_v1/manifest.json")
    generated = with_external_rows(dataset, v1)
    selected = select_rows(dataset, generated)
    all_query_rows = {task: selected["target"][task]["query"] for task in TASKS}
    access_log = {
        "schema": "blrpng_ns_pimple_method_v1_target_query_access_log",
        "freeze_sha256": sha256_file(freeze_path), "started_utc": utc_now(),
        "query_pair_ids": {task: [row["pair_id"] for row in rows] for task, rows in all_query_rows.items()},
        "sealed_test_opened": True,
        "purpose": "one frozen evaluation pass; no checkpoint selection or model changes",
    }
    write_json("artifacts/reports/ns_pimple_method_v1/query_access_log.json", access_log)
    prepared_queries = {
        task: [prepare_case(dataset, row, "hr", training=False) for row in rows]
        for task, rows in all_query_rows.items()
    }
    results = {"schema": "blrpng_ns_pimple_method_v1_frozen_query_result",
               "freeze_sha256": sha256_file(freeze_path), "query_access_log": str(access_log_path.relative_to(ROOT)),
               "query_access_started_utc": access_log["started_utc"], "seed": 42, "device": str(device),
               "primary_metric": "mean 20-step vector velocity relative L2", "tasks": {}}
    for task in TASKS:
        results["tasks"][task] = {"split": selected["target"][task]["split"], "arms": {}}
        for arm in TARGET_ARMS:
            checkpoint_path = CHECKPOINT_ROOT / f"target_{task}_{arm}_seed42.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            base = build_model(model_config, representation="lifted")
            if checkpoint.get("adapter") == "nlt":
                base.configure_adaptation("nlt", rank=4, alpha=4.0)
            base.load_state_dict(checkpoint["base_state_dict"], strict=True)
            model = TimeDependentLiftedOperator(base).to(device)
            model.eval()
            query_rows = [detailed_frozen_evaluation(model, case, device) for case in prepared_queries[task]]
            results["tasks"][task]["arms"][arm] = {
                "query_pair_ids": [row["pair_id"] for row in query_rows],
                "mean_relative_l2": float(np.mean([row["rollout_rel_l2_mean"] for row in query_rows])),
                "final_step_relative_l2_mean": float(np.mean([row["rollout_rel_l2_final"] for row in query_rows])),
                "per_query": query_rows,
            }
            del model, base, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
    results["development_mean"] = {}
    results["confirmation_mean"] = {}
    for arm in TARGET_ARMS:
        for split_name, task_prefix, dest in (("development", "F5_", results["development_mean"]),
                                              ("confirmation", "F6_", results["confirmation_mean"])):
            task_values = [results["tasks"][task]["arms"][arm]["mean_relative_l2"] for task in TASKS if task.startswith(task_prefix)]
            dest[arm] = float(np.mean(task_values))
    path_value = results["development_mean"]["selected_pimple_path"]
    results["development_relative_reduction_vs"] = {
        arm: float((results["development_mean"][arm] - path_value) / results["development_mean"][arm])
        for arm in TARGET_ARMS if arm != "selected_pimple_path"
    }
    path_value = results["confirmation_mean"]["selected_pimple_path"]
    results["confirmation_relative_reduction_vs"] = {
        arm: float((results["confirmation_mean"][arm] - path_value) / results["confirmation_mean"][arm])
        for arm in TARGET_ARMS if arm != "selected_pimple_path"
    }
    access_log["completed_utc"] = utc_now()
    access_log["status"] = "complete"
    write_json("artifacts/reports/ns_pimple_method_v1/query_access_log.json", access_log)
    write_json("artifacts/reports/ns_pimple_method_v1/frozen_query_result.json", results)
    print(json.dumps({"status": "query_evaluation_complete", "development_mean": results["development_mean"],
                      "confirmation_mean": results["confirmation_mean"],
                      "development_reduction": results["development_relative_reduction_vs"],
                      "confirmation_reduction": results["confirmation_relative_reduction_vs"]}, indent=2))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "source", "target", "evaluate", "all"), default="all")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("seed is preregistered at 42; create a new preregistration for another seed")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() <= device.index):
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    base_config = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = build_model_config(base_config)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    phases = ("prepare", "source", "target", "evaluate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "prepare":
            phase_prepare(model_config)
        elif phase == "source":
            phase_source(model_config, base_config, args.seed, device,
                         batch_size=int(config["source_pretraining"]["batch_size"]))
        elif phase == "target":
            phase_target(model_config, args.seed, device)
        elif phase == "evaluate":
            freeze = load_json(REPORT_ROOT / "global_checkpoint_freeze.json")
            verify_freeze(freeze)
            evaluate_frozen_queries(model_config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
