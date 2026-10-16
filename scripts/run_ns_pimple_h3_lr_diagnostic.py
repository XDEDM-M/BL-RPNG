#!/usr/bin/env python3
"""Exploratory NS H3 target-adaptation diagnostic; reads LR support/validation only."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import run_ns_pimple_method_v1 as method
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.training.losses import per_sample_relative_l2
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256


ROOT = method.ROOT
REPORT_DIR = method.REPORT_ROOT / "h3_lr_diagnostic"
ARMS = method.TARGET_ARMS
HORIZON = 3


def make_windows(data: dict[str, torch.Tensor], *, pair_count: int) -> dict[str, torch.Tensor]:
    transitions = int(data["x"].shape[0])
    expected = pair_count * 20
    if transitions != expected:
        raise ValueError(f"expected {expected} ordered transitions, found {transitions}")
    windows = {"x": [], "y": [], "boundary_lift": [], "boundary_mask": [], "future_targets": []}
    for trajectory in range(pair_count):
        offset = trajectory * 20
        for start in range(20 - HORIZON + 1):
            ids = [offset + start + step for step in range(HORIZON)]
            windows["x"].append(data["x"][ids])
            windows["y"].append(data["y"][ids[0]])
            windows["boundary_lift"].append(data["boundary_lift"][ids[0]])
            windows["boundary_mask"].append(data["boundary_mask"][ids[0]])
            windows["future_targets"].append(data["y"][ids])
    return {key: torch.stack(value) for key, value in windows.items()}


def batch_windows(windows: dict[str, torch.Tensor], ids: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(ids, dtype=torch.long)
    return {key: value.index_select(0, index).to(device) for key, value in windows.items()}


def h3_relative_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    sequence_x = batch["x"]
    targets = batch["future_targets"]
    current = sequence_x[:, 0, : method.FIELD_COUNT]
    history = sequence_x[:, 0, 4: 4 + method.FIELD_COUNT]
    losses = []
    for step in range(HORIZON):
        template = sequence_x[:, step]
        if step == 0:
            step_x = template
        else:
            # Keep the known per-step grid, boundary lift, viscosity and dt,
            # while feeding the model's prior state and autoregressive history.
            step_x = torch.cat((current, template[:, 2:4], history, template[:, 6:]), dim=1)
        prediction, _ = model(step_x, batch["boundary_lift"] if step == 0 else batch["future_lifts"][:, step],
                              batch["boundary_mask"] if step == 0 else batch["future_masks"][:, step])
        losses.append(per_sample_relative_l2(prediction, targets[:, step]).mean())
        history, current = current, prediction
    return torch.stack(losses).mean()


def add_future_constraints(data: dict[str, torch.Tensor], windows: dict[str, torch.Tensor]) -> None:
    """Attach per-step lifts/masks without loading or consulting any HR queries."""
    lifts, masks = [], []
    transitions = int(data["x"].shape[0])
    pair_count = transitions // 20
    for trajectory in range(pair_count):
        offset = trajectory * 20
        for start in range(20 - HORIZON + 1):
            ids = [offset + start + step for step in range(HORIZON)]
            lifts.append(data["boundary_lift"][ids])
            masks.append(data["boundary_mask"][ids])
    windows["future_lifts"] = torch.stack(lifts)
    windows["future_masks"] = torch.stack(masks)


def train_h3_arm(task: str, arm: str, source_payload: dict | None, support: dict[str, torch.Tensor],
                 validation_cases: list[dict], model_config: dict, *, seed: int,
                 device: torch.device, support_pair_count: int) -> dict:
    set_seed(seed, deterministic=True)
    windows = make_windows(support, pair_count=support_pair_count)
    add_future_constraints(support, windows)
    base = method.build_model(model_config, representation="lifted")
    if source_payload is not None:
        base.load_state_dict(source_payload["base_state_dict"], strict=True)
        adapter_info = base.configure_adaptation("nlt", rank=4, alpha=4.0)
        learning_rate, weight_decay, adapter_name = 1e-3, 1e-6, "nlt"
        trainable_parameters = int(adapter_info["trainable_parameters"])
    else:
        learning_rate, weight_decay, adapter_name = 1e-4, 1e-4, "none_scratch"
        trainable_parameters = sum(parameter.numel() for parameter in base.parameters())
    model = method.TimeDependentLiftedOperator(base).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    sample_count = int(windows["x"].shape[0])
    steps_per_epoch = math.ceil(sample_count / 4)
    permutation_generator = torch.Generator(device="cpu").manual_seed(271828)
    checkpoint_path = ensure_artifact_path(
        f"artifacts/checkpoints/ns_pimple_method_v1/h3_lr_diagnostic/target_{task}_{arm}_seed{seed}.pt"
    )
    best_score, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, 101):
        model.train()
        permutation = torch.randperm(sample_count, generator=permutation_generator)
        losses = []
        for start in range(0, sample_count, 4):
            ids = permutation[start : start + 4].tolist()
            batch = batch_windows(windows, ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss = h3_relative_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite H3 loss in {task}/{arm} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        score, validation_rows = method.mean_rollout_error(model, validation_cases, device)
        row = {"epoch": epoch, "optimizer_updates": epoch * steps_per_epoch,
               "train_h3_loss_mean": float(np.mean(losses)), "validation_rollout_rel_l2": score,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            digest = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_h3_lr_diagnostic_checkpoint",
                "task": task, "arm": arm, "seed": seed, "epoch": epoch,
                "adapter": adapter_name, "base_state_dict": model.base.state_dict(),
                "model_config": model_config, "model_state_sha256": digest,
                "source_checkpoint_sha256": source_payload["model_state_sha256"] if source_payload else None,
                "horizon": HORIZON, "selection_data": "LR validation only",
                "target_hr_query_used": False,
            }, checkpoint_path)
            best_validation_rows = validation_rows
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"H3 {task} {arm} epoch={epoch} train={row['train_h3_loss_mean']:.6g} "
                  f"LR-val={score:.6g} best={best_score:.6g}@{best_epoch}", flush=True)
    return {
        "task": task, "arm": arm, "horizon": HORIZON, "support_window_count": sample_count,
        "optimizer_updates": len(history) * steps_per_epoch, "epochs_run": len(history),
        "best_epoch": best_epoch, "best_lr_validation_rollout_rel_l2": best_score,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trainable_parameters": trainable_parameters,
        "target_hr_query_used": False, "validation_per_rollout": best_validation_rows,
        "training_history": history,
    }


def main() -> int:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.device_count() < 2:
        raise RuntimeError("The configured diagnostic device cuda:1 is unavailable.")
    import yaml

    model_config = method.build_model_config(
        yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    )
    source_report = method.load_json(method.REPORT_ROOT / "source_training.json")
    dataset_index = method.load_json(method.DATA_ROOT / "dataset_index.json")
    report = {
        "schema": "blrpng_ns_pimple_method_v1_h3_lr_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Allen-Cahn R2 fixed-horizon-3 analogue with NS-specific per-step history/lifts/nu/dt",
        "selection_metric": "mean 20-step LR autoregressive relative L2",
        "device": str(device), "seed": 42, "horizon": HORIZON,
        "target_hr_query_used": False, "tasks": {},
        "data_index_sha256": sha256_file(method.DATA_ROOT / "dataset_index.json"),
        "diagnostic_preregistration_sha256": sha256_file(
            ROOT / "docs/prereg/ns_pimple_method_v1_h3_lr_diagnostic.json"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_checkpoint_sha256": {
            arm: source_report["arms"][arm]["checkpoint_sha256"]
            for arm in method.SOURCE_ARMS
        },
    }
    source_payloads = {}
    for arm in method.SOURCE_ARMS:
        record = source_report["arms"][arm]
        source_payloads[arm] = torch.load(ROOT / record["checkpoint"], map_location="cpu", weights_only=False)
    for task in method.TASKS:
        report["tasks"][task] = {}
        support = torch.load(method.DATA_ROOT / f"{task}_support_final_only.pt", map_location="cpu", weights_only=False)
        validation_cases = method.load_torch(method.DATA_ROOT / f"{task}_validation_rollouts.pt")
        task_index = dataset_index["target_tasks"][task]
        if len(task_index["support_pair_ids"]) != 4 or task_index["support_transition_count"] != 80:
            raise ValueError(f"unexpected LR support role/count for {task}")
        for arm in ARMS:
            source_payload = None if arm == "lifted_lr_scratch" else source_payloads[arm]
            report["tasks"][task][arm] = train_h3_arm(
                task, arm, source_payload, support, validation_cases, model_config,
                seed=42, device=device, support_pair_count=len(task_index["support_pair_ids"]),
            )
            write_json("artifacts/reports/ns_pimple_method_v1/h3_lr_diagnostic.json", report)
    write_json("artifacts/reports/ns_pimple_method_v1/h3_lr_diagnostic.json", report)
    summary = {}
    for prefix, label in (("F5_", "development"), ("F6_", "confirmation")):
        summary[label] = {}
        for arm in ARMS:
            values = [report["tasks"][task][arm]["best_lr_validation_rollout_rel_l2"]
                      for task in method.TASKS if task.startswith(prefix)]
            summary[label][arm] = float(np.mean(values))
    report["split_mean_lr_validation_rollout_rel_l2"] = summary
    write_json("artifacts/reports/ns_pimple_method_v1/h3_lr_diagnostic.json", report)
    print(json.dumps({"status": "H3_LR_diagnostic_complete", "means": summary,
                      "target_hr_query_used": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
