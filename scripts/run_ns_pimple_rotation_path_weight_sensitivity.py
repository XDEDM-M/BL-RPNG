#!/usr/bin/env python3
"""LR-only 0.20/0.50 path-gradient sensitivity for rotated NS PIMPLE transfer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

import run_ns_pimple_method_v1 as method
import run_ns_pimple_rotation_source_lr_diagnostic as rotation
import run_ns_pimple_rotation_target_path_lr_diagnostic as rotation_target_path
import run_ns_pimple_target_path_lr_diagnostic as target_path
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256


ROOT = method.ROOT
PREREG = ROOT / "docs/prereg/ns_pimple_method_v1_rotation_path_weight_sensitivity.json"
REPORT_PATH = method.REPORT_ROOT / "rotation_path_weight_sensitivity.json"
DEV_TASKS = tuple(task for task in method.TASKS if task.startswith("F5_"))
RATIOS = (0.20, 0.50)
OBJECTIVES = ("selected_pimple_path", "matched_endpoint_control")


def train_target(task: str, objective_name: str, ratio: float, source: dict,
                 support: dict[str, torch.Tensor], validation: list[dict],
                 model_config: dict, *, device: torch.device) -> dict:
    ratio_index = RATIOS.index(ratio)
    objective_index = OBJECTIVES.index(objective_name)
    seed = 8200 + method.TASKS.index(task) * 100 + ratio_index * 10 + objective_index
    set_seed(seed, deterministic=True)
    base = method.build_model(model_config, representation="lifted")
    base.load_state_dict(source["base_state_dict"], strict=True)
    adapter_info = base.configure_adaptation("nlt", rank=4, alpha=4.0)
    model = method.TimeDependentLiftedOperator(base).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-6)
    training = dict(target_path.TRAINING)
    training["path_gradient_ratio"] = ratio
    training["auxiliary_target"] = (
        "newton_correction" if objective_name == "selected_pimple_path"
        else "endpoint_direction_path_norm"
    )
    count, batch_size = int(support["x"].shape[0]), 4
    steps_per_epoch = math.ceil(count / batch_size)
    generator = torch.Generator(device="cpu").manual_seed(271828)
    ratio_key = f"r{int(round(ratio * 100)):03d}"
    output_path = ensure_artifact_path(
        "artifacts/checkpoints/ns_pimple_method_v1/rotation_path_weight_sensitivity/"
        f"target_{task}_{objective_name}_{ratio_key}_seed42.pt"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite sensitivity checkpoint: {output_path}")
    best_score, best_epoch, stale = float("inf"), 0, 0
    best_rows, history = None, []
    for epoch in range(1, 101):
        model.train()
        permutation = torch.randperm(count, generator=generator)
        losses, last_metrics = [], {}
        for start in range(0, count, batch_size):
            ids = permutation[start:start + batch_size].tolist()
            batch = method._batch(support, ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss, last_metrics = method._train_batch(
                model=model, batch=batch, objective="blrpng_safe",
                training_config=training, epoch_progress=1.0,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {task}/{objective_name}/{ratio}/{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        score, validation_rows = method.mean_rollout_error(model, validation, device)
        history.append({
            "epoch": epoch,
            "optimizer_updates": epoch * steps_per_epoch,
            "train_loss_mean": float(np.mean(losses)),
            "lr_validation_rollout_rel_l2": score,
            "path_loss_last_batch": float(last_metrics.get("path_loss", 0.0)),
            "path_gradient_ratio_last_batch": float(last_metrics.get("gradient_ratio", 0.0)),
            "path_gradient_cosine_last_batch": float(last_metrics.get("gradient_cosine", 0.0)),
        })
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            digest = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_rotation_path_weight_checkpoint_v1",
                "task": task, "source_arm": "rotation_selected_pimple_path",
                "target_objective": objective_name, "path_gradient_ratio": ratio,
                "seed": seed, "epoch": epoch, "adapter": "nlt",
                "base_state_dict": model.base.state_dict(), "model_config": model_config,
                "model_state_sha256": digest,
                "source_checkpoint_sha256": source["model_state_sha256"],
                "selection_data": "six LR validation trajectories only",
                "target_hr_query_used": False,
            }, output_path)
            best_rows = validation_rows
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"rot-weight {task} {objective_name} ratio={ratio} epoch={epoch} "
                  f"LR-val={score:.5g} best={best_score:.5g}@{best_epoch}", flush=True)
        if epoch >= 30 and stale >= 12:
            break
    if best_epoch == 0 or best_rows is None:
        raise RuntimeError(f"no LR-validation checkpoint selected: {task}/{objective_name}/{ratio}")
    return {
        "task": task, "source_arm": "rotation_selected_pimple_path",
        "target_objective": objective_name, "path_gradient_ratio": ratio,
        "best_epoch": best_epoch, "epochs_run": history[-1]["epoch"],
        "optimizer_updates": history[-1]["epoch"] * steps_per_epoch,
        "best_lr_validation_rollout_rel_l2": best_score,
        "validation_per_rollout": best_rows,
        "support_transition_count": count,
        "trainable_parameters": int(adapter_info["trainable_parameters"]),
        "checkpoint": str(output_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(output_path),
        "target_hr_query_used": False, "training_history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device is unavailable: {device}")
    prereg = method.load_json(PREREG)
    if prereg.get("status") != "registered_before_weight_sensitivity_training":
        raise ValueError("weight-sensitivity experiment is not preregistered")
    model_base = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = method.build_model_config(model_base)
    support, _, support_ids = target_path.prepare_support_data()
    validation, validation_ids, _ = rotation.prepare_validation(model_config)
    source_report, source = rotation_target_path.source_payload()
    rotation_report = method.load_json(method.REPORT_ROOT / "rotation_source_lr_diagnostic.json")
    final_report = method.load_json(method.REPORT_ROOT / "rotation_target_path_lr_diagnostic.json")
    report = {
        "schema": "blrpng_ns_pimple_rotation_path_weight_sensitivity_v1",
        "status": "training_lr_only",
        "device": str(device), "target_hr_query_used": False,
        "sealed_f6_hr_used_for_training_or_selection": False,
        "preregistration_sha256": sha256_file(PREREG),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_checkpoint_sha256": source_report["source_arms"]["selected_pimple_path"]["checkpoint_sha256"],
        "support_pair_ids": {task: support_ids[task] for task in DEV_TASKS},
        "validation_pair_ids": {task: validation_ids[task] for task in DEV_TASKS},
        "existing_final_only_baselines": {},
        "h6_default_weight_result_sha256": sha256_file(method.REPORT_ROOT / "rotation_target_path_f5_dev_result.json"),
        "tasks": {},
    }
    if REPORT_PATH.exists():
        prior = method.load_json(REPORT_PATH)
        if prior.get("preregistration_sha256") != report["preregistration_sha256"]:
            raise ValueError("existing sensitivity report uses a different preregistration")
        report = prior
    for task in DEV_TASKS:
        baseline = rotation_report["target_tasks"][task]["selected_pimple_path"]
        if sha256_file(ROOT / baseline["checkpoint"]) != baseline["checkpoint_sha256"]:
            raise ValueError(f"final-only baseline hash mismatch: {task}")
        report["existing_final_only_baselines"][task] = baseline
        report["tasks"].setdefault(task, {})
        for ratio in RATIOS:
            ratio_key = f"r{int(round(ratio * 100)):03d}"
            report["tasks"][task].setdefault(ratio_key, {})
            for objective in OBJECTIVES:
                prior = report["tasks"][task][ratio_key].get(objective)
                if prior:
                    path = ROOT / prior["checkpoint"]
                    if not path.is_file() or sha256_file(path) != prior["checkpoint_sha256"]:
                        raise ValueError(f"sensitivity checkpoint hash mismatch: {task}/{ratio_key}/{objective}")
                    continue
                row = train_target(task, objective, ratio, source, support[task], validation[task],
                                   model_config, device=device)
                report["tasks"][task][ratio_key][objective] = row
                write_json("artifacts/reports/ns_pimple_method_v1/rotation_path_weight_sensitivity.json", report)
    report["f5_lr_mean_by_arm"] = {}
    for ratio in RATIOS:
        ratio_key = f"r{int(round(ratio * 100)):03d}"
        report["f5_lr_mean_by_arm"][ratio_key] = {
            arm: float(np.mean([report["tasks"][task][ratio_key][arm][
                "best_lr_validation_rollout_rel_l2"] for task in DEV_TASKS]))
            for arm in OBJECTIVES
        }
        report["f5_lr_mean_by_arm"][ratio_key]["final_only"] = float(np.mean([
            report["existing_final_only_baselines"][task]["best_lr_validation_rollout_rel_l2"]
            for task in DEV_TASKS
        ]))
    report["status"] = "lr_checkpoints_frozen_before_f5_hr_development_open"
    report["f5_hr_development_opened"] = False
    write_json("artifacts/reports/ns_pimple_method_v1/rotation_path_weight_sensitivity.json", report)
    print(json.dumps(report["f5_lr_mean_by_arm"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
