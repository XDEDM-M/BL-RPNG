#!/usr/bin/env python3
"""Factorial LR-only PIMPLE path target-adaptation diagnostic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import yaml

import run_ns_pimple_method_v1 as method
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256


ROOT = method.ROOT
PREREG = ROOT / "docs/prereg/ns_pimple_method_v1_target_path_lr_diagnostic.json"
REPORT_PATH = method.REPORT_ROOT / "target_path_lr_diagnostic.json"
SOURCE_ARMS = method.SOURCE_ARMS
TARGET_OBJECTIVES = ("final_only", "selected_pimple_path", "matched_endpoint_control")
TRAINING = {
    "final_loss": "joint_relative_l2",
    "rollout_horizon": 1,
    "path_layer": 1,
    "path_gradient_ratio": 0.05,
    "path_log_magnitude_weight": 0.1,
    "path_epsilon": 1e-8,
    "auxiliary_target": "newton_correction",
}


def prepare_support_data() -> tuple[dict[str, dict], dict[str, list[dict]], dict]:
    """Prepare only LR support and LR validation; never dereference target HR rows."""
    dataset = method.NSPimpleDataset(method.NS_ROOT)
    v1 = method.load_json(method.NS_ROOT / "data/method_v1/manifest.json")
    generated = method.with_external_rows(dataset, v1)
    selected = method.select_rows(dataset, generated)
    support_by_task: dict[str, dict] = {}
    validations: dict[str, list[dict]] = {}
    support_ids: dict[str, list[str]] = {}
    for task in method.TASKS:
        roles = selected["target"][task]
        cases = [method.prepare_case(dataset, row, "lr", training=False) for row in roles["support"]]
        if any(case["role"] != "support" or case["level"] != "lr" for case in cases):
            raise RuntimeError(f"non-support or non-LR data entered target support for {task}")
        if len(cases) != 4 or any(len(case["time"]) != 21 for case in cases):
            raise RuntimeError(f"unexpected support count/length for {task}")
        support_by_task[task] = method.merge_tensors(cases, include_path=True)
        validations[task] = [
            method.rollout_payload(method.prepare_case(dataset, row, "lr", training=False))
            for row in roles["validation"]
        ]
        support_ids[task] = [case["pair_id"] for case in cases]
        if len(validations[task]) != 4:
            raise RuntimeError(f"unexpected LR validation count for {task}")
    return support_by_task, validations, support_ids


def source_payloads() -> dict[str, dict]:
    report = method.load_json(method.REPORT_ROOT / "source_training.json")
    output = {}
    for arm in SOURCE_ARMS:
        record = report["arms"][arm]
        payload = torch.load(ROOT / record["checkpoint"], map_location="cpu", weights_only=False)
        if payload.get("query_used") is not False or payload.get("arm") != arm:
            raise RuntimeError(f"source checkpoint audit failed for {arm}")
        if record["checkpoint_sha256"] != sha256_file(ROOT / record["checkpoint"]):
            raise RuntimeError(f"source checkpoint hash changed for {arm}")
        output[arm] = payload
    return output


def train_target(
    task: str,
    source_arm: str,
    target_objective: str,
    source: dict,
    support: dict[str, torch.Tensor],
    validation_cases: list[dict],
    model_config: dict,
    *,
    device: torch.device,
) -> dict:
    set_seed(4200 + method.TASKS.index(task) * 10 + SOURCE_ARMS.index(source_arm) * 3
             + TARGET_OBJECTIVES.index(target_objective), deterministic=True)
    base = method.build_model(model_config, representation="lifted")
    base.load_state_dict(source["base_state_dict"], strict=True)
    adapter_info = base.configure_adaptation("nlt", rank=4, alpha=4.0)
    model = method.TimeDependentLiftedOperator(base).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-6)
    objective = "final_only" if target_objective == "final_only" else "blrpng_safe"
    training = dict(TRAINING)
    training["auxiliary_target"] = (
        "newton_correction" if target_objective == "selected_pimple_path"
        else "endpoint_direction_path_norm"
    )
    count = int(support["x"].shape[0])
    batch_size = 4
    steps_per_epoch = math.ceil(count / batch_size)
    generator = torch.Generator(device="cpu").manual_seed(271828)
    checkpoint_path = ensure_artifact_path(
        "artifacts/checkpoints/ns_pimple_method_v1/target_path_lr_diagnostic/"
        f"target_{task}_{source_arm}_to_{target_objective}_seed42.pt"
    )
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic checkpoint: {checkpoint_path}")
    best_score, best_epoch, stale = float("inf"), 0, 0
    best_rows, history = None, []
    for epoch in range(1, 101):
        model.train()
        permutation = torch.randperm(count, generator=generator)
        losses = []
        last_metrics: dict = {}
        for start in range(0, count, batch_size):
            indices = permutation[start:start + batch_size].tolist()
            batch = method._batch(support, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, last_metrics = method._train_batch(
                model=model,
                batch=batch,
                objective=objective,
                training_config=training,
                epoch_progress=1.0,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {task}/{source_arm}/{target_objective}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        score, validation_rows = method.mean_rollout_error(model, validation_cases, device)
        row = {
            "epoch": epoch,
            "optimizer_updates": epoch * steps_per_epoch,
            "train_loss_mean": float(np.mean(losses)),
            "lr_validation_rollout_rel_l2": score,
            "path_loss_last_batch": float(last_metrics.get("path_loss", 0.0)),
            "path_gradient_ratio_last_batch": float(last_metrics.get("gradient_ratio", 0.0)),
            "path_gradient_cosine_last_batch": float(last_metrics.get("gradient_cosine", 0.0)),
            "path_gate_active_last_batch": float(last_metrics.get("path_gate_active", 0.0)),
        }
        history.append(row)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            digest = state_dict_sha256(model.base.state_dict())
            atomic_torch_save({
                "schema": "blrpng_ns_pimple_target_path_lr_diagnostic_checkpoint_v1",
                "task": task,
                "source_arm": source_arm,
                "target_objective": target_objective,
                "seed": 42,
                "epoch": epoch,
                "adapter": "nlt",
                "base_state_dict": model.base.state_dict(),
                "model_config": model_config,
                "model_state_sha256": digest,
                "source_checkpoint_sha256": source["model_state_sha256"],
                "selection_data": "LR validation only",
                "target_hr_query_used": False,
            }, checkpoint_path)
            best_rows = validation_rows
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"target-path {task} {source_arm}->{target_objective} epoch={epoch} "
                  f"train={row['train_loss_mean']:.5g} LR-val={score:.5g} "
                  f"best={best_score:.5g}@{best_epoch}", flush=True)
        if epoch >= 30 and stale >= 12:
            break
    if best_epoch == 0 or best_rows is None:
        raise RuntimeError(f"no LR-validation checkpoint selected for {task}/{source_arm}/{target_objective}")
    return {
        "task": task,
        "source_arm": source_arm,
        "target_objective": target_objective,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "optimizer_updates": history[-1]["epoch"] * steps_per_epoch,
        "best_lr_validation_rollout_rel_l2": best_score,
        "validation_per_rollout": best_rows,
        "support_trajectory_count": 4,
        "support_transition_count": count,
        "trainable_parameters": int(adapter_info["trainable_parameters"]),
        "target_path_used": target_objective != "final_only",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history": history,
    }


def main() -> int:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=str(device))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device is unavailable: {device}")
    model_base = yaml.safe_load((ROOT / "configs/navier_stokes_sfepy_seed42.yaml").read_text())
    model_config = method.build_model_config(model_base)
    support_data, validation_cases, support_ids = prepare_support_data()
    sources = source_payloads()
    report = {
        "schema": "blrpng_ns_pimple_target_path_lr_diagnostic_v1",
        "status": "exploratory_lr_only_followup",
        "device": str(device),
        "seed_rule": "deterministic task x source-arm x target-objective seed offsets",
        "target_hr_query_used": False,
        "sealed_hr_claim": False,
        "selection_metric": "mean 20-step LR autoregressive relative-L2",
        "preregistration_sha256": sha256_file(PREREG),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "method_runner_sha256": sha256_file(ROOT / "scripts/run_ns_pimple_method_v1.py"),
        "engine_sha256": sha256_file(ROOT / "blrpng/training/engine.py"),
        "data_index_sha256": sha256_file(method.DATA_ROOT / "dataset_index.json"),
        "support_pair_ids": support_ids,
        "source_checkpoint_sha256": {arm: payload["model_state_sha256"] for arm, payload in sources.items()},
        "tasks": {},
    }
    if REPORT_PATH.exists():
        prior = method.load_json(REPORT_PATH)
        if (prior.get("schema") != report["schema"]
                or prior.get("preregistration_sha256") != report["preregistration_sha256"]
                or prior.get("data_index_sha256") != report["data_index_sha256"]):
            raise ValueError("existing target-path LR report has a different frozen protocol or data index")
        report = prior
    write_json("artifacts/reports/ns_pimple_method_v1/target_path_lr_diagnostic.json", report)
    for task in method.TASKS:
        report["tasks"].setdefault(task, {})
        for source_arm in SOURCE_ARMS:
            report["tasks"][task].setdefault(source_arm, {})
            for target_objective in TARGET_OBJECTIVES:
                prior_row = report["tasks"][task][source_arm].get(target_objective)
                if prior_row is not None:
                    checkpoint = ROOT / prior_row["checkpoint"]
                    if (not checkpoint.is_file()
                            or sha256_file(checkpoint) != prior_row["checkpoint_sha256"]):
                        raise ValueError(f"saved diagnostic checkpoint hash mismatch: {checkpoint}")
                    continue
                row = train_target(
                    task, source_arm, target_objective, sources[source_arm], support_data[task],
                    validation_cases[task], model_config, device=device,
                )
                report["tasks"][task][source_arm][target_objective] = row
                write_json("artifacts/reports/ns_pimple_method_v1/target_path_lr_diagnostic.json", report)
    split_means = {}
    for prefix, split in (("F5_", "development"), ("F6_", "confirmation")):
        split_means[split] = {}
        tasks = [task for task in method.TASKS if task.startswith(prefix)]
        for source_arm in SOURCE_ARMS:
            split_means[split][source_arm] = {}
            for target_objective in TARGET_OBJECTIVES:
                values = [report["tasks"][task][source_arm][target_objective][
                    "best_lr_validation_rollout_rel_l2"
                ] for task in tasks]
                split_means[split][source_arm][target_objective] = float(np.mean(values))
    report["split_mean_lr_validation_rollout_rel_l2"] = split_means
    write_json("artifacts/reports/ns_pimple_method_v1/target_path_lr_diagnostic.json", report)
    print(split_means)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
