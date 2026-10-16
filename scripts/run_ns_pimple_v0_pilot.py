from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = ROOT.parent / "NS-PIMPLE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NS_ROOT / "src"))

from ns_pimple_dataset.reader import NSPimpleDataset  # noqa: E402
from blrpng.config import load_config  # noqa: E402
from blrpng.models.fno import build_model  # noqa: E402
from blrpng.paths import ensure_artifact_path, sha256_file, write_json  # noqa: E402
from blrpng.training.engine import _train_batch  # noqa: E402
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256  # noqa: E402


PROTOCOL_PATH = "docs/prereg/ns_pimple_v0_fv_pilot_v2.json"
RESULT_DIR = "artifacts/reports/ns_pimple_v0_pilot"
CHECKPOINT_DIR = "artifacts/checkpoints/ns_pimple_v0_pilot"
TARGET_PAIR_ID = "F4_poly24_top_a060_ic_a"
FIELD_COUNT = 2
CONDITION_COUNT = 8


class TimeDependentLiftedOperator(torch.nn.Module):
    """BL-RPNG operator in homogeneous coordinates for moving wall data."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.config = base.config
        self.representation = base.representation

    def forward(self, x, boundary_lift, boundary_mask):
        input_lift_prediction, probes = self.base(x, boundary_lift, boundary_mask)
        fields = int(self.config.out_channels)
        # x = [current physical state, grid, history, current lift,
        #      next lift, viscosity, dt]. The base subtracts/re-adds the
        # current lift; shift that reconstruction to the target-time lift.
        next_lift_start = fields + 2 + 2 + fields
        next_lift = x[:, next_lift_start : next_lift_start + fields]
        return input_lift_prediction - boundary_lift + next_lift, probes

    def path_active_parameters(self, layer: int):
        return self.base.path_active_parameters(layer)

    def configure_adaptation(self, *args, **kwargs):
        return self.base.configure_adaptation(*args, **kwargs)


def _read_boundary_ranges(h5: h5py.File) -> dict[str, Any]:
    raw = h5["grid/boundary_face_ranges_json"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


@lru_cache(maxsize=4)
def _fv_laplace_system(size: int):
    """Cell-centred FV Laplacian with half-cell Dirichlet face distances."""
    rows, cols, values = [], [], []
    for j in range(size):
        for i in range(size):
            row = j * size + i
            diagonal = 0.0
            for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                jj, ii = j + dj, i + di
                if 0 <= jj < size and 0 <= ii < size:
                    diagonal += 1.0
                    rows.append(row)
                    cols.append(jj * size + ii)
                    values.append(-1.0)
                else:
                    diagonal += 2.0
            rows.append(row)
            cols.append(row)
            values.append(diagonal)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(size * size, size * size))
    return matrix, spla.factorized(matrix.tocsc())


def _finite_volume_lift(face_values: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    size = int(face_values["left"].shape[0])
    matrix, solve = _fv_laplace_system(size)
    rhs = np.zeros((FIELD_COUNT, size, size), dtype=np.float64)
    rhs[:, :, 0] += 2.0 * face_values["left"].T
    rhs[:, :, -1] += 2.0 * face_values["right"].T
    rhs[:, 0, :] += 2.0 * face_values["bottom"].T
    rhs[:, -1, :] += 2.0 * face_values["top"].T
    lift = np.stack([solve(channel.reshape(-1)).reshape(size, size) for channel in rhs])
    residual = max(float(np.max(np.abs(matrix @ lift[c].reshape(-1) - rhs[c].reshape(-1))))
                   for c in range(FIELD_COUNT))
    return lift.astype(np.float32), residual


def prepare_case(dataset: NSPimpleDataset, pair: dict, level: str, *, training: bool) -> dict[str, Any]:
    sample = (
        dataset.training_pair(pair["pair_id"])
        if training
        else dataset.load(pair["pair_id"], level=level, purpose="audit")
    )
    case_path = NS_ROOT / pair["cases"][level]["hdf5"]
    with h5py.File(case_path, "r") as h5:
        times = h5["time/physical_time"][:]
        u_cells = h5["fields/u"][:]
        previous_cells = h5["physical_previous/u"][:]
        solver_initial_cells = h5["solver_initial/u"][:]
        path_cells = h5["path/u"][:]
        path_mask = h5["path/valid_mask"][:].astype(bool)
        x_centres = h5["grid/x"][:]
        y_centres = h5["grid/y"][:]
        face_centres = h5["grid/face_centers"][:]
        face_ranges = _read_boundary_ranges(h5)
        re_value = float(h5.attrs["re"])
        nu_value = float(h5.attrs["nu"])
        pressure_norm = str(h5.attrs["pressure_normalization"])

    face_series = dataset._boundary_series(
        sample["metadata"]["boundary"], times, x_centres, y_centres,
        face_centres, face_ranges,
    )
    def face_at(index: int) -> dict[str, np.ndarray]:
        result = {}
        for key in ("left", "right", "bottom", "top"):
            spec = face_ranges[key]
            start, count = int(spec["startFace"]), int(spec["nFaces"])
            centers = face_centres[start : start + count]
            tangent_axis = 1 if key in ("left", "right") else 0
            order = np.argsort(centers[:, tangent_axis])
            result[key] = face_series[key]["velocity"][index][order]
        return result
    lifts, lift_residuals = zip(*[_finite_volume_lift(face_at(index)) for index in range(len(times))])
    lifts = np.asarray(lifts, dtype=np.float32)
    first_corrections = []
    endpoint_increments = []
    path_counts = path_mask.sum(axis=1)
    for index in range(len(solver_initial_cells)):
        first_index = int(np.flatnonzero(path_mask[index])[0])
        first_corrections.append(path_cells[index, first_index] - solver_initial_cells[index])
        endpoint_increments.append(u_cells[index + 1] - solver_initial_cells[index])
    first_corrections = np.asarray(first_corrections, dtype=np.float32)
    endpoint_increments = np.asarray(endpoint_increments, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(x_centres, y_centres, indexing="xy")
    grid = np.stack((grid_x, grid_y)).astype(np.float32)
    steps = len(times) - 1
    xs, ys, lifts_for_model, masks = [], [], [], []
    boundary_mask = np.zeros((FIELD_COUNT, len(y_centres), len(x_centres)), dtype=bool)
    for index in range(steps):
        history = previous_cells[index].astype(np.float32)
        condition = np.concatenate((
            history,
            lifts[index],
            lifts[index + 1],
            np.full((1, len(y_centres), len(x_centres)), nu_value, dtype=np.float32),
            np.full((1, len(y_centres), len(x_centres)), sample["dt"][index], dtype=np.float32),
        ), axis=0)
        xs.append(np.concatenate((u_cells[index].astype(np.float32), grid, condition), axis=0))
        ys.append(u_cells[index + 1].astype(np.float32))
        lifts_for_model.append(lifts[index])
        masks.append(boundary_mask)

    diagnostics = sample["trajectory"]["diagnostics"]
    d_first = np.asarray(diagnostics["D"][:, 0], dtype=np.float64)
    s_first = np.asarray(diagnostics["S"][:, 0], dtype=np.float64)
    lift_residual_max = max(float(value) for value in lift_residuals)
    labels_finite = bool(np.isfinite(first_corrections).all() and np.isfinite(endpoint_increments).all())

    tensors = {
        "x": torch.from_numpy(np.asarray(xs, dtype=np.float32)),
        "y": torch.from_numpy(np.asarray(ys, dtype=np.float32)),
        "boundary_lift": torch.from_numpy(np.asarray(lifts_for_model, dtype=np.float32)),
        "boundary_mask": torch.from_numpy(np.asarray(masks, dtype=bool)),
        "newton_correction": torch.from_numpy(first_corrections),
        "endpoint_increment": torch.from_numpy(endpoint_increments),
        "valid_correction_mask": torch.ones((steps, 1), dtype=torch.bool),
        "field_scales": torch.ones((FIELD_COUNT,), dtype=torch.float32),
    }
    return {
        "pair_id": pair["pair_id"],
        "task_id": pair["task_id"],
        "role": pair["role"],
        "split": pair["split"],
        "level": level,
        "tensors": tensors,
        "u_cells": u_cells,
        "previous_cells": previous_cells,
        "lifts": lifts,
        "grid": grid,
        "time": times,
        "x_centres": x_centres,
        "y_centres": y_centres,
        "nu": nu_value,
        "re": re_value,
        "pressure_normalization": pressure_norm,
        "path_counts": path_counts.tolist(),
        "first_D_median": float(np.median(d_first)),
        "first_S_median": float(np.median(s_first)),
        "finite_volume_lift_residual_max": lift_residual_max,
        "native_cell_centres_preserved": bool(np.array_equal(tensors["y"].numpy(), u_cells[1:].astype(np.float32))),
        "path_labels_finite": labels_finite,
    }


def _batch(item: dict[str, torch.Tensor], indices: list[int], device: torch.device):
    index = torch.as_tensor(indices, dtype=torch.long)
    batch = {
        key: (value if key == "field_scales" else value.index_select(0, index))
        for key, value in item.items()
    }
    return {key: value.to(device) for key, value in batch.items()}


def _model_config(base: dict) -> dict:
    candidate = dict(base["model_search"]["candidates"]["small"])
    return {
        **candidate,
        "in_channels": 2 * FIELD_COUNT + 2 + CONDITION_COUNT,
        "out_channels": FIELD_COUNT,
        "condition_channels": CONDITION_COUNT,
        "probe_mode": "per_layer",
        "field_scales": [1.0, 1.0],
        "gauge_channels": [],
    }


def _training_config(base: dict, auxiliary: str | None) -> dict:
    training = dict(base["training"])
    training.update({
        "final_loss": "joint_relative_l2",
        "path_layer": 1,
        "path_gradient_ratio": 0.05,
        "path_log_magnitude_weight": 0.1,
        "path_epsilon": 1e-8,
        "vorticity_weight": 0.0,
        "divergence_weight": 0.0,
        "rollout_horizon": 1,
    })
    training.pop("path_anneal_start", None)
    training.pop("path_anneal_end", None)
    if auxiliary is not None:
        training["auxiliary_target"] = auxiliary
    return training


def train_source_arm(
    name: str,
    source_items: list[dict[str, torch.Tensor]],
    model_config: dict,
    base_config: dict,
    *,
    seed: int,
    device: torch.device,
    updates: int,
    batch_size: int,
) -> tuple[TimeDependentLiftedOperator, dict[str, Any]]:
    set_seed(seed, deterministic=True)
    base_model = build_model(model_config, representation="lifted")
    model = TimeDependentLiftedOperator(base_model).to(device)
    auxiliary = {
        "final_only": None,
        "first_pimple": "newton_correction",
        "matched_endpoint": "endpoint_direction_path_norm",
    }[name]
    objective = "final_only" if name == "final_only" else "blrpng_safe"
    training = _training_config(base_config, auxiliary)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
        weight_decay=1e-6,
    )
    sample_count = int(source_items[0]["x"].shape[0])
    # Identical deterministic minibatch order across the three source arms.
    generator = torch.Generator(device="cpu").manual_seed(271828)
    update_rows = []
    model.train()
    for update in range(updates):
        permutation = torch.randperm(sample_count, generator=generator)
        start = (update * batch_size) % sample_count
        indices = permutation[start : start + batch_size].tolist()
        if len(indices) < batch_size:
            indices += permutation[: batch_size - len(indices)].tolist()
        item_order = (update // max(1, sample_count // batch_size)) % len(source_items)
        item = source_items[item_order]
        batch = _batch(item, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _train_batch(
            model=model, batch=batch, objective=objective,
            training_config=training, epoch_progress=update / max(updates - 1, 1),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in source arm {name} at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 100 == 0 or update == updates - 1:
            row = {
                "update": update + 1,
                "total_loss": float(loss.detach().cpu()),
                "final_loss": float(metrics["final_loss"].detach().cpu()),
                "path_loss": float(metrics.get("path_loss", 0.0)),
                "gradient_ratio": float(metrics.get("gradient_ratio", 0.0)),
            }
            update_rows.append(row)
            print(f"source {name}: {row}", flush=True)
    model.eval()
    return model, {"objective": name, "optimizer_updates": updates, "samples_per_rollout": sample_count,
                   "training_log": update_rows}


def adapt_target(
    source_model: TimeDependentLiftedOperator,
    support: dict[str, torch.Tensor],
    model_config: dict,
    base_config: dict,
    *,
    seed: int,
    device: torch.device,
    updates: int,
    budget: int,
) -> tuple[TimeDependentLiftedOperator, dict[str, Any]]:
    set_seed(seed, deterministic=True)
    base_model = build_model(model_config, representation="lifted")
    base_model.load_state_dict(source_model.base.state_dict(), strict=True)
    base_model.configure_adaptation("nlt", rank=4, alpha=4.0)
    model = TimeDependentLiftedOperator(base_model).to(device)
    training = _training_config(base_config, None)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-6,
    )
    indices = list(range(min(budget, int(support["x"].shape[0]))))
    batch = _batch(support, indices, device)
    model.train()
    log = []
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _train_batch(
            model=model, batch=batch, objective="final_only",
            training_config=training, epoch_progress=1.0,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite target NLT loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 50 == 0 or update == updates - 1:
            row = {"update": update + 1, "final_loss": float(loss.detach().cpu())}
            log.append(row)
            print(f"target NLT: {row}", flush=True)
    model.eval()
    return model, {"adaptation": "final-only NLT", "support_budget_transitions": len(indices),
                   "optimizer_updates": updates, "training_log": log}


@torch.no_grad()
def evaluate_rollout(model: TimeDependentLiftedOperator, case: dict, device: torch.device) -> dict[str, Any]:
    fields = FIELD_COUNT
    grid = torch.from_numpy(case["grid"][None]).to(device)
    height, width = case["u_cells"].shape[-2:]
    boundary_mask = torch.zeros((1, fields, height, width), dtype=torch.bool, device=device)
    current = case["u_cells"][0].astype(np.float32)
    history = case["previous_cells"][0].astype(np.float32)
    step_errors = []
    model.eval()
    for step in range(len(case["time"]) - 1):
        condition = np.concatenate((
            history,
            case["lifts"][step],
            case["lifts"][step + 1],
            np.full((1, height, width), case["nu"], dtype=np.float32),
            np.full((1, height, width), case["time"][step + 1] - case["time"][step], dtype=np.float32),
        ), axis=0)
        x_np = np.concatenate((current, case["grid"], condition), axis=0)
        x = torch.from_numpy(x_np[None].astype(np.float32)).to(device)
        input_lift = torch.from_numpy(case["lifts"][step][None]).to(device)
        prediction, _ = model(x, input_lift, boundary_mask)
        prediction_np = prediction[0].detach().cpu().numpy()
        target = case["u_cells"][step + 1]
        step_errors.append(float(np.linalg.norm(prediction_np - target) / max(np.linalg.norm(target), 1e-12)))
        history = current
        current = prediction_np
    return {
        "pair_id": case["pair_id"],
        "resolution": [int(height), int(width)],
        "steps": len(step_errors),
        "native_cell_center_rollout_rel_l2_mean": float(np.mean(step_errors)),
        "native_cell_center_rollout_rel_l2_final": float(step_errors[-1]),
        "per_step_native_cell_center_rel_l2": step_errors,
    }


def _require_bridge_gate(cases: list[dict]) -> dict[str, Any]:
    maximum_lift_residual = max(case["finite_volume_lift_residual_max"] for case in cases)
    minimum_d = min(case["first_D_median"] for case in cases)
    minimum_s = min(case["first_S_median"] for case in cases)
    checks = {
        "native_cell_center_fields_preserved_without_resampling": all(case["native_cell_centres_preserved"] for case in cases),
        "finite_volume_harmonic_lift_residual_below_1e-10": maximum_lift_residual <= 1e-10,
        "every_case_first_hook_D_median_at_least_5pct": minimum_d >= 0.05,
        "every_case_first_hook_S_median_at_least_1pct": minimum_s >= 0.01,
        "first_path_and_matched_endpoint_labels_are_finite": all(case["path_labels_finite"] for case in cases),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "maximum_finite_volume_lift_residual": maximum_lift_residual,
        "minimum_first_hook_D_median": minimum_d,
        "minimum_first_hook_S_median": minimum_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-updates", type=int, default=600)
    parser.add_argument("--target-updates", type=int, default=200)
    parser.add_argument("--target-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_file = ROOT / PROTOCOL_PATH
    protocol_hash = sha256_file(protocol_file)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    dataset = NSPimpleDataset(NS_ROOT)
    source_support = [
        row for row in dataset.pairs(split="source", training_only=True)
        if not row["pair_id"].startswith("F4_")
    ]
    target_pair = dataset.by_id.get(TARGET_PAIR_ID)
    if target_pair is None or target_pair.get("training_access") != "source_support_lr_only":
        raise RuntimeError("The preregistered target support is missing or access-denied.")
    if len(source_support) != 6:
        raise RuntimeError(f"Expected six F1-F3 source-support rollouts, found {len(source_support)}")

    prepared_source = [prepare_case(dataset, row, "lr", training=True) for row in source_support]
    prepared_target = prepare_case(dataset, target_pair, "lr", training=True)
    bridge_gate = _require_bridge_gate([*prepared_source, prepared_target])
    gate_path = ensure_artifact_path(f"{RESULT_DIR}/method_bridge_gate.json")
    write_json(gate_path.relative_to(ROOT), {
        "schema": "blrpng_ns_pimple_method_bridge_gate_v1",
        "protocol_sha256": protocol_hash,
        "dataset_id": dataset.manifest["dataset_id"],
        "source_support_pair_ids": [row["pair_id"] for row in source_support],
        "target_support_pair_id": TARGET_PAIR_ID,
        "query_opened": False,
        "path_semantics": "first complete PIMPLE outer state minus solver initial; not Newton",
        "per_case": [{key: case[key] for key in (
            "pair_id", "level", "path_counts", "first_D_median", "first_S_median",
            "finite_volume_lift_residual_max", "native_cell_centres_preserved",
            "path_labels_finite",
        )} for case in [*prepared_source, prepared_target]],
        **bridge_gate,
    })
    if bridge_gate["status"] != "pass":
        print(json.dumps({"status": "method_bridge_gate_failed", "gate": bridge_gate}, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps({"status": "method_bridge_gate_passed", "query_opened": False,
                          "gate": bridge_gate, "report": str(gate_path)}, indent=2))
        return 0

    base_config = load_config("configs/navier_stokes_sfepy_seed42.yaml")
    model_config = _model_config(base_config)
    source_tensors = [case["tensors"] for case in prepared_source]
    arms = ("final_only", "first_pimple", "matched_endpoint")
    trained: dict[str, TimeDependentLiftedOperator] = {}
    arm_reports: dict[str, Any] = {}
    checkpoint_paths: dict[str, Path] = {}
    for arm in arms:
        model, report = train_source_arm(
            arm, source_tensors, model_config, base_config,
            seed=args.seed, device=device, updates=args.source_updates, batch_size=8,
        )
        model_hash = state_dict_sha256(model.base.state_dict())
        checkpoint = ensure_artifact_path(f"{CHECKPOINT_DIR}/source_{arm}_seed{args.seed}.pt")
        atomic_torch_save({
            "schema": "blrpng_ns_pimple_pilot_checkpoint_v1",
            "base_state_dict": model.base.state_dict(),
            "model_config": model_config,
            "representation": "lifted_homogeneous_coordinates_with_time_dependent_wall_lift",
            "source_arm": arm,
            "source_pair_ids": [row["pair_id"] for row in source_support],
            "state_dict_sha256": model_hash,
            "protocol_sha256": protocol_hash,
            "query_used": False,
        }, checkpoint)
        checkpoint_paths[arm] = checkpoint
        trained[arm] = model
        arm_reports[arm] = {**report, "checkpoint": str(checkpoint.relative_to(ROOT)),
                            "checkpoint_sha256": sha256_file(checkpoint),
                            "model_state_sha256": model_hash}

    adapted: dict[str, TimeDependentLiftedOperator] = {}
    for arm in arms:
        model, adaptation_report = adapt_target(
            trained[arm], prepared_target["tensors"], model_config,
            base_config,
            seed=args.seed, device=device, updates=args.target_updates,
            budget=args.target_budget,
        )
        model_hash = state_dict_sha256(model.base.state_dict())
        checkpoint = ensure_artifact_path(f"{CHECKPOINT_DIR}/adapted_{arm}_seed{args.seed}_m{args.target_budget}.pt")
        atomic_torch_save({
            "schema": "blrpng_ns_pimple_pilot_adapter_v1",
            "base_state_dict": model.base.state_dict(),
            "model_config": model_config,
            "representation": "lifted_homogeneous_coordinates_with_time_dependent_wall_lift",
            "source_arm": arm,
            "target_support_pair_id": TARGET_PAIR_ID,
            "target_budget_transitions": args.target_budget,
            "state_dict_sha256": model_hash,
            "protocol_sha256": protocol_hash,
            "query_used": False,
        }, checkpoint)
        adaptation_report.update({"checkpoint": str(checkpoint.relative_to(ROOT)),
                                  "checkpoint_sha256": sha256_file(checkpoint),
                                  "model_state_sha256": model_hash})
        arm_reports[arm]["target_adaptation"] = adaptation_report
        adapted[arm] = model

    frozen_payload = {
        "schema": "blrpng_ns_pimple_v0_pilot_checkpoint_freeze_v1",
        "protocol_sha256": protocol_hash,
        "query_used": False,
        "source_pair_ids": [row["pair_id"] for row in source_support],
        "target_support_pair_id": TARGET_PAIR_ID,
        "target_budget_transitions": args.target_budget,
        "arms": {arm: {
            "source_checkpoint_sha256": arm_reports[arm]["checkpoint_sha256"],
            "adapter_checkpoint_sha256": arm_reports[arm]["target_adaptation"]["checkpoint_sha256"],
        } for arm in arms},
    }
    freeze_path = ensure_artifact_path(f"{RESULT_DIR}/checkpoint_freeze.json")
    write_json(freeze_path.relative_to(ROOT), frozen_payload)

    # Open only source-split query-role HR trajectories, after the checkpoint freeze.
    query_pairs = [
        row for row in dataset.pairs(split="source")
        if row["role"] == "query" and row["task_id"] == target_pair["task_id"]
    ]
    if len(query_pairs) != 2:
        raise RuntimeError(f"Expected two held-out query ICs for the frozen target task, got {len(query_pairs)}")
    query_cases = [prepare_case(dataset, row, "hr", training=False) for row in query_pairs]
    query_results = {
        arm: [evaluate_rollout(adapted[arm], case, device) for case in query_cases]
        for arm in arms
    }
    summaries = {}
    for arm, rows in query_results.items():
        summaries[arm] = {
            "query_pair_ids": [row["pair_id"] for row in rows],
            "mean_native_cell_center_rollout_rel_l2": float(np.mean([row["native_cell_center_rollout_rel_l2_mean"] for row in rows])),
            "mean_final_native_cell_center_rel_l2": float(np.mean([row["native_cell_center_rollout_rel_l2_final"] for row in rows])),
            "per_query": rows,
        }
    report = {
        "schema": "blrpng_ns_pimple_v0_pilot_result_v1",
        "status": "pilot_complete",
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": protocol_hash,
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_gate_report": str((NS_ROOT / "runs/ns-method-review-20261015/dataset_audit.json")),
        "reader_gate_report": str((NS_ROOT / "runs/ns-method-review-20261015/interface_validation.json")),
        "method_bridge_gate": bridge_gate,
        "path_semantics": "PIMPLE outer-loop correction; this does not test Newton-path supervision.",
        "target_pair_id": TARGET_PAIR_ID,
        "target_task_id": target_pair["task_id"],
        "source_training_pair_ids": [row["pair_id"] for row in source_support],
        "target_budget_transitions": args.target_budget,
        "source_training_updates": args.source_updates,
        "target_adaptation_updates": args.target_updates,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_freeze": str(freeze_path.relative_to(ROOT)),
        "query_opened_after_checkpoint_freeze": True,
        "sealed_test_opened": False,
        "source_arms": arm_reports,
        "query_summary": summaries,
        "interpretation": [
            "A lower pilot error is a screening signal only; one fold, one task, and one seed are not confirmatory.",
            "The path arm uses the first actual PIMPLE outer correction, not a Newton correction.",
            "The pilot predicts velocity and does not test a pressure-output model.",
        ],
    }
    output = ensure_artifact_path(f"{RESULT_DIR}/result.json")
    write_json(output.relative_to(ROOT), report)
    print(json.dumps({"status": report["status"], "checkpoint_freeze": str(freeze_path),
                      "query_summary": summaries, "result": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
