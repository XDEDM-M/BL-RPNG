from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import materialize_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file, write_json


REPRESENTATIVE_TASKS = (0, 8, 16, 24, 32, 40)


def _families() -> dict[int, str]:
    payload = json.loads(resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    return {int(row["target_task_id"]): str(row["family"]) for row in payload["tasks"]}


@torch.no_grad()
def _representative(task: int, resolution: int, device: torch.device) -> dict:
    bundle = materialize_index(
        "artifacts/data/reconfirmation/"
        f"task{task:02d}_query_{resolution}_all.index.json"
    )
    checkpoint = resolve_in_project(
        "artifacts/checkpoints/reconfirmation/"
        f"reconfirmation_task{task:02d}_m4_s42_nlt.adapter.pt",
        must_exist=True,
    )
    model, _ = load_checkpoint_model(checkpoint, device=device)
    rollout_id = int(bundle.tensors["rollout_id"].min())
    indices = [
        index for index, value in enumerate(bundle.tensors["rollout_id"].tolist())
        if int(value) == rollout_id
    ]
    indices.sort(key=lambda index: int(bundle.tensors["local_step_id"][index]))
    indices = indices[:20]
    current = bundle.tensors["x"][indices[0]:indices[0] + 1].to(device)
    prediction = None
    for index in indices:
        lift = bundle.tensors["boundary_lift"][index:index + 1].to(device)
        mask = bundle.tensors["boundary_mask"][index:index + 1].to(device).bool()
        prediction, _ = model(current, lift, mask)
        current = torch.cat((
            prediction,
            bundle.tensors["x"][
                index:index + 1, model.config.out_channels:
            ].to(device),
        ), dim=1)
    assert prediction is not None
    target = bundle.tensors["y"][indices[-1]:indices[-1] + 1].to(device)
    mask = bundle.tensors["boundary_mask"][indices[-1]:indices[-1] + 1].to(device).bool()
    error = (prediction - target).abs()
    near = F.max_pool2d(mask.float(), kernel_size=9, stride=1, padding=4).bool()
    relative_l2 = float(
        (prediction - target).flatten().norm() / target.flatten().norm().clamp_min(1e-12)
    )
    result = {
        "task": task,
        "rollout_id": rollout_id,
        "initial_condition_id": int(bundle.tensors["initial_condition_id"][indices[0]]),
        "relative_l2_at_step20": relative_l2,
        "boundary_error_max": float((prediction[mask] - target[mask]).abs().max()),
        "target": target[0, 0].cpu().numpy(),
        "prediction": prediction[0, 0].cpu().numpy(),
        "error": error[0, 0].cpu().numpy(),
        "near_error": np.where(near[0, 0].cpu().numpy(), error[0, 0].cpu().numpy(), np.nan),
    }
    del model, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render pre-frozen representative blind-query fields."
    )
    parser.add_argument("--resolution", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    resolution = int(args.resolution)
    families = _families()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        _representative(task, resolution, device)
        for task in REPRESENTATIVE_TASKS
    ]
    figure, axes = plt.subplots(len(rows), 4, figsize=(12.0, 16.0), constrained_layout=True)
    column_titles = ("Ground truth", "BL-RPNG-NLT", "Absolute error", "Near-boundary error")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    records = []
    for row_index, row in enumerate(rows):
        state_min = min(float(np.nanmin(row["target"])), float(np.nanmin(row["prediction"])))
        state_max = max(float(np.nanmax(row["target"])), float(np.nanmax(row["prediction"])))
        error_max = max(float(np.nanmax(row["error"])), 1e-12)
        images = (
            axes[row_index, 0].imshow(row["target"], origin="lower", cmap="coolwarm", vmin=state_min, vmax=state_max),
            axes[row_index, 1].imshow(row["prediction"], origin="lower", cmap="coolwarm", vmin=state_min, vmax=state_max),
            axes[row_index, 2].imshow(row["error"], origin="lower", cmap="magma", vmin=0.0, vmax=error_max),
            axes[row_index, 3].imshow(row["near_error"], origin="lower", cmap="magma", vmin=0.0, vmax=error_max),
        )
        axes[row_index, 0].set_ylabel(
            families[row["task"]].replace("_", " ")
            + f"\nrelL2={100 * row['relative_l2_at_step20']:.2f}%"
        )
        for column, image in enumerate(images):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            figure.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.02)
        records.append({
            "task": row["task"],
            "family": families[row["task"]],
            "rollout_id": row["rollout_id"],
            "initial_condition_id": row["initial_condition_id"],
            "relative_l2_at_step20": row["relative_l2_at_step20"],
            "boundary_error_max": row["boundary_error_max"],
            "selection_rule": "lowest task ID in family and lowest frozen query rollout ID",
        })
    outputs = []
    with workspace_audit(f"visualize_reconfirmation_fields_{resolution}") as audit:
        for suffix in ("png", "pdf", "svg"):
            path = ensure_artifact_path(
                "artifacts/figures/"
                f"allen_cahn_reconfirmation_query{resolution}_fields.{suffix}"
            )
            figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            outputs.append({
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            })
        plt.close(figure)
        metadata = write_json(
            "artifacts/figures/"
            f"allen_cahn_reconfirmation_query{resolution}_fields.json",
            {
                "schema": "blrpng_frozen_field_figure_v1",
                "selection_performed": False,
                "resolution": resolution,
                "horizon": 20,
                "records": records,
                "figures": outputs,
            },
        )
        audit.update({"metadata": str(metadata), "figures": len(outputs)})
    print(json.dumps({"metadata": str(metadata), "figures": outputs}, indent=2))


if __name__ == "__main__":
    main()
