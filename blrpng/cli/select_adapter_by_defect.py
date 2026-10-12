from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import resolve_in_project, write_json
from blrpng.training.physics import relative_defect_loss


def _checkpoint(task: int, method: str, budget: int, seed: int) -> str:
    return f"artifacts/checkpoints/v2/v2_task{task:02d}_m{budget}_s{seed}_{method}.adapter.pt"


@torch.no_grad()
def _score(checkpoint: str, index_path: str, resolution: int, device: torch.device) -> float:
    model, _ = load_checkpoint_model(checkpoint, device=device)
    index = read_index(index_path, verify_hashes=True)
    values = []
    axis = torch.linspace(0, 1, resolution, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((xx, yy)).unsqueeze(0)
    for row in index["shards"]:
        shard = torch.load(resolve_in_project(row["path"], must_exist=True),
                           map_location="cpu", weights_only=False)
        # Only the LR validation initial state and static boundary information
        # are read. No HR trajectory or target next state is available here.
        previous = F.interpolate(shard["trajectory"][0:1].to(device),
                                 size=(resolution, resolution), mode="bilinear", align_corners=True)
        lift = F.interpolate(shard["boundary_lift"].unsqueeze(0).to(device),
                             size=(resolution, resolution), mode="bilinear", align_corners=True)
        mask = F.interpolate(shard["constraint_mask"].float().unsqueeze(0).to(device),
                             size=(resolution, resolution), mode="nearest").bool()
        x = torch.cat((previous, grid), dim=1)
        prediction, _ = model(x, lift, mask)
        values.append(float(relative_defect_loss("allen_cahn", previous, prediction,
                                                 {"dt": .2, "eps": .03, "gamma": 5.0})))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select adapters using unlabeled fine-grid PDE defect.")
    parser.add_argument("--methods", default="lora,nlt,blrpng_v2,blrpng_plus_d1e2")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output", default="artifacts/reports/v2_unlabeled_defect_adapter_selection.json")
    args = parser.parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = {}
    with workspace_audit("select_adapter_by_defect"):
        for task in range(8):
            scores = {}
            index = f"artifacts/data/tasks/task{task:02d}_validation_lr_64_all.index.json"
            for method in methods:
                checkpoint = _checkpoint(task, method, args.budget, args.seed)
                scores[method] = _score(checkpoint, index, args.resolution, device)
            tasks[str(task)] = {"scores": scores, "selected": min(scores, key=scores.get)}
        output = write_json(args.output, {
            "schema": "blrpng_unlabeled_defect_selection_v1", "exploratory": True,
            "uses_hr_solution_labels": False, "selection_source": "LR validation initial states",
            "calibration_resolution": args.resolution, "methods": methods, "tasks": tasks,
        })
    print(json.dumps({"output": str(output),
                      "selections": {task: value["selected"] for task, value in tasks.items()}}, indent=2))


if __name__ == "__main__":
    main()
