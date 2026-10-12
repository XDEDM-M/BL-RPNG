from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.schema import load_dataset
from blrpng.models.fno import build_model
from blrpng.training.engine import (
    BundleDataset,
    _batch_to_device,
    _load_initialization,
    _train_batch,
)
from blrpng.paths import write_json
from blrpng.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--initialize-from")
    parser.add_argument("--max-batches", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    set_seed(seed, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_dataset(args.data, modes2=int(config["model"]["modes2"]))
    model = build_model(config["model"], representation="lifted")
    initialization = _load_initialization(
        model, args.initialize_from, representation="lifted"
    )
    model.to(device).train()
    loader = DataLoader(
        BundleDataset(bundle),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    rows = []
    with workspace_audit("diagnose_gradient_conflict"):
        for index, cpu_batch in enumerate(loader):
            if index >= args.max_batches:
                break
            _, metrics = _train_batch(
                model=model,
                batch=_batch_to_device(cpu_batch, device),
                objective="blrpng",
                training_config=config["training"],
            )
            rows.append(metrics)
        cosine = [row["gradient_cosine"] for row in rows]
        report = {
            "initialization": initialization,
            "batches": len(rows),
            "mean_gradient_cosine": sum(cosine) / len(cosine),
            "negative_gradient_fraction": sum(value < 0 for value in cosine)
            / len(cosine),
            "minimum_gradient_cosine": min(cosine),
            "maximum_gradient_cosine": max(cosine),
            "mean_path_weight": sum(row["path_weight"] for row in rows) / len(rows),
            "rows": rows,
        }
        write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
