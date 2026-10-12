from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from blrpng.models.fno import build_model
from blrpng.paths import resolve_in_project


def load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_in_project(path, must_exist=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") == "blrpng_adapter_delta_v1":
        from blrpng.training.adapters import load_adapter_delta
        return load_adapter_delta(checkpoint_path, device=device)
    if checkpoint.get("format") != "blrpng_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')}")
    model = build_model(
        checkpoint["model_config"],
        representation=checkpoint["representation"],
    )
    adapter = checkpoint.get("adapter", {})
    if adapter.get("strategy") in {"lora", "nlt", "nlt_mix"}:
        model.configure_adaptation(
            str(adapter["strategy"]),
            rank=int(adapter.get("rank", 4)),
            alpha=float(adapter.get("alpha", 4.0)),
            train_last_blocks=int(adapter.get("train_last_blocks", 0)),
        )
    model.load_state_dict(
        checkpoint["ema_weights"] if use_ema else checkpoint["raw_weights"],
        strict=True,
    )
    model.to(device).eval()
    return model, checkpoint
