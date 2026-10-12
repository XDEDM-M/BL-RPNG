from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from blrpng.models.fno import DirichletOperator
from blrpng.models.fno import build_model
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file
from blrpng.utils import atomic_torch_save


ADAPTER_FORMAT = "blrpng_adapter_delta_v1"


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in names
    }


def save_adapter_delta(
    path: str | Path,
    model: DirichletOperator,
    *,
    base_checkpoint: str | Path,
    adapter_config: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    base = resolve_in_project(base_checkpoint, must_exist=True)
    output = ensure_artifact_path(path)
    payload = {
        "format": ADAPTER_FORMAT,
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": sha256_file(base),
        "model_config": model.model_config_dict(),
        "representation": model.representation,
        "field_scales": list(model.field_scales),
        "gauge_channels": list(model.gauge_channels),
        "adapter_config": adapter_config,
        "adapter_state": trainable_state(model),
        "metadata": metadata,
    }
    atomic_torch_save(payload, output)
    return output


def load_adapter_delta(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[DirichletOperator, dict[str, Any]]:
    from blrpng.evaluation.checkpoint import load_checkpoint_model

    adapter_path = resolve_in_project(path, must_exist=True)
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if payload.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"Unsupported adapter format: {payload.get('format')}")
    base_path = resolve_in_project(payload["base_checkpoint"], must_exist=True)
    if sha256_file(base_path) != payload["base_checkpoint_sha256"]:
        raise ValueError("Base checkpoint changed after adapter creation.")
    model, _ = load_checkpoint_model(base_path, device=torch.device("cpu"))
    config = payload["adapter_config"]
    model.configure_adaptation(
        config["strategy"], rank=int(config.get("rank", 4)),
        alpha=float(config.get("alpha", 4.0)),
        train_last_blocks=int(config.get("train_last_blocks", 0)),
    )
    missing, unexpected = model.load_state_dict(payload["adapter_state"], strict=False)
    if unexpected:
        raise ValueError(f"Unexpected adapter keys: {unexpected}")
    adapter_names = set(payload["adapter_state"])
    if not adapter_names or any(name not in model.state_dict() for name in adapter_names):
        raise ValueError("Adapter state is empty or incompatible.")
    model.to(device).eval()
    return model, payload


def compact_task_checkpoint(
    checkpoint_path: str | Path,
    *,
    base_checkpoint: str | Path,
) -> Path:
    """Replace an adapter training checkpoint by a verified compact delta."""
    full_path = resolve_in_project(checkpoint_path, must_exist=True)
    checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
    adapter = dict(checkpoint.get("adapter", {}))
    if adapter.get("strategy") not in {"lora", "nlt", "nlt_mix", "nlt_lora"}:
        raise ValueError("Only LoRA/NLT checkpoints can be compacted against a base model.")
    model = build_model(checkpoint["model_config"], representation=checkpoint["representation"])
    model.configure_adaptation(
        adapter["strategy"], rank=int(adapter.get("rank", 4)),
        alpha=float(adapter.get("alpha", 4.0)),
        train_last_blocks=int(adapter.get("train_last_blocks", 0)),
    )
    model.load_state_dict(checkpoint["ema_weights"], strict=True)
    output = ensure_artifact_path(full_path.with_suffix(".adapter.pt"))
    save_adapter_delta(
        output, model, base_checkpoint=base_checkpoint,
        adapter_config=adapter,
        metadata={
            "objective": checkpoint["objective"],
            "training_seed": checkpoint["training_seed"],
            "best_validation_metric": checkpoint["best_validation_metric"],
            "elapsed_seconds": checkpoint["elapsed_seconds"],
            "gpu_hours": checkpoint.get("gpu_hours", checkpoint["elapsed_seconds"] / 3600.0),
            "energy_joules": checkpoint.get("energy_joules"),
            "peak_gpu_memory_bytes": checkpoint.get("peak_gpu_memory_bytes"),
            "optimizer_steps": checkpoint.get("optimizer_steps"),
            "stopped_early": checkpoint.get("stopped_early", False),
            "history": checkpoint["history"],
            "data_sha256": checkpoint["data_sha256"],
            "manifest_sha256": checkpoint["manifest_sha256"],
            "source_full_checkpoint_sha256": sha256_file(full_path),
            "adapter_initialization": checkpoint.get("adapter_initialization"),
        },
    )
    # Verify reconstruction before removing only the newly created redundant file.
    reconstructed, _ = load_adapter_delta(output, device=torch.device("cpu"))
    expected = model.state_dict()
    actual = reconstructed.state_dict()
    if expected.keys() != actual.keys() or any(
        not torch.allclose(expected[key], actual[key], atol=2e-6, rtol=1e-6)
        for key in expected
    ):
        output.unlink(missing_ok=True)
        raise RuntimeError("Adapter compaction verification failed.")
    full_path.unlink()
    # The engine keeps resumable snapshots in artifacts/checkpoints even when
    # the final checkpoint lives in a method subdirectory.
    recovery = ensure_artifact_path(
        f"artifacts/checkpoints/{full_path.stem}_recovery.pt"
    )
    recovery.unlink(missing_ok=True)
    return output


def compact_full_checkpoint_for_inference(path: str | Path) -> Path:
    """Retain one float32 EMA copy for a full/scratch model after training.

    Full checkpoints normally store independent raw and EMA tensors.  A
    target-only scratch baseline needs no further adaptation after its frozen
    validation selection, so both loader keys can safely reference the same
    selected EMA state.  Torch serialization preserves this alias and stores
    the tensor storage only once.
    """
    full_path = resolve_in_project(path, must_exist=True)
    checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "blrpng_checkpoint_v1":
        raise ValueError("Expected a full BL-RPNG checkpoint.")
    if checkpoint.get("adapter", {}).get("strategy") != "full":
        raise ValueError("Only a full-adaptation checkpoint can use full compaction.")
    selected = checkpoint["ema_weights"]
    checkpoint["raw_weights"] = selected
    checkpoint["ema_weights"] = selected
    checkpoint["inference_only"] = True
    output = ensure_artifact_path(full_path.with_suffix(".inference.pt"))
    atomic_torch_save(checkpoint, output)
    from blrpng.evaluation.checkpoint import load_checkpoint_model

    reconstructed, _ = load_checkpoint_model(output, device=torch.device("cpu"))
    actual = reconstructed.state_dict()
    if selected.keys() != actual.keys() or any(
        not torch.equal(selected[key], actual[key]) for key in selected
    ):
        output.unlink(missing_ok=True)
        raise RuntimeError("Full inference-checkpoint compaction verification failed.")
    full_path.unlink()
    recovery = ensure_artifact_path(
        f"artifacts/checkpoints/{full_path.stem}_recovery.pt"
    )
    recovery.unlink(missing_ok=True)
    return output
