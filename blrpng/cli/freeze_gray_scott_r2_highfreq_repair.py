from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    stage_path = resolve_in_project(
        "artifacts/manifests/gray_scott_r2_seed42_stage.json", must_exist=True
    )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if stage.get("passed") is not False or stage.get("degraded_boundary_families") != ["high_frequency"]:
        raise PermissionError("The frozen R2 failure does not authorize this focused repair.")
    source = resolve_in_project(
        "artifacts/checkpoints/gray_scott_r2/"
        "gray_scott_r2_source_s42_blrpng_safe_annealed_r005.inference.pt",
        must_exist=True,
    )
    payload = {
        "schema": "blrpng_gray_scott_r2_highfreq_repair_v1",
        "frozen": True,
        "passed": True,
        "exploratory_after_query": True,
        "task": 0,
        "mechanism": (
            "Combine diagonal NLT with rank-4 spectral LoRA so the frozen "
            "Newton-path feature basis can alter its Fourier transfer response."
        ),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "adapter": {"strategy": "nlt_lora", "rank": 4, "alpha": 4.0},
        "objective": "final_only",
        "training": {
            "epochs": 100,
            "minimum_epochs": 60,
            "early_stopping_patience": 15,
            "batch_size": 4,
            "lr": 0.001,
            "lr_drop_epoch": 90,
            "validation": "64-grid/50-step only",
        },
        "acceptance": {
            "must_beat_frozen_path_nlt_lr_validation": True,
            "current_query_prohibited_for_training_or_selection": True,
            "successful_repair_requires_a_second_fresh_query": True,
        },
        "additional_adapter_or_rank_sweeps_allowed": False,
    }
    with workspace_audit("freeze_gray_scott_r2_highfreq_repair") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_r2_highfreq_repair.json", payload
        )
        audit.update({"output": str(output), "exploratory": True})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
