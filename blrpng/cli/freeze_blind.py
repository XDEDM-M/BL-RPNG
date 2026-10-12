from __future__ import annotations

import argparse
import json
from pathlib import Path

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def _checkpoint_paths(experiment: str, seed: int, budgets: list[int]) -> list[str]:
    paths = [
        f"artifacts/checkpoints/{experiment}_a_final_only_seed{seed}.pt",
        f"artifacts/checkpoints/{experiment}_a_blrpng_seed{seed}.pt",
        f"artifacts/checkpoints/{experiment}_a_legacy_path_seed{seed}.pt",
        f"artifacts/checkpoints/{experiment}_a_dual_stage_seed{seed}.pt",
    ]
    for budget in budgets:
        for initialization in ("scratch", "final_pretrained", "blrpng_pretrained"):
            for objective in ("final_only", "blrpng"):
                paths.append(
                    f"artifacts/checkpoints/b{budget}_s{seed}_{initialization}_to_{objective}.pt"
                )
    paths.extend(
        [
            f"artifacts/checkpoints/ablation_b4_s{seed}_raw_scratch_final.pt",
            f"artifacts/checkpoints/ablation_b4_s{seed}_raw_legacy_path.pt",
            f"artifacts/checkpoints/ablation_b4_s{seed}_dual_stage.pt",
        ]
    )
    for budget in budgets:
        diagnostic = resolve_in_project(
            f"artifacts/checkpoints/ablation_b{budget}_s{seed}_blrpng_annealed.pt"
        )
        if diagnostic.exists():
            paths.append(str(diagnostic.relative_to(PROJECT_ROOT)))
    return paths


def _source_paths() -> list[Path]:
    allowed_suffixes = {".py", ".yaml", ".toml", ".md", ".sh"}
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and "artifacts" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix in allowed_suffixes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a seed before blind evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--output", default="artifacts/manifests/blind_evaluation_freeze.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]["name"]
    budgets = [int(value) for value in config["data"]["b_budgets"]]
    checkpoint_paths = _checkpoint_paths(experiment, args.seed, budgets)
    data_paths = [
        f"artifacts/data/{experiment}_{suffix}.pt"
        for suffix in (
            "a_train_fine",
            "a_validation_fine",
            "b_train_coarse",
            "b_validation_seen_coarse",
            "b_validation_unseen_coarse",
        )
    ]
    with workspace_audit("freeze_blind"):
        checkpoints = {
            path: sha256_file(resolve_in_project(path, must_exist=True))
            for path in checkpoint_paths
        }
        datasets = {
            path: sha256_file(resolve_in_project(path, must_exist=True))
            for path in data_paths
        }
        sources = {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in _source_paths()
        }
        payload = {
            "schema": "blrpng_blind_freeze_v1",
            "experiment": experiment,
            "seed": args.seed,
            "config_sha256": config["_config_sha256"],
            "formal_manifest_sha256": sha256_file(
                resolve_in_project(
                    f"artifacts/manifests/{experiment}.json", must_exist=True
                )
            ),
            "checkpoint_sha256": checkpoints,
            "dataset_sha256": datasets,
            "source_sha256": sources,
        }
        write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": args.output,
                "seed": args.seed,
                "checkpoints": len(checkpoints),
                "datasets": len(datasets),
                "sources": len(sources),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
