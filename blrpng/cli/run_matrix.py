from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, write_json
from blrpng.training.engine import train
from blrpng.training.matrix import matrix_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the 120-task causal matrix; execution requires generated data."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", default="artifacts/manifests/allen_cahn_experiment_matrix.json"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help=(
            "Execute only this subset of frozen seeds (for example: --seeds 42). "
            "The written experiment matrix remains the full frozen matrix."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse already completed final checkpoints when resuming an execution.",
    )
    parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("scratch", "final_pretrained", "blrpng_pretrained"),
        help="Execute only these causal initialization columns.",
    )
    parser.add_argument(
        "--skip-pretraining",
        action="store_true",
        help="Do not execute the two A pretraining jobs.",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Do not execute the three budget-4 ablation branches.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    with workspace_audit("run_matrix"):
        payload = matrix_payload(config)
        write_json(args.output, payload)
        completed = []
        if args.execute:
            name = config["experiment"]["name"]
            frozen_seeds = [int(seed) for seed in config["training"]["seeds"]]
            selected_seeds = frozen_seeds if args.seeds is None else list(args.seeds)
            invalid_seeds = sorted(set(selected_seeds) - set(frozen_seeds))
            if invalid_seeds:
                raise ValueError(
                    f"Requested seeds are not in the frozen config: {invalid_seeds}"
                )
            if len(set(selected_seeds)) != len(selected_seeds):
                raise ValueError("Each requested execution seed must be unique.")

            progress_output = (
                "artifacts/reports/matrix_progress.json"
                if selected_seeds == frozen_seeds
                else "artifacts/reports/matrix_progress_seeds_"
                + "_".join(str(seed) for seed in selected_seeds)
                + ".json"
            )

            def run_or_reuse(*, output_checkpoint: str, **kwargs):
                checkpoint = resolve_in_project(output_checkpoint)
                if args.skip_existing and checkpoint.exists():
                    completed.append(str(checkpoint))
                    return
                result = train(
                    config,
                    output_checkpoint=output_checkpoint,
                    **kwargs,
                )
                completed.append(result["checkpoint"])
                write_json(
                    progress_output,
                    {
                        "completed": completed,
                        "selected_seeds": selected_seeds,
                        "completed_count": len(completed),
                    },
                )

            if not args.skip_pretraining:
                for seed in selected_seeds:
                    for pretrain_objective in ("final_only", "blrpng"):
                        output = (
                            f"artifacts/checkpoints/{name}_a_{pretrain_objective}_seed{seed}.pt"
                        )
                        run_or_reuse(
                            seed=int(seed),
                            objective=pretrain_objective,
                            output_checkpoint=output,
                            train_data_path=f"artifacts/data/{name}_a_train_fine.pt",
                            validation_data_paths=[
                                f"artifacts/data/{name}_a_validation_fine.pt"
                            ],
                        )
            train_data = f"artifacts/data/{name}_b_train_coarse.pt"
            validation_data = [
                f"artifacts/data/{name}_b_validation_seen_coarse.pt",
                f"artifacts/data/{name}_b_validation_unseen_coarse.pt",
            ]
            selected_tasks = [
                task
                for task in payload["causal_tasks"]
                if int(task["seed"]) in selected_seeds
                and (
                    args.initializations is None
                    or task["initialization"] in args.initializations
                )
            ]
            for task in selected_tasks:
                pretrain = None
                if task["initialization"] != "scratch":
                    pretrain_objective = (
                        "final_only"
                        if task["initialization"] == "final_pretrained"
                        else "blrpng"
                    )
                    pretrain = (
                        f"artifacts/checkpoints/{name}_a_{pretrain_objective}"
                        f"_seed{task['seed']}.pt"
                    )
                    resolve_in_project(pretrain, must_exist=True)
                output = f"artifacts/checkpoints/{task['task_id']}.pt"
                run_or_reuse(
                    seed=int(task["seed"]),
                    objective=task["target_objective"],
                    initialization_checkpoint=pretrain,
                    output_checkpoint=output,
                    train_data_path=train_data,
                    validation_data_paths=validation_data,
                    boundary_budget=int(task["boundary_budget"]),
                )
            for seed in ([] if args.skip_ablations else selected_seeds):
                # Raw three-channel scratch final-only.
                run_or_reuse(
                    seed=int(seed),
                    objective="final_only",
                    representation_override="raw_legacy",
                    output_checkpoint=(
                        f"artifacts/checkpoints/ablation_b4_s{seed}_raw_scratch_final.pt"
                    ),
                    train_data_path=train_data,
                    validation_data_paths=validation_data,
                    boundary_budget=4,
                )

                # Raw absolute-hook0 legacy pretraining and fine-tuning.
                legacy_pretrain = (
                    f"artifacts/checkpoints/{name}_a_legacy_path_seed{seed}.pt"
                )
                run_or_reuse(
                    seed=int(seed),
                    objective="legacy_path",
                    output_checkpoint=legacy_pretrain,
                    train_data_path=f"artifacts/data/{name}_a_train_fine.pt",
                    validation_data_paths=[
                        f"artifacts/data/{name}_a_validation_fine.pt"
                    ],
                )
                run_or_reuse(
                    seed=int(seed),
                    objective="legacy_path",
                    initialization_checkpoint=legacy_pretrain,
                    output_checkpoint=(
                        f"artifacts/checkpoints/ablation_b4_s{seed}_raw_legacy_path.pt"
                    ),
                    train_data_path=train_data,
                    validation_data_paths=validation_data,
                    boundary_budget=4,
                )

                # Two-progress/two-layer lifted ablation.
                dual_pretrain = (
                    f"artifacts/checkpoints/{name}_a_dual_stage_seed{seed}.pt"
                )
                run_or_reuse(
                    seed=int(seed),
                    objective="dual_stage",
                    output_checkpoint=dual_pretrain,
                    train_data_path=f"artifacts/data/{name}_a_train_fine.pt",
                    validation_data_paths=[
                        f"artifacts/data/{name}_a_validation_fine.pt"
                    ],
                )
                run_or_reuse(
                    seed=int(seed),
                    objective="dual_stage",
                    initialization_checkpoint=dual_pretrain,
                    output_checkpoint=(
                        f"artifacts/checkpoints/ablation_b4_s{seed}_dual_stage.pt"
                    ),
                    train_data_path=train_data,
                    validation_data_paths=validation_data,
                    boundary_budget=4,
                )
            payload["executed_checkpoints"] = completed
    print(
        json.dumps(
            {
                "tasks": payload["causal_task_count"],
                "output": args.output,
                "executed_checkpoints": len(completed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
