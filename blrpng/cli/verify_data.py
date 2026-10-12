from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.data.schema import DatasetBundle, load_dataset, validate_coarse_fine_pairing
from blrpng.paths import resolve_in_project, write_json


def _formal_paths(experiment_name: str) -> dict[str, str]:
    return {
        role: f"artifacts/data/{experiment_name}_{suffix}.pt"
        for role, suffix in {
            "a_train": "a_train_fine",
            "a_validation": "a_validation_fine",
            "b_train_coarse": "b_train_coarse",
            "b_validation_seen_coarse": "b_validation_seen_coarse",
            "b_validation_unseen_coarse": "b_validation_unseen_coarse",
            "b_train_fine": "b_train_fine",
            "blind_seen": "blind_seen_fine",
            "blind_unseen": "blind_unseen_fine",
            "blind_frequency_ood": "blind_frequency_ood_fine",
            "blind_amplitude_ood": "blind_amplitude_ood_fine",
        }.items()
    }


def _common_step_prefix(
    fine: DatasetBundle, *, num_steps: int
) -> DatasetBundle:
    selected = fine.tensors["local_step_id"] < num_steps
    return DatasetBundle(
        tensors={key: value[selected] for key, value in fine.tensors.items()},
        meta=dict(fine.meta),
        validation=dict(fine.validation),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model = config["model"]
    paths = config["data"]
    with workspace_audit("verify_data"):
        reports = {}
        bundles = {}
        for role in ("train", "validation", "test"):
            path_key = f"{role}_path"
            if path_key in paths:
                bundle = load_dataset(paths[path_key], modes2=int(model["modes2"]))
                bundles[role] = bundle
                reports[role] = bundle.validation
        if "validation_path" in paths and "test_path" in paths:
            reports["coarse_fine_pairing"] = validate_coarse_fine_pairing(
                bundles["validation"],
                bundles["test"],
            )
        if not reports:
            name = config["experiment"]["name"]
            for role, path in _formal_paths(name).items():
                if resolve_in_project(path).exists():
                    bundle = load_dataset(path, modes2=int(model["modes2"]))
                    bundles[role] = bundle
                    reports[role] = bundle.validation
            if not reports:
                raise FileNotFoundError(
                    f"No generated datasets found for experiment {name!r}."
                )
            manifest_hashes = {
                bundle.meta["manifest_sha256"] for bundle in bundles.values()
            }
            if len(manifest_hashes) != 1:
                raise ValueError(
                    f"Formal datasets do not share one frozen manifest: {manifest_hashes}"
                )
            reports["shared_manifest_sha256"] = next(iter(manifest_hashes))
            if "b_train_coarse" in bundles and "b_train_fine" in bundles:
                coarse = bundles["b_train_coarse"]
                common_fine = _common_step_prefix(
                    bundles["b_train_fine"],
                    num_steps=int(coarse.meta["num_time_steps"]),
                )
                reports["b_train_coarse_fine_pairing"] = (
                    validate_coarse_fine_pairing(coarse, common_fine)
                )
            output = f"artifacts/reports/{name}_data_verification.json"
        else:
            output = "artifacts/reports/data_verification.json"
        write_json(output, reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
