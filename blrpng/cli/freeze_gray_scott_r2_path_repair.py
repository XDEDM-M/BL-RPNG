from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    names = {
        "final_only": "gray_scott_r2_source_s42_final_only_freeze.json",
        "safe_path_r015": "gray_scott_r2_source_s42_blrpng_safe_annealed_freeze.json",
    }
    inputs = {}
    for key, name in names.items():
        path = resolve_in_project(f"artifacts/manifests/{name}", must_exist=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("target_query_used") is not False:
            raise PermissionError("R2 path repair must remain query-untouched.")
        inputs[key] = {"path": str(path), "sha256": sha256_file(path), "metric": payload["best_validation_metric"]}
    degradation = inputs["safe_path_r015"]["metric"] / inputs["final_only"]["metric"] - 1.0
    if degradation <= 0.0:
        raise RuntimeError("The registered ratio-0.15 path model does not require repair.")
    payload = {
        "schema": "blrpng_gray_scott_r2_path_ratio_repair_v1",
        "frozen": True, "passed": True, "seed": 42,
        "query_used": False, "observed_relative_degradation": degradation,
        "repair": {
            "path_gradient_ratio": 0.05,
            "all_other_data_model_optimizer_and_schedule_settings_unchanged": True,
            "single_repair_value_no_sweep": True,
            "run_tag": "r005",
        },
        "inputs": inputs,
        "authorized_next_experiment": "Train exactly one A256 safe-path source repair at ratio 0.05.",
    }
    with workspace_audit("freeze_gray_scott_r2_path_repair") as audit:
        output = write_json("artifacts/manifests/gray_scott_r2_path_ratio_repair.json", payload)
        audit.update({"output": str(output), "query_used": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
