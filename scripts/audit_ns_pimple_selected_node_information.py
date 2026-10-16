#!/usr/bin/env python3
"""Compare the frozen first-hook label with per-step maximum-S path nodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = ROOT.parent / "NS-PIMPLE"
OUTPUT = ROOT / "artifacts/reports/ns_pimple_method_v1/selected_node_information_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_rows() -> list[tuple[str, str, str, str]]:
    v0 = json.loads((NS_ROOT / "data/v0/manifest.json").read_text())
    source_v0 = [
        (row["pair_id"], "source_training", "hr", row["cases"]["hr"]["hdf5"])
        for row in v0["pairs"]
        if row.get("split") == "source" and row.get("role") == "support"
        and row.get("task_id") != "F4_poly24_top/a060"
    ]
    source_v1 = [
        (row["pair_id"], "source_training", row["level"], row["hdf5"])
        for row in json.loads((NS_ROOT / "data/source_pretraining_v1/manifest.json").read_text())["trajectories"]
        if row.get("role") == "source_training"
    ]
    target_support = [
        (row["pair_id"], "target_support", row["level"], row["hdf5"])
        for row in json.loads((NS_ROOT / "data/method_v1/manifest.json").read_text())["trajectories"]
        if row.get("role") == "support"
    ]
    rows = source_v0 + source_v1 + target_support
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("duplicate training/support pair IDs in selection audit")
    return rows


def main() -> int:
    first_s, max_s = [], []
    first_nodes, max_s_nodes = [], []
    case_records = []
    rows = selected_rows()
    for pair_id, role, level, relative in rows:
        path = NS_ROOT / relative
        with h5py.File(path, "r") as h5:
            scores = h5["diagnostics/S"][:]
            valid = h5["path/valid_mask"][:].astype(bool)
            endpoint = h5["path/is_endpoint"][:].astype(bool)
            outer = h5["path/outer_iteration_index"][:]
        local_first, local_max, local_first_nodes, local_max_nodes = [], [], [], []
        for time_index in range(scores.shape[0]):
            candidates = np.flatnonzero(valid[time_index] & ~endpoint[time_index]
                                        & np.isfinite(scores[time_index]))
            if not len(candidates):
                raise ValueError(f"no valid non-endpoint S value: {pair_id}/{time_index}")
            first = int(candidates[0])
            informative = int(candidates[int(np.argmax(scores[time_index, candidates]))])
            local_first.append(float(scores[time_index, first]))
            local_max.append(float(scores[time_index, informative]))
            local_first_nodes.append(int(outer[time_index, first]))
            local_max_nodes.append(int(outer[time_index, informative]))
        first_s.extend(local_first)
        max_s.extend(local_max)
        first_nodes.extend(local_first_nodes)
        max_s_nodes.extend(local_max_nodes)
        case_records.append({
            "pair_id": pair_id,
            "role": role,
            "level": level,
            "physical_steps": len(local_first),
            "first_node_is_max_S_fraction": float(np.mean(np.asarray(local_first_nodes)
                                                           == np.asarray(local_max_nodes))),
            "median_S_first_node": float(np.median(local_first)),
            "median_S_max_node": float(np.median(local_max)),
        })
    report = {
        "schema": "blrpng_ns_pimple_selected_node_information_audit_v1",
        "status": "complete",
        "selection_rule_tested": "argmax finite S across valid non-endpoint outer nodes independently at each physical step",
        "comparison_rule": "the method training pipeline selects the earliest complete non-endpoint outer node",
        "source_training_trajectories": sum(row[1] == "source_training" for row in rows),
        "target_lr_support_trajectories": sum(row[1] == "target_support" for row in rows),
        "physical_steps": len(first_s),
        "first_node_is_max_S_steps": int(np.sum(np.asarray(first_nodes) == np.asarray(max_s_nodes))),
        "first_node_is_max_S_fraction": float(np.mean(np.asarray(first_nodes) == np.asarray(max_s_nodes))),
        "median_S_first_node": float(np.median(first_s)),
        "median_S_max_node": float(np.median(max_s)),
        "median_selected_outer_index_first_node": float(np.median(first_nodes)),
        "median_selected_outer_index_max_S": float(np.median(max_s_nodes)),
        "max_S_outer_index_counts": {
            str(int(key)): int(value) for key, value in zip(*np.unique(max_s_nodes, return_counts=True))
        },
        "target_hr_query_fields_read": False,
        "case_records": case_records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "case_records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
