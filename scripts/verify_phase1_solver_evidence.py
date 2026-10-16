"""Package and audit a fresh solver evidence smoke without opening query data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blrpng.data.shards import write_v4_rollouts_from_arrays, validate_shard
from blrpng.paths import PROJECT_ROOT, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = PROJECT_ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Preserve existing evidence: {root}")
    npz = PROJECT_ROOT / args.npz
    meta_path = PROJECT_ROOT / args.meta
    meta = json.loads(meta_path.read_text())
    if meta.get("split") != "support" or not meta.get("all_converged"):
        raise ValueError("This implementation smoke requires converged support-role solves")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: torch.from_numpy(archive[key]) for key in archive.files}
    mapping = {int(b): int(t) for b, t in zip(arrays["boundary_id"], arrays["target_task_id"])}
    rows = write_v4_rollouts_from_arrays(arrays, meta, shard_dir=root / "shards",
                                       target_task_by_boundary=mapping, role="support",
                                       source_sha256=sha256_file(npz))
    checks = []
    for row in rows:
        payload = torch.load(PROJECT_ROOT / row["path"], weights_only=False)
        check = validate_shard(payload)
        if "accepted_newton_step_norms" not in payload:
            raise ValueError("Selection replay evidence did not survive shard packaging")
        checks.append({"shard": row, "validation": check,
                       "selected_index": payload["selected_newton_state_index"].tolist(),
                       "trace_count": payload["accepted_newton_counts"].tolist()})
    write_json(root / "audit.json", {
        "schema": "phase1_real_solver_selection_replay_smoke_v1", "query_read": False,
        "implementation_smoke_only": True, "all_converged": meta["all_converged"],
        "raw_npz": args.npz, "raw_npz_sha256": sha256_file(npz),
        "raw_meta": args.meta, "raw_meta_sha256": sha256_file(meta_path),
        "worker_sha256": sha256_file(PROJECT_ROOT / "blrpng/solvers/sfepy_worker.py"),
        "selector_sha256": sha256_file(PROJECT_ROOT / "blrpng/solvers/progress.py"),
        "shards_module_sha256": sha256_file(PROJECT_ROOT / "blrpng/data/shards.py"),
        "selection_metadata": meta["newton_path_selection"], "checks": checks,
        "limitation": "Residuals and step norms replay selection; full node states are not retained.",
    })
    print(json.dumps({"audit": str(root / "audit.json"), "rollouts": len(rows),
                      "transitions": int(arrays['y'].shape[0]), "all_converged": True}, indent=2))


if __name__ == "__main__":
    main()
