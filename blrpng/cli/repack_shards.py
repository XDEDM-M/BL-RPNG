from __future__ import annotations

import argparse
import json

import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file, sha256_json
from blrpng.utils import atomic_torch_save


def _write_index(path, payload: dict) -> None:
    payload.pop("sha256", None)
    payload["sha256"] = sha256_json(payload)
    output = ensure_artifact_path(path)
    temporary = ensure_artifact_path(output.with_suffix(output.suffix + ".partial"))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detach shared tensor storage in v4 shards and refresh indexes.")
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    parent_path = resolve_in_project(args.index, must_exist=True)
    parent = read_index(parent_path, verify_hashes=True)
    shard_paths = {row["path"] for row in parent["shards"]}
    before = sum(resolve_in_project(path).stat().st_size for path in shard_paths)
    with workspace_audit("repack_shards") as audit:
        hashes = {}
        for relative in sorted(shard_paths):
            path = resolve_in_project(relative, must_exist=True)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            detached = {key: value.clone() if isinstance(value, torch.Tensor) else value
                        for key, value in payload.items()}
            validate_shard(detached)
            atomic_torch_save(detached, path)
            hashes[relative] = sha256_file(path)
        for row in parent["shards"]:
            row["sha256"] = hashes[row["path"]]
        _write_index(parent_path, parent)
        refreshed_parent = read_index(parent_path)
        parent_relative = str(parent_path.relative_to(PROJECT_ROOT))
        for task_path in sorted((PROJECT_ROOT / "artifacts/data/tasks").glob("*.index.json")):
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            if not any(row.get("path") in hashes for row in payload.get("shards", [])):
                continue
            for row in payload["shards"]:
                if row.get("path") in hashes:
                    row["sha256"] = hashes[row["path"]]
            if payload.get("metadata", {}).get("parent_index") == parent_relative:
                payload["metadata"]["parent_sha256"] = refreshed_parent["sha256"]
            _write_index(task_path, payload)
        after = sum(resolve_in_project(path).stat().st_size for path in shard_paths)
        audit.update({"index": parent_relative, "shards": len(shard_paths),
                      "bytes_before": before, "bytes_after": after})
    print(json.dumps({"shards": len(shard_paths), "bytes_before": before,
                      "bytes_after": after, "ratio": after / max(before, 1)}, indent=2))


if __name__ == "__main__":
    main()
