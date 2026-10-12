from __future__ import annotations

import argparse
import json

from blrpng.config import load_config
from blrpng.data.blind_boundaries import freeze_reconfirmation_manifest
from blrpng.data.generate import workspace_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze unopened Allen-Cahn reconfirmation tasks.")
    parser.add_argument("--config", default="configs/allen_cahn_reconfirmation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    with workspace_audit("freeze_reconfirmation_manifest") as audit:
        path = freeze_reconfirmation_manifest(config)
        audit["manifest"] = path
    print(json.dumps({"manifest": path, "query_opened": False}, indent=2))


if __name__ == "__main__":
    main()

