from __future__ import annotations

import argparse

from blrpng.config import load_config
from blrpng.data.generate import generate_from_config, workspace_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate isolated BL-RPNG datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "manifest", "a", "b-coarse", "blind"),
        default="all",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    with workspace_audit("generate") as audit:
        outputs = generate_from_config(config, stage=args.stage)
        audit["outputs"] = [str(path) for path in outputs]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
