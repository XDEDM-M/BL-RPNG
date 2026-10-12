from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_json


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_in_project(path, must_exist=True)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration must contain a mapping: {config_path}")
    config = copy.deepcopy(loaded)
    config["_config_path"] = str(config_path.relative_to(PROJECT_ROOT))
    config["_config_sha256"] = sha256_json(loaded)
    return config


def require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing configuration key: {dotted_key}")
        value = value[key]
    return value

