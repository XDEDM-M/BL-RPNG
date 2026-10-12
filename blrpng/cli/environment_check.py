from __future__ import annotations

import json
import subprocess
import sys

import torch

from blrpng.data.generate import SFEPY_PYTHON, workspace_audit
from blrpng.paths import PROJECT_ROOT, write_json


def main() -> None:
    with workspace_audit("environment_check"):
        probe = subprocess.run(
            [
                str(SFEPY_PYTHON),
                "-c",
                "import json, sfepy, sys; print(json.dumps("
                "{'python': sys.executable, 'sfepy': sfepy.__version__, 'path': sfepy.__file__}))",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = {
            "project_root": str(PROJECT_ROOT),
            "training_python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "sfepy_worker": json.loads(probe.stdout),
            "packages_modified": False,
        }
        write_json("artifacts/reports/environment.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
