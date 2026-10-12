#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/bin/activate fnotest
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
python -m blrpng.cli.environment_check
python -m blrpng.cli.generate --config configs/smoke.yaml
python -m blrpng.cli.verify_data --config configs/smoke.yaml
python -m blrpng.cli.train --config configs/smoke.yaml
python -m blrpng.cli.evaluate --config configs/smoke.yaml
python -m pytest tests
