#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/bin/activate fnotest
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
python -m blrpng.cli.generate --config configs/allen_cahn_formal.yaml --stage manifest
python -m blrpng.cli.run_matrix --config configs/allen_cahn_formal.yaml
