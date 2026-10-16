# Repository and artifact policy

BL-RPNG is a standalone project. It contains no imports, links or runtime
dependency on the former parent FNOtest source tree.

## Version-controlled material

- all Python source under `blrpng/`;
- experiment configurations under `configs/`;
- tests and execution scripts;
- method and theory contracts;
- curated immutable experiment records under `docs/experiment_records/`;
- experiment protocol, chronology, ledger and conclusions under `docs/`;
- a small allowlist of curated Phase-1 Markdown reports and aggregate JSON under
  `artifacts/reports/` (raw per-task histories remain local);
- a small representative PNG set under `docs/figures/`.

## Local-only material

The `artifacts/` tree is mostly excluded from Git. It contains raw FEM rollouts,
model and optimizer states, per-task evaluation histories, generated figures,
run progress, and workspace audit logs. Selected summary reports and aggregate
JSON files are explicit exceptions so the conclusions and their machine
readable aggregates are available from GitHub. Raw data, checkpoints, detailed
histories and operational logs stay local. At the earlier snapshot this tree
occupied approximately 217 GiB.

The top-level `results/` tree stores complete per-task training histories and
is also local-only; selected conclusions and task-aggregate reports live in
version-controlled Markdown and JSON instead.

The ignore policy also rejects `*.pt`, `*.pth`, `*.ckpt`, `*.npz`, `*.npy`,
HDF5 and Zarr files outside `artifacts/`, preventing accidental publication of
large numerical or model artifacts.

## Reproduction environment

```bash
source /opt/miniconda3/bin/activate fnotest
cd /gemini/code/BL-RPNG
python -m blrpng.cli.environment_check
python -m pytest tests
```

SfePy workers use `/opt/miniconda3/envs/sfepy_env/bin/python`; the project does
not install or modify either Conda environment.
