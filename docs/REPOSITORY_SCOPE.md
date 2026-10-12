# Repository and artifact policy

BL-RPNG is a standalone project. It contains no imports, links or runtime
dependency on the former parent FNOtest source tree.

## Version-controlled material

- all Python source under `blrpng/`;
- experiment configurations under `configs/`;
- tests and execution scripts;
- method and theory contracts;
- curated immutable experiment records under `docs/experiment_records/`;
- a small representative PNG set under `docs/figures/`.

## Local-only material

The `artifacts/` tree is intentionally excluded from Git. It contains raw FEM
rollouts, model and optimizer states, full evaluation tables, generated
figures, run progress, and workspace audit logs. At snapshot time it occupied
approximately 217 GiB. These files are retained locally for reproducibility
but are neither required to inspect the implementation nor suitable for a
normal GitHub repository.

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
