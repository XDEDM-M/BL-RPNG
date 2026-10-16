# Stage-0 repository map

## Primary Allen–Cahn route

- Source and target task configuration: `configs/allen_cahn_v2_calibration.yaml`, `configs/allen_cahn_reconfirmation.yaml`.
- Data generation and split manifests: `blrpng/data/`, `blrpng/cli/generate_calibration.py`, `blrpng/cli/generate_reconfirmation.py`, `blrpng/cli/prepare_reconfirmation_split.py`.
- SfePy accepted Newton states/corrections: `blrpng/solvers/` and schema/verification under `blrpng/data/` (mechanism agent is checking exact label path).
- Lifted representation and Dirichlet projection: `blrpng/data/lifting.py`, `blrpng/models/fno.py`.
- Training losses and optimization: `blrpng/training/losses.py`, `blrpng/training/engine.py`; task adaptation wrapper: `blrpng/cli/train_reconfirmation.py`.
- HR rollout and metrics: `blrpng/evaluation/rollout.py`, `blrpng/evaluation/metrics.py`, `blrpng/cli/evaluate_reconfirmation.py`.
- Frozen historical report: `docs/EXPERIMENT_STATUS.md`, `docs/experiment_records/allen_cahn_current_state_seed42.json`.

## Reusable assets and limits

- The local `artifacts/` tree contains 68 GiB of data and 149 GiB of checkpoints, including source models, LR support/validation, and opened seed-42 HR queries.
- The historical 48-task, six-family query has been opened and is prohibited for new method/hyperparameter selection.
- Known historic evidence is seed 42 only; the HR specialist reference is resource-capped and unsaturated.
- The repository is independent of the original FNOtest source. `fnotest` is the training environment; SfePy solver jobs use `sfepy_env`.

See `docs/agents/MECHANISM.md` and `docs/agents/PERFORMANCE.md` when the read-only audits finish.
