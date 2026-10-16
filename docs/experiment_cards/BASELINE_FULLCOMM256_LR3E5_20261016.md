# Ordinary full-FT + direct256 baseline: low-LR bracket

## Question

Does `3e-5` improve the strongest ordinary full-FT + direct256 baseline (`1e-4/cosine-100`) after the paired `3e-4` screen tests the high side?

## Frozen comparison

- One preselected model seed `0`, support seed `271829`, tasks `[0,1,2,3,4,5]`, `m=4`; exactly the same development support and LR-validation manifests used by the existing `1e-4` baseline and the high-LR screen.
- Ordinary final-only source checkpoint `source_final_s0_scratch.pt`; full-parameter fine-tuning, final-only plus direct64-to256 two-step consistency at weight `0.05`, 100 epochs, cosine schedule, minimum LR `1e-6`, batch size inherited from the frozen calibration config.
- The only change from the strongest baseline configuration is target learning rate `1e-4` → `3e-5`. This is the low side of the local `{3e-5, 1e-4, 3e-4}` bracket; `1e-4` has the existing three-seed results, and `3e-4` is the paired R5 high-side screen.
- LR validation selects each task checkpoint. Evaluate all six HR-dev reports only after the six LR runs finish. Report per-task and equal-task LR/HR development values; no query/test access.
- This is a one-seed hyperparameter screen, not a substitute for the three-seed baseline or frozen confirmation.

## Dispatch gate and budget

- Do not start until the high-LR R5 screen and its six HR-dev evaluations are terminal and included in a fresh ledger calculation.
- Dispatch only if fresh exploration headroom is at least `1.8 GPU-h`; expected training plus HR-dev evaluation cost is about `0.94 GPU-h`, leaving at least `0.9 GPU-h` for accounting and runtime uncertainty. Preserve the 48 GPU-h confirmation/control reserve.
- Reuse the existing support and validation data; generate no new target data. No concurrent GPU work.

## Artifacts

- Config: `configs/phase1_baseline_fullcomm256_lr3e-5_screen.yaml`.
- LR command: `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_trials.py --config configs/phase1_baseline_fullcomm256_lr3e-5_screen.yaml --trial ordinary_full_comm256_lr3e-5 --seed 0 --attempt-suffix lr3e5screen`.
- After all LR runs finish, evaluate each result with `scripts/evaluate_phase1_hrdev.py` on its canonical HR-dev validation index.
