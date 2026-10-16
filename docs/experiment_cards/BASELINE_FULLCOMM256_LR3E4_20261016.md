# Ordinary full-FT + direct256 baseline: local LR screen

## Question

Does a 3× higher target-adaptation learning rate improve the strongest ordinary-pretraining baseline (full FT + direct256 consistency), whose current setting is `1e-4/cosine-100`?

## Frozen comparison

- One preselected model seed `0`, support seed `271829`, tasks `[0,1,2,3,4,5]`, `m=4`; the same six-family development support and LR-validation manifests as the Stage-2 pilot.
- Ordinary final-only source checkpoint `source_final_s0_scratch.pt`; full-parameter fine-tuning, final-only plus direct64-to256 two-step consistency at weight `0.05`, 100 epochs, cosine schedule, minimum LR `1e-6`, batch size inherited from the frozen calibration config.
- Only target learning rate changes from the strongest baseline configuration: `1e-4` → `3e-4`. LR validation selects each task checkpoint. Evaluate all six HR-dev reports only after the six LR runs finish; never open confirmation query data.
- This is a one-seed baseline tuning screen. It cannot replace the three-seed `1e-4` baseline or support a general performance claim. Report both LR- and HR-dev outcomes and task-level changes.

## Budget gate

Measured matching Stage-2 full-FT baseline cost is about `0.832 GPU-h` for six tasks at one seed; six HR-dev checks historically cost under `0.005 GPU-h`. Dispatch only after the m=16 scratch queue and source-initialized HR expert are terminal, and only if fresh exploration headroom is at least `1.8 GPU-h`, retaining at least `0.9 GPU-h` for accounting/overhead uncertainty. No additional support data or GPU jobs are allowed under this card.

## Artifacts

- Config: `configs/phase1_baseline_fullcomm256_lr3e-4_screen.yaml`
- LR command: `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_trials.py --config configs/phase1_baseline_fullcomm256_lr3e-4_screen.yaml --trial ordinary_full_comm256_lr3e-4 --seed 0 --attempt-suffix lr3e4screen`
- HR-dev: one `scripts/evaluate_phase1_hrdev.py` invocation per task, using each frozen task result and its canonical HR-dev validation index.
