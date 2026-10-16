# Ordinary full-FT + direct256 baseline: LR=3e-4 screen

Date: 2026-10-16. This is a one-seed, six-family development screen to tune the strongest ordinary full-FT + direct256 baseline. It is not a confirmation result.

## Protocol and selection

The screen keeps the ordinary final-only source checkpoint, full-parameter target fine-tuning, final-only loss plus direct64-to256 two-step consistency (`weight=0.05`), `m=4`, 100 epochs, cosine schedule to `1e-6`, model seed 0, and support seed 271829 fixed. It changes only target LR from the established `1e-4` to `3e-4`. All six families `[0,1,2,3,4,5]` use the same support and 64×64 LR-validation indices as the seed0 `1e-4` comparison. Per-task checkpoints are selected from LR validation only; HR-dev is evaluated after all six LR runs complete.

This is a local bracket point, not a replacement for the three-seed `1e-4` control. The new HR-dev checks are independent development evaluations, not training/selection data. Query/test data remains closed; all six reports have `query_read=false`.

Config `configs/phase1_baseline_fullcomm256_lr3e-4_screen.yaml`; `645a9ef6037f9833b9f8859eaab043e0717ae930ba55fe9b94e68b14094b9172` is the canonical resolved-config descriptor fingerprint recorded by the runner, while the raw YAML byte SHA-256 is `a77ee30bb8ed08a11ce35d183d990857049a3995257e7024d72a29888fc4ec0b`. Training engine SHA-256 `6a2ab12001562413f62c19918ea66824308077ef2b4c141317a98c1c0ddea5b0`.

## Per-family development comparison

| Task / family | LR-val `1e-4` | LR-val `3e-4` | HR-dev `1e-4` | HR-dev `3e-4` | HR-dev ratio |
|---|---:|---:|---:|---:|---:|
| 0 / high-frequency | 0.04337541 | 0.16078084 | 0.07271281 | 0.17564816 | 2.416× |
| 1 / amplitude-OOD | 0.04804767 | 0.11531357 | 0.04805644 | 0.09549712 | 1.987× |
| 2 / matern-GP | 0.03322085 | 0.09026610 | 0.04177176 | 0.11483409 | 2.749× |
| 3 / localized-bump | 0.03694564 | 0.10606825 | 0.06366513 | 0.16877616 | 2.651× |
| 4 / mixed-Fourier | 0.04212978 | 0.12062085 | 0.08776810 | 0.20089716 | 2.289× |
| 5 / piecewise-smooth | 0.03767342 | 0.15312910 | 0.04268695 | 0.12378177 | 2.900× |
| **Equal-task mean** | **0.04023213** | **0.12436312** | **0.05944353** | **0.14657241** | **2.466×** |

The `3e-4` LR-val mean is `3.091×` the paired seed0 `1e-4` mean. Its HR-dev mean is `146.57%` higher; each family is worse by `1.99×–2.90×`. The LR-selected best epoch (zero-based) for `3e-4` was `[2,3,1,0,8,2]`, followed by degradation for the remainder of the 100-epoch schedules on all tasks. This is evidence that the high side is too aggressive for this baseline under the frozen schedule; do not use it to inflate the baseline comparison or claim a path-specific advantage.

Across the prior three-seed Stage-2 comparison, ordinary full-FT + comm256 mean HR-dev is `0.05488117`; path-source + comm256 is `0.05347209`, a 2.57% gap with a paired interval crossing zero. Thus the established `1e-4` baseline remains the strongest ordinary comparison. The one-seed `1e-4` values above are included only for paired LR-bracket diagnostics.

## Measured cost and next bracket point

- Six target-training runs cost `0.924529 GPU-h`; six HR-dev evaluations cost `0.004193 GPU-h`, total `0.928721 GPU-h`. The existing source checkpoint and support pool are shared; source pretraining and support generation are not charged again.
- A refreshed terminal-ledger snapshot after this screen is `artifacts/reports/phase1_progress_after_r5_high_lr_complete_20261016_1250.json`: `69.705972 GPU-h` terminal campaign use and an estimated `2.294028 GPU-h` exploration headroom after protecting the 48 GPU-h reserve. The snapshot's terminal charge excludes some early untimed HR-dev evaluations, so this is an optimistic estimate rather than a complete campaign-occupancy total. The frozen low-side screen card budgets about `0.94 GPU-h` plus a `0.9 GPU-h` uncertainty buffer; its dispatch gate used this ledger estimate and required both GPUs idle.
- The low-side point LR=`3e-5` is frozen in `configs/phase1_baseline_fullcomm256_lr3e-5_screen.yaml` and `docs/experiment_cards/BASELINE_FULLCOMM256_LR3E5_20261016.md`. It will reuse the same six tasks/support/validation indices and retain LR-only selection. Its one-seed outcome cannot replace the three-seed `1e-4` baseline.

## Reproduction and raw evidence

- Frozen card/config: `docs/experiment_cards/BASELINE_FULLCOMM256_LR3E4_20261016.md`, `configs/phase1_baseline_fullcomm256_lr3e-4_screen.yaml`.
- LR command: `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_trials.py --config configs/phase1_baseline_fullcomm256_lr3e-4_screen.yaml --trial ordinary_full_comm256_lr3e-4 --seed 0 --attempt-suffix lr3e4screen`.
- LR results and aggregate: `results/phase1/R5/ordinary_full_comm256_lr3e-4-lr3e4screen_m4_s0_ss271829/`.
- HR-dev reports: `artifacts/reports/phase1_hrdev_evaluations/ordinary_full_comm256_lr3e-4-lr3e4screen_task{00..05}_s0_ss271829.json`.
- Paired `1e-4` LR task outputs: `results/phase1/S2PILOT/pilot_ordinary_full_comm256_m4_s0_ss271829/`; HR-dev reports share the corresponding `pilot_ordinary_full_comm256_taskXX_s0_ss271829.json` names.
- Current ledger and audit trail: `docs/EXPERIMENTS.csv`; the independent read-only audit of the aggregate is recorded in `docs/agents/REVIEW.md` when completed.
