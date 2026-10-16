# Strong baseline capacity and convergence card

- Ordinary transfer must compare NLT and full FT. Existing ordinary NLT search has three complete cosine-100 LR points, best mean LR validation 0.03420821. At card creation, target full FT was still unmeasured; the completed scored values are appended below.
- Test full FT at LR3e-5/1e-4, cosine100, same source final-only initialization, same m4 LR support/validation and task seeds. Choose checkpoints by LR validation only. Evaluate shortlisted ordinary NLT/full candidates on HR development.
- Scratch three 100-epoch screens have lowest LR mean 0.53478639 (lr1e-4); lr1e-3 final train loss is ~0.0016 but long rollout error remains ~0.60. Rule out additional training/regularization benefits using independently trained 500-epoch cosine trials at LR1e-4/3e-4 and weight decay1e-4. Do not discard earlier trials. All methods receive generic improvements that prove useful.
- New immutable configuration: `configs/phase1_baseline_capacity.yaml`; round namespace R1B avoids overwriting earlier evidence. Frozen tasks0/1/3/5, model42, support271829, m4, 20-step LR-only checkpoint selection. No query access.
- Budget: these four trials plus the existing three-per-baseline screens remain within 12 baseline trial slots per method. Run sequentially on one GPU; verify full-FT smoke before scoring. Later comparisons use paired tasks/support/seeds, with independent method tuning.

## Interim capacity results

- Ordinary full FT completed at LR `3e-5` and `1e-4` with four-family LR-dev means `0.030261243111` and `0.029712766977`; the latter is 1.81% lower. Exact task values, unique selected epochs, costs and independent provenance audit are in `docs/STATE.md` and `docs/agents/REVIEW.md`. HR-development evaluation remains to confirm their HR ranking.
- Scratch LR `1e-4`, cosine-500, WD `1e-4` completed at `0.542029540986/0.471997121815/0.429840025399/0.405628048163` (tasks 0/1/3/5; mean `0.462373684091`; cost `0.406252357090` GPU-h). LR-only selection epochs are `365/290/354/498`. For task 5, validation moved from `0.407222` at epoch 400 to `0.405675` at epoch 499 (minimum `0.405628` at 498), a small late improvement but not enough to close the large scratch-to-transfer gap. Task 0's complete provenance was independently reviewed; see `docs/agents/REVIEW.md`. This is a valid but weak scratch screen, not a query result.
- Scratch LR `3e-4`, cosine-500, WD `1e-4` is still running on the same four frozen tasks. Do not select a scratch winner or run HR-development evaluation until its LR screen terminates.


## Completed capacity screen (seed 42, m=4, support 271829)

| Baseline | LR mean | HR-dev mean | GPU-hours |
|---|---:|---:|---:|
| Ordinary full FT, lr=3e-5, cosine-100 | 0.03026124 | 0.06297840 | 0.09208 |
| Ordinary full FT, lr=1e-4, cosine-100 | 0.02971277 | 0.06203766 | 0.09153 |
| LR scratch, lr=1e-4, cosine-500, wd=1e-4 | 0.46237368 | not evaluated | 0.40625 |
| LR scratch, lr=3e-4, cosine-500, wd=1e-4 | 0.43320730 | not evaluated | 0.40568 |

- The full-FT task-0 smoke completed two epochs with finite loss/validation history and query unopened. Formal checkpoints were selected only from LR validation. Both full-FT HR reports are independent development evaluations with `query_read=false`.
- Full FT improved mean LR validation over ordinary NLT's best screen `0.03420821` by 11.5% (`3e-5`) and 13.1% (`1e-4`), but its best HR mean `0.06203766` only ties/slightly improves the prior ordinary NLT HR mean `0.06219577` and remains worse than ordinary-source direct256 NLT (`0.05715385`). Therefore full FT final-only is not the strongest ordinary performance baseline; give the best full-FT arm the direct256 consistency objective before final baseline ranking.
- Extending scratch to 500 epochs plus wd=1e-4 improved its best LR mean from `0.53478639` to `0.43320730` (lr=3e-4, 19.0% lower relative error); it remains far from transfer performance. This rules out short 100-epoch training as the sole explanation for scratch's gap, but the screen is still single-seed/four-family.
- Timed source outputs are `results/phase1/R1B/{ordinary_full_lr3e-5_cos100,ordinary_full_lr1e-4_cos100,scratch_lr1e-4_cos500_wd1e-4,scratch_lr3e-4_cos500_wd1e-4}-capacity1_m4_s42_ss271829/summary.json`. Full-FT HR reports use matching trial prefixes under `artifacts/reports/phase1_hrdev_evaluations/`. Progress snapshot: `artifacts/reports/phase1_progress_20261014_0248_hr.json`.
