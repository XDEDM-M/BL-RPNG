# Ordinary full-FT plus direct256 consistency control

- **Question:** Does label-free direct64-to256 consistency also improve the strongest ordinary-pretraining baseline when the target adapter updates all model parameters?
- **Selection of base:** The ordinary final-only full-FT screen selected `lr=1e-4/cosine-100` by its LR-dev mean `0.02971277`; its HR-dev mean is `0.06203766`. The same generic direct256 consistency loss lowered the ordinary NLT HR-dev mean from `0.06219577` to `0.05715385`, while the best path-source direct256 result is `0.05287928`.
- **Frozen design:** Ordinary final-only source checkpoint, full FT at LR `1e-4`, cosine-100, direct64-to256 two-step consistency at weight `0.05`, m=4, tasks 0/1/3/5, model seed 42, support seed 271829. Only LR validation selects checkpoints; HR-dev is evaluated after the four-task screen. No query access.
- **Config:** `configs/phase1_baseline_fullcomm256.yaml`; new namespace `R2C` preserves the earlier R2/R2B evidence.
- **Decision rule:** Compare its LR mean, HR-dev mean and worst family with ordinary full-FT final-only, ordinary NLT direct256, and path-source direct256. This is a baseline control, not a new main-method trial.

## Execution status

- The two-epoch task-0 smoke passed at LR-dev `0.039508754202` without query access.
- Formal tasks 0/1/3/5 completed at LR-dev `0.033986920293/0.031091272814/0.030633970612/0.025385082053`, runner-selected epochs `33/12/10/18`, and total training charge `0.554417839611` GPU-hours. The four-task LR mean is `0.030274311443`; each task has `query_metric=null`. Independent review confirmed each epoch is the unique LR-history minimum and all source/config/data/checkpoint/index/ledger bindings verify.
- Sequential HR-development evaluation then scored `0.070172757702/0.037068503466/0.064227065106/0.052035644988`, mean `0.055875992816`, with all four reports `query_read=false`. This improves the ordinary NLT+direct256 baseline `0.057153851914` by `2.24%` and ordinary full-FT final-only `0.062037663115` by `9.93%`, but remains `5.67%` worse than path-source direct256 `0.052879281671`. The strongest measured overall method remains path-source direct256.
- Raw training outputs: `results/phase1/R2C/ordinary_full_comm256_lr1e-4_cw005-fullcomm1_m4_s42_ss271829/`; HR reports use the same prefix in `artifacts/reports/phase1_hrdev_evaluations/`. Fresh summary: `artifacts/reports/phase1_progress_20261014_0335_r2c_complete.json` (terminal campaign ledger `11.706041885620` GPU-hours). Independent review also verified HR report seals and `query_read=false`; see `docs/agents/REVIEW.md`. Its cost note flags duplicated same-path HRDEV rows for the ordinary NLT+direct256 comparator; its accuracy outputs are unchanged, but that comparator's cost record is ambiguous.
