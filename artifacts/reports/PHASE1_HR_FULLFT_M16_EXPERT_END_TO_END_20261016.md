# Source-initialized HR full-FT expert, m=16: development results

Date: 2026-10-16. This is a four-family, one-model-seed, one-support-draw development study. It is not the frozen confirmation result and does not establish HR-expert saturation.

## Frozen protocol and selection

The model starts from the multi-boundary path-source checkpoint `artifacts/checkpoints/phase1/R1/source_path_s0_scratch.pt` (SHA-256 `b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`). Every parameter is fine-tuned on 16 HR256 support rollouts per target task, using final-only loss, batch size 2, AdamW weight decay `1e-4`, cosine schedule from LR `1e-4` to `1e-6`, for 250 epochs. Model seed is 0, support seed is 271829, and the tasks are `[0,1,3,5]` (`high_frequency`, `amplitude_ood`, `localized_bump`, `piecewise_smooth`). Checkpoints are selected by the minimum 20-step rollout relative L2 over the full eight-rollout HR-dev trajectory.

This is a budget-limited expert comparator, not an independently optimized HR-expert configuration. Its HR-dev checkpoint selection sees more selection information than the matched path-source reference, which uses LR-validation selection. Interpret the comparison as a useful expert/control comparison with that asymmetry. No confirmation query or test data was opened or used.

Frozen config: `configs/phase1_hr_fullft_m16_dev.yaml` (SHA-256 `fe86663979641c5b6169acba03bb1b3a9bb1b04eab8f87d24b47141b596fdc4d`). Training engine SHA-256: `6a2ab12001562413f62c19918ea66824308077ef2b4c141317a98c1c0ddea5b0`; adapter module SHA-256: `8f7e91a3465a1f88769726610ad1bb29ca137c3a6fb91f654a51a88af08a9137`. The task manifest SHA-256 is `bb50c7e4d5b8d0900e21d7ae64235c6d955244cc3bfdca1a97ab6056596a334a`.

## HR-development results

| Task / family | Full-FT best HR-dev | Best epoch (zero-based) | Path-source reference | Full-FT / reference | GPU-h |
|---|---:|---:|---:|---:|---:|
| 0 / high-frequency | 0.07183518 | 233 | 0.06831793 | 1.051× | 1.016510 |
| 1 / amplitude-OOD | 0.05558206 | 231 | 0.04473285 | 1.243× | 1.015879 |
| 3 / localized-bump | 0.09273513 | 0 (first epoch) | 0.06021613 | 1.540× | 1.015041 |
| 5 / piecewise-smooth | 0.06261736 | 218 | 0.05135978 | 1.219× | 1.141114 |
| **Equal-task mean** | **0.07069243** | — | **0.05615667** | **1.259×** | **4.188544 total** |

The full-FT expert's mean HR-dev error is 25.88% above the matched path-source references; it does not reach the reference on any family. Task 3 is the main gap: its best checkpoint was already the first epoch, and its final epoch worsened to `0.15002265`. The other three tasks improve later in training but remain above their references. All four time-to-match values are null (right-censored at the 250-epoch budget).

Against the random-init HR-scratch m=16 control, source initialization plus full-FT improves equal-task mean from `0.16426903` to `0.07069243` (56.97% lower). This indicates meaningful value from source initialization, while neither expert arm establishes saturation or a competitive substitute for the path-source m=4 route. The HR-scratch and full-FT report selection rules are both based on HR-dev, unlike the path-source reference's LR-dev selection.

## Measured costs

- Full-FT training charges sum to `4.188544 GPU-h` over the four tasks. The source-pretraining cost is a separate shared prerequisite and is not included in this target-expert sum.
- Generating 64 HR support rollouts (16 per task; 1,280 transitions) took `4,786.624 s` wall time, with `1,609.029 s` solver time. These CPU-side generation measures are reported separately from GPU-hours. Report: `artifacts/reports/phase1_hr_scratch_data_generation_hrdev_20261015_hrsupport_m16_s271829.json`.
- The matched path-source reference adapts on four LR target tasks from the already shared source checkpoint and uses only m=4 target support; this table does not compare full lifecycle cost or HR inference timing. No break-even claim is made.

## Reproduction and raw evidence

- Frozen card: `docs/experiment_cards/HR_FULLFT_M16_20261016.md`.
- Run: `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_hr_fullft_m16_dev.py --config configs/phase1_hr_fullft_m16_dev.yaml --attempt-suffix fit1`.
- Per-task reports, checkpoints and run summary: `results/phase1/HRFULLFTM16/hr_fullft_pathsource_hr256_m16_s0_ss271829/`.
- Machine-readable aggregate: `results/phase1/HRFULLFTM16/hr_fullft_pathsource_hr256_m16_s0_ss271829/summary.json`.
- Exact DONE charges are recorded in `docs/EXPERIMENTS.csv`; independent read-only audits recomputed all task costs and aggregate metrics in `docs/agents/REVIEW.md`. The earlier 11:53 progress snapshot predates the final expert completion and is not used as evidence here.
- Independent read-only audit passed for tasks 00, 01, 03, and 05; report histories, selected checkpoints, index seals, references, ledger charges, and aggregate metrics were checked in `docs/agents/REVIEW.md`. The audit did not open query/test data.
