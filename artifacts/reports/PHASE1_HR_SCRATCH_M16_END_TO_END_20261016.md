# HR-scratch m=16 end-to-end comparison

Date: 2026-10-16. This is a development comparison on four selected families, one model seed, and one support draw. It is not the frozen confirmation result.

## Protocol and selection

The direct-HR model starts from random initialization, trains the full FNO at 256×256 on 16 HR support rollouts per target task, uses final-only loss, batch size 2, AdamW weight decay `1e-4`, cosine schedule for 500 epochs, initial LR `3e-4`, minimum LR `1e-6`, and model seed 0. Support seed is `271829`; tasks are `[0,1,3,5]` (`high_frequency`, `amplitude_ood`, `localized_bump`, `piecewise_smooth`). Each task uses its own sealed HR support index and the canonical eight-rollout 256×256 HR-dev index. The best checkpoint is selected across the full HR-dev trajectory. The matched path-source reference uses the same task/seed/support draw and HR-dev index, but its checkpoint was selected using LR validation; it starts from multi-boundary source pretraining and receives target path-source + comm256 adaptation.

HR-dev checkpoint selection gives direct HR scratch more selection information than path-source. Treat this as a useful end-to-end expert/control comparison, not a selection-symmetric estimate. No confirmation query/test data was opened or used.

Frozen config: `configs/phase1_hr_scratch_m16_dev.yaml` (SHA-256 `df7303d38ec840dd84b9ff3ce230dab64766ec35f869368336071666ea5d127b`). Training engine SHA-256: `6a2ab12001562413f62c19918ea66824308077ef2b4c141317a98c1c0ddea5b0`; adapter module SHA-256: `8f7e91a3465a1f88769726610ad1bb29ca137c3a6fb91f654a51a88af08a9137`. The task manifest SHA-256 is `bb50c7e4d5b8d0900e21d7ae64235c6d955244cc3bfdca1a97ab6056596a334a`.

## HR-development results

| Task / family | HR scratch m=4 | HR scratch m=16 | Matched path-source | m=16 / path-source | m=16 GPU-h |
|---|---:|---:|---:|---:|---:|
| 0 / high-frequency | 0.64071624 | 0.15876050 | 0.06831793 | 2.324× | 2.057014 |
| 1 / amplitude-OOD | 0.30019393 | 0.09559203 | 0.04473285 | 2.137× | 2.051367 |
| 3 / localized-bump | 0.48594406 | 0.30002690 | 0.06021613 | 4.983× | 2.040271 |
| 5 / piecewise-smooth | 0.34155925 | 0.10269669 | 0.05135978 | 2.000× | 2.545329 |
| **Equal-task mean** | **0.44210337** | **0.16426903** | **0.05615667** | **2.925×** | **8.693981 total** |

Moving from `m=4` to `m=16` reduced the direct-HR scratch mean error by 62.84%; every family improved, by 38.26%–75.22%. The larger-support scratch model still had 2.925× the mean error of path-source and did not reach its matched path-source reference in any task during 500 epochs. All four match times are right-censored at the completed training budget. The hardest residual is task03/localized-bump (0.3000 vs 0.0602); task00/high-frequency is also above twice its matched reference (0.1588 vs 0.0683).

These four-family, one-seed values establish neither generalization over seeds/support draws nor an `m` learning curve for path-source or ordinary transfer. The m=16 arm uses four times as many target HR examples as m=4, while both use 500 epochs; it does not isolate support count from resolution or show expert saturation.

## Measured costs

- Direct HR support data: 64 converged target HR rollouts (16 per task; 1,280 transitions total). Generation report: `artifacts/reports/phase1_hr_scratch_data_generation_hrdev_20261015_hrsupport_m16_s271829.json`; orchestration wall `4786.624 s` (1.329618 h), solver time `1609.029 s` (0.446952 h). CPU wall is reported separately and is not converted to GPU-hours.
- Direct HR scratch training: sum of task elapsed/GPU charges `8.693981 GPU-h` across the parallelized queue. This is `3.291×` the prior m=4 queue (`2.642104 GPU-h`). The full HR label-generation cost is an additional prerequisite for direct training.
- Path-source has a one-time seed-0 source-pretraining cost of `2.068094 GPU-h` and four target LR adaptations totaling `0.486617 GPU-h`; the shared 96-case six-family LR support pool used `349.981 s` wall time. The source pretraining can be amortized across target boundaries. Existing reports do not provide a complete, code-hashed HR inference timing comparison, so no HR inference speed or break-even claim is made here.
- Historical generation wall time for the cached HR source-training dataset is not recoverable from the artifacts and is excluded, not assumed to be zero.

## Ordinary non-path LR baseline context

On the separate seed42, four-task, m=4 slice, ordinary source + plain LR-NLT has mean HR-dev error `0.062196` and target adaptation cost `0.078080 GPU-h`. Path-source + LR-NLT has mean `0.048561` and target adaptation cost `0.494669 GPU-h` on the same tasks/support draw. The path route reduces error by 21.9% in this one-seed slice; ordinary LR adaptation uses 6.3× fewer target GPU-hours. This supports the speed/accuracy tradeoff for the plain baseline, but it is not a matched comparison to the new seed0 scratch run.

The stronger ordinary full-FT + comm256 comparison is much closer: across three seeds and six families its mean is `0.054881`, versus `0.053472` for path-source + comm256; the gap is about 2.6% and the paired interval crosses zero. Therefore the plain ordinary LR baseline is weaker, but the strongest ordinary control does not establish a stable path-specific accuracy advantage.

The direct-HR m=16 model is not yet an HR expert: this is one support size and a single 500-epoch schedule. The separately frozen source-initialized HR full-FT expert is now running with 250 epochs over the same four tasks, subject to its 4.7 GPU-h cap; that run is budget-limited and cannot establish saturation.

## Reproduction and raw evidence

- Prepared HR support: `scripts/prepare_phase1_hr_scratch_m16_dev.py --config configs/phase1_hr_scratch_m16_dev.yaml` (already completed; regenerating data is not needed to inspect results).
- Training: `CUDA_VISIBLE_DEVICES=1 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_hr_scratch_m16_dev.py --config configs/phase1_hr_scratch_m16_dev.yaml`.
- Four task reports and run summary: `results/phase1/HRSCRATCHM16/hr_scratch_from_zero_hr256_m16_s0_ss271829/`.
- Compact inference checkpoints: `artifacts/checkpoints/phase1/HRSCRATCHM16/hr_scratch_from_zero_hr256_m16_s0_ss271829/`.
- Machine-readable aggregate: `results/phase1/HRSCRATCHM16/hr_scratch_from_zero_hr256_m16_s0_ss271829/summary.json`.
- Official terminal-ledger snapshot: `artifacts/reports/phase1_progress_after_m16_complete_20261016_0741.json`.
- Independent task03/aggregate audit: `docs/agents/REVIEW.md` (2026-10-16); report/checkpoint/index seals/reference/ledger and aggregate mean/cost were recomputed with no mismatch.
