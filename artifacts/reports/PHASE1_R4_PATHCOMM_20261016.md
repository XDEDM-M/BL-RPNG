# R4: target path supervision with commutator calibration

Development-only paired analysis of the frozen R4 matrix. All LR checkpoints were selected on LR validation before these HR-dev evaluations. No confirmation query was read.

## Overall HR-dev results

| Arm | Mean rollout relative L2 | Worst family mean | Seed means (0 / 1 / 42) |
|---|---:|---:|---|
| A0 `pilot_pathsource_comm256` | 0.05347209 | 0.06726502 | 0.05736148 / 0.05100475 / 0.05205003 |
| A1 `pathsource_firstpath_ratio005_comm256-r4full` | 0.05387774 | 0.06849161 | 0.05810308 / 0.05162977 / 0.05190039 |
| A2 `pathsource_endpoint_ratio005_comm256-r4full` | 0.05363236 | 0.06724094 | 0.05732694 / 0.05178592 / 0.05178422 |

## Paired comparisons

Positive improvement means the candidate in the comparison label has lower error than its reference. Confidence intervals are paired t intervals over the three seed-level six-family means (df=2); they are descriptive and are not confirmation evidence.

| Comparison | Absolute mean improvement | Relative improvement | Seed wins | Family wins | 95% paired t interval |
|---|---:|---:|---:|---:|---:|
| A1_vs_A0 | -0.00040566 | -0.759% | 1/3 | 1/6 | [-0.00160904, 0.00079773] |
| A1_vs_A2 | -0.00024538 | -0.458% | 1/3 | 2/6 | [-0.00143625, 0.00094549] |
| A2_vs_A0 | -0.00016027 | -0.300% | 2/3 | 3/6 | [-0.00152656, 0.00120601] |

A1 meets the pre-evaluation repeatable/practical development rule: **False**. The rule requires at least 1% lower overall error than both comparators, all three seed means to favor A1, and at least four of six family means to favor A1 against each comparator.

## Boundary-family means

| Family | A0 | A1 | A2 | A1 improvement vs A0 | A1 improvement vs A2 |
|---|---:|---:|---:|---:|---:|
| high_frequency | 0.05685483 | 0.05701124 | 0.05735016 | -0.00015642 | 0.00033892 |
| amplitude_ood | 0.03519925 | 0.03667194 | 0.03567894 | -0.00147269 | -0.00099300 |
| matern_gp | 0.04937875 | 0.04887582 | 0.04944356 | 0.00050293 | 0.00056774 |
| localized_bump | 0.05947680 | 0.05947893 | 0.05947574 | -0.00000214 | -0.00000319 |
| mixed_fourier | 0.06726502 | 0.06849161 | 0.06724094 | -0.00122659 | -0.00125067 |
| piecewise_smooth | 0.05265788 | 0.05273691 | 0.05260482 | -0.00007903 | -0.00013209 |

## Every paired task×seed result

| Seed | Task | Family | A0 | A1 | A2 | A1 improvement vs A0 | A1 improvement vs A2 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | high_frequency | 0.06831793 | 0.06884113 | 0.06876653 | -0.00052321 | -0.00007460 |
| 0 | 1 | amplitude_ood | 0.04473285 | 0.04478349 | 0.04431659 | -0.00005064 | -0.00046691 |
| 0 | 2 | matern_gp | 0.04867816 | 0.04821593 | 0.04868719 | 0.00046223 | 0.00047126 |
| 0 | 3 | localized_bump | 0.06021613 | 0.06006049 | 0.05986925 | 0.00015564 | -0.00019124 |
| 0 | 4 | mixed_fourier | 0.07086402 | 0.07526116 | 0.07127251 | -0.00439714 | -0.00398865 |
| 0 | 5 | piecewise_smooth | 0.05135978 | 0.05145627 | 0.05104960 | -0.00009649 | -0.00040668 |
| 1 | 0 | high_frequency | 0.05184649 | 0.05194562 | 0.05237268 | -0.00009913 | 0.00042706 |
| 1 | 1 | amplitude_ood | 0.02836940 | 0.03191456 | 0.03258159 | -0.00354516 | 0.00066703 |
| 1 | 2 | matern_gp | 0.04754582 | 0.04735083 | 0.04752313 | 0.00019500 | 0.00017230 |
| 1 | 3 | localized_bump | 0.06014429 | 0.06015622 | 0.06015183 | -0.00001193 | -0.00000438 |
| 1 | 4 | mixed_fourier | 0.06478832 | 0.06523104 | 0.06485595 | -0.00044272 | -0.00037510 |
| 1 | 5 | piecewise_smooth | 0.05333420 | 0.05318033 | 0.05323033 | 0.00015386 | 0.00005000 |
| 42 | 0 | high_frequency | 0.05040007 | 0.05024698 | 0.05091127 | 0.00015309 | 0.00066429 |
| 42 | 1 | amplitude_ood | 0.03249549 | 0.03331778 | 0.03013864 | -0.00082228 | -0.00317914 |
| 42 | 2 | matern_gp | 0.05191226 | 0.05106070 | 0.05212037 | 0.00085156 | 0.00105967 |
| 42 | 3 | localized_bump | 0.05806998 | 0.05822009 | 0.05840615 | -0.00015011 | 0.00018605 |
| 42 | 4 | mixed_fourier | 0.06614273 | 0.06498263 | 0.06559437 | 0.00116010 | 0.00061174 |
| 42 | 5 | piecewise_smooth | 0.05327967 | 0.05357414 | 0.05353454 | -0.00029446 | -0.00003960 |

## Evidence and limits

- Machine-readable paired summary: `artifacts/reports/phase1_r4_pathcomm_paired_summary_20261016.json`.
- `scripts/summarize_phase1_hrdev.py` verifies each report task result, checkpoint, support/validation index seals, HR-dev manifest, and query flags. This R4 pair analysis additionally requires matching support/LR-validation/HR-validation index paths and seals plus recorded source-checkpoint paths and SHA values for each paired seed×task; it records those seals per cell.
- Raw R4 task reports: `results/phase1/R4/`; HR-dev outputs: `artifacts/reports/phase1_hrdev_evaluations/`.
- The six tasks represent six development boundary families and use one common support seed; each task has its own HR-support index, paired across A0/A1/A2. The three model seeds are repeated measurements, not 18 independent tasks. Do not generalize this comparison to the confirmation population.
- Query access: `false`; HR-dev did not influence training or within-task checkpoint selection.
