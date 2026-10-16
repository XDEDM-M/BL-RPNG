# BL-RPNG consolidated results (2026-08-20)

| Method | 256²/20-step mean relative L2 | Status |
|---|---:|---|
| BL-RPNG-NLT | 0.052238 | exploratory best |
| BL-RPNG v2 LoRA | 0.060793 | frozen confirmatory |
| ordinary pretrain → final | 0.078432 | baseline |
| path-pretrain → fixed path full FT | 0.076434 | baseline |
| lifted LR scratch | 0.784797 | baseline |

The prior HR gate is a current-budget comparison, not a saturated-oracle result.
The previous NLT+path runs had zero effective path weight and are invalid for
assessing target-stage Newton-path supervision.
