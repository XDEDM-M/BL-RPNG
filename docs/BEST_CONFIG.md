# Best configuration

No new champion has been selected in this campaign. The historic seed-42
BL-RPNG-NLT result is recorded in `docs/EXPERIMENT_STATUS.md`; its HR query was
already opened and will not be used to select new settings.

## Historic source model

- Checkpoint: `artifacts/checkpoints/allen_cahn_formal_a_blrpng_seed42.pt`
- Source supervision: accepted Newton path at normalized log-residual progress 0.25.
- Historic target adaptation: NLT final-only, m=4 LR rollouts.
- Historic evaluation: 256², 20-step mean relative L2 0.052226; seed 42 only.

## Campaign status

No confirmed champion yet. The best evaluated three-seed candidate for the primary multi-boundary comparison remains `pilot_pathsource_comm256`: multi-boundary path-source pretraining followed by final-only LR adaptation with 64-to-256 commutator consistency. Its six-family HR-dev mean is `0.053472088528` (tasks 0–5, support seed 271829). The Stage-2 ordinary-pretraining/full-FT + comm256 baseline at LR=`1e-4` is `0.054881171919`, 2.57% higher in error; the paired interval spans zero. This is a small, unresolved development difference, not a confirmation claim. The path-source direct64-to256 two-step consistency candidate below remains the best four-family single-seed result (`0.052879281671`), not the promoted primary candidate.

R5 subsequently gave the ordinary full-FT + comm256 baseline a local target-LR bracket `{3e-5,1e-4,3e-4}`. LR=`3e-5` was strongest on the six-family seed0 screen (`0.044725204533` HR-dev) and reproduced on predeclared seed1 (`0.044410349642`); the matched two-seed mean is `0.044567777087`. On the same seed0/1 subset, ordinary LR=`1e-4` averages `0.056316934007`, and path-source C256 averages `0.054183115603`. These descriptive two-seed results suggest the earlier path-source edge may disappear after baseline tuning, but they do not replace the three-seed Stage-2 aggregates or establish a stable ranking. They use one support draw and six fixed development families. Full evidence: `artifacts/reports/PHASE1_BASELINE_LR3E5_SCREEN_20261016.md` and `artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md`.

## Best currently reproducible ordinary baseline candidate (development only)

- Target config: `configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml`; protocol card: `docs/experiment_cards/BASELINE_FULLCOMM256_LR3E5_SEED1_REPLICATION_20261016.md`.
- Source initialization for model seed 1: `artifacts/checkpoints/phase1/R1/source_final_s1_scratch.pt` (SHA-256 `8b9509f482b2029c0c12302449c93509762991e72a3343978a2bd2b55273507b`).
- Selected per-task compact target checkpoints for seed1: `artifacts/checkpoints/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5seed1rep-task{00..05}_m4_s1_ss271829/`.
- Seed0 comparison checkpoints: `artifacts/checkpoints/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5screen_m4_s0_ss271829/`.
- Reproduce each fit with `scripts/run_phase1_trials.py`, one task at a time, using the command template in the card; after all six fits, evaluate with `scripts/evaluate_phase1_hrdev.py`. Recompute the paired table with `scripts/summarize_phase1_r5_lr3e5_replication.py`.
- This is the current low-error development candidate, not a confirmed or promoted campaign champion. Checkpoint selection used LR-dev only; HR-dev was evaluated after the six fits.

The focused R4 interaction tested first-accepted Newton-path and endpoint target supervision on top of comm256, with three paired model seeds and six tasks. HR-dev means were A1 first-path `0.05387774431082006`, A2 endpoint `0.05363236245983796`, and A0 final-only `0.05347208852763286`. A1 failed the frozen practical/repeatability rule against both controls; do not continue path-specific search on this evidence. Independent audit checked all 54 reports and their task/checkpoint/index/source hashes; details are in `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md` and `docs/agents/REVIEW.md`.

## R1 screening leaders (provisional; not promoted)

- Ordinary transfer, final-only, `lr=1e-3`, cosine-100: four-task mean LR-validation rollout relative L2 `0.03420821`, seed 42, m=4, support seed 271829. Summary: `results/phase1/R1/ordinary_lr1e-3_cosine100_m4_s42_ss271829/summary.json`.
- Path-source pretraining followed by final-only adaptation, `lr=1e-3`, cosine-100: four-task mean LR-validation rollout relative L2 `0.02791603`, seed 42, m=4, support seed 271829. This is 18.39% lower than the ordinary-transfer candidate on the same four LR development tasks. Summary: `results/phase1/R1/final_lr1e-3_cosine100_m4_s42_ss271829/summary.json`; config SHA `a62c6619e4b54a8377ae8d0132547ff84427bfe1fb673a85fa02fbc21d02ad3e`.
- Both values are single-seed LR-screen results on tasks 0/1/3/5. HR development was evaluated subsequently; neither result is a campaign champion. Promote only after full-family and multi-seed development evaluation.
- True target path-FT from the same path-source initialization has now been tested at `lr=1e-3`, cosine-100, with gradient-ratio settings 0.05/0.15/0.30. Four-task LR-validation means were `0.02792730`, `0.02804492`, and `0.02819331`, all slightly worse than paired final-only `0.02791603`. The ratio-0.05 arm is the best path-FT candidate so far but is not a leader or champion. Evidence: `results/phase1/R1/path_lr1e-3_ratio{005,015,030}_m4_s42_ss271829/summary.json`.
- Retrospective, independently evaluated HR-development means for the same path-FT ratio .05/.15/.30 arms are `0.05524994/0.05579156/0.05504368`, versus paired final-only `0.05602457`; same-layer endpoint L1 A2 at ratio .05 gives `0.05607655`. The largest measured reduction is 1.75% at ratio .30, despite worse LR validation. Treat this only as a one-seed/one-support development signal. It needs a clean multi-seed/support replication and does not promote a champion.
- Mechanism qualification: retained LR accepted-path targets are almost collinear with endpoint increments, so any later A1 benefit requires a same-hook endpoint A2 control before it can be attributed to nontrivial Newton geometry.

## Four-family HR development findings (seed 42, support 271829)

- Path-source final-only `lr=1e-3/cos100`: mean 0.05602456698252354.
- Ordinary-source final-only `lr=1e-3/cos100`: mean 0.06219576909206807. Path-source mean is 9.9222% lower, but piecewise-smooth is worse (0.05272269 versus 0.04629440).
- Path-source final-only `lr=3e-4/cos100`: mean 0.05648068202135619. No improvement over the 1e-3 candidate.
- Best of three scratch cosine-100 LR screens: `lr=1e-4`, LR mean 0.5347863933304324; HR development mean 0.6103115785401314. Additional general training improvements still need fair scratch/transfer checks.
- R2 path-source NLT with 64-to-128 one-step commutator, weight 0.05: LR mean 0.027985551886376925, HR-development mean 0.0543033644760726 (3.0722% lower than R1 path-source final-only). All four HR families improve. This is the provisional development candidate, pending six-family/three-seed promotion.
- Raw evidence: `artifacts/reports/phase1_hrdev_evaluations/<trial>_task{00,01,03,05}_s42_ss271829.json`; training selections remain LR-only. These are development findings and do not meet confirmation evidence requirements.
- Ordinary full FT final-only, `lr=1e-4/cos100`: LR-dev mean `0.029712766977`, HR-dev mean `0.062037663115`; `lr=3e-5` gives `0.030261243111/0.062978398058`. Its direct256 consistency arm at LR `1e-4`, weight `0.05`, has LR/HR means `0.03027409/0.05587599`, improving the prior ordinary NLT+direct256 result by 2.24%; it remains 5.67% behind the path-source direct256 candidate. This is the strongest measured ordinary baseline so far, while path-source direct256 remains the overall provisional development leader. See `docs/experiment_cards/BASELINE_FULLCOMM256.md`.
- LR scratch cosine-500 with `wd=1e-4`: best tested lr `3e-4`, LR mean `0.433207295847`, versus `0.462373684091` at lr `1e-4` and previous 100-epoch best `0.534786393330`. Longer regularized training helps but leaves a large gap to transfer; no HR evaluation was warranted from the LR screen.
- R2 direct64-to256 two-step consistency weight0.05 completed: LR mean `0.02810983473464148`, HR-development mean `0.052879281670902854` (5.6141% better than R1; 14.9793% better than ordinary transfer). HR per family0/1/3/5 is `0.06769019/0.03647126/0.05997935/0.04737632`. Four-task adaptation charge `0.4948127285383321` GPU-hours. This is the provisional lowest measured four-family HR-development error, with higher training cost and no three-seed/six-family promotion yet. Timed reports: `comm256_lr1e-3_cw005_r2_task*_s42_ss271829.json`.
- Ordinary-source direct256 consistency, `ordinary_comm256_lr1e-3_cw005_r2-baseline1`: LR-dev mean `0.034392807039`, HR-dev mean `0.057153851914`, four tasks, seed42/support271829, cost `0.49436767` GPU-h. The matched path-source direct256 candidate is `0.052879281671` HR, 7.48% lower error; this ordinary candidate still improves on the earlier ordinary NLT baseline `0.06219577`. Reports and interpretation: `docs/experiment_cards/R2.md`.
- R2B H3 plus direct256 consistency paired screen: path-source LR/HR `0.02660829/0.05384847`; ordinary-source `0.03257527/0.05858175`. The matched path-source result is 8.08% better than ordinary-source but 1.83% worse than the simpler path-source direct256 candidate above, so it is not the development leader. See `docs/experiment_cards/R2B.md`.
- R2 fixed-H3 rollout supervision is now scored: LR-dev mean `0.02642924`, HR-dev mean `0.05596649`. It is the best LR screen so far but is 5.84% worse than direct256 on HR development, so it is not promoted. Ordinary-H3 and scratch-H3 controls remain pending. Reports and full task outputs are listed in `docs/experiment_cards/R2.md`.
- R3/R4 first-accepted-path target fine-tuning and its matched endpoint controls are complete. The R4 paired development evidence does not show a repeatable target-path advantage over final-only or endpoint supervision; any future path-specific claim needs new evidence under a frozen, adequately powered design.
