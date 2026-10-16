# BL-RPNG experiment status

Current project dashboard: 2026-10-16. The current Phase-1 follow-up and
historical cross-PDE snapshot are separated below. The latest full study is
reported in [`reports/PHASE1_FINAL.md`](../reports/PHASE1_FINAL.md), with
chronology in `docs/LOG.md`, the append-only task ledger in
`docs/EXPERIMENTS.csv`, and independent checks in `docs/agents/REVIEW.md`.

## Current Phase-1 follow-up (2026-10-16)

The development campaign directly tested multi-boundary path-source
pretraining, four-rollout low-resolution adaptation on held-out boundary
instances, and 256² high-resolution inference. A frozen six-family shortlist
completed 108/108 LR adaptation and HR-development cells across model seeds
`[0,1,42]`; the source/checkpoint/data seals, LR-only selection, and query flags
were independently audited. The current confirmation query remains closed.

- The path-source + direct256 consistency candidate had mean HR-dev relative
  L2 `0.053472` across the three model seeds and six fixed development families;
  lifted LR scratch had `0.509044`. Transfer therefore gives a large, consistent
  improvement over scratch in this development panel.
- The ordinary final-only source + full-FT + comm256 comparator at LR=`1e-4`
  averaged `0.054881`, so the path-source candidate's point estimate was 2.57%
  lower. Its paired three-seed interval includes zero; treat this as a promising
  lead rather than a confirmed advantage.
- Matched one-seed/four-family source comparisons favored path pretraining
  under both direct256 adaptation (7.48% lower HR-dev error than ordinary
  source) and H3+comm256 adaptation (8.08% lower). These are development
  comparisons, not independent multi-seed confirmation.
- The ordinary baseline was then tuned on the same six-family support and
  validation protocol. LR=`3e-5` gave seed0 HR-dev `0.044725` and its
  predeclared seed1 replication `0.044410`; the matched two-seed mean
  `0.044568` is lower than matched ordinary LR=`1e-4` (`0.056317`) and
  path-source C256 (`0.054183`). This shows baseline tuning materially changes
  the ranking; do not present the earlier 2.57% edge as robust to tuning.
- True target first-accepted-path fine-tuning was compared against endpoint,
  endpoint-direction and final-only controls. It performed competitively but
  did not show a repeatable path-specific gain under the current Phase-1
  protocol. A distinct earlier frozen experiment found that, at an eight-
  boundary budget, annealed path fine-tuning improved HR error by 3.34% over
  same-initialization final-only and by 10.86% over ordinary pretraining;
  this is a positive schedule-specific result, not a replication of the
  per-target Phase-1 protocol. Details: `artifacts/reports/FORMAL_SEED42_RESULTS.md`.
- Boundary lifting enforces the Dirichlet trace exactly. HR scratch with
  `m=4` and `m=16`, plus source-initialized HR full-FT `m=16`, remained above
  the matched low-resolution transfer result on the four tested tasks; these
  finite runs are useful practical baselines, not saturated HR oracles.

The complete three-method confirmation matrix, few-shot curve, expert
saturation, complete cost break-even and full mechanism controls were not
completed within the campaign allocation. The final report records the
remaining evidence gap and budget without changing the positive development
findings: [`reports/PHASE1_FINAL.md`](../reports/PHASE1_FINAL.md).

## Historical frozen snapshot (2026-10-12)

## Method frozen for evaluation

The primary candidate is **BL-RPNG-NLT**:

1. impose non-homogeneous Dirichlet data through a discrete boundary lift;
2. learn the zero-trace residual operator;
3. use accepted Newton-path information only as privileged source-pretraining
   supervision;
4. adapt to a target boundary with final-only neuron-linear transformation
   (NLT), without target high-resolution solution labels.

The target-stage NLT+path, commutator and fine-defect variants remain
ablations. None consistently displaced final-only NLT on a frozen blind set.

## Evidence by PDE

| PDE/stage | Status | Frozen result | Permitted interpretation |
|---|---|---|---|
| Allen--Cahn | Strong seed-42 confirmation | 256-square, 20-step, `m=4`: BL-RPNG-NLT relL2 5.223%; ordinary-pretrain+NLT 6.053%; lifted LR scratch 54.959% | Strong evidence for boundary lifting, source Newton-path pretraining and low-dimensional target adaptation under one seed |
| Poisson | Linear negative control passed | ordinary+NLT 0.820%; path+NLT 0.849%; converged LR scratch 54.738% | Pretraining/lifting transfer strongly; Newton-path supervision adds no material benefit when Newton is one-step linear |
| Gray--Scott | Mixed/limited | Safe router 256-square, 50-step balanced relL2 23.44%; always-final 25.49%; always-path 28.86%; superiority CI crosses zero | Practical routing signal only; no Newton-specific mechanism claim in the tested `dt=1` regime |
| Transient incompressible NS | Numerical gate only | Natural BDF1 candidate at `dt=0.07`, `Re=800`; pressure path gate passes on 64/128, velocity path remains endpoint-redundant | A credible candidate protocol exists, but no neural-operator performance claim has been tested |
| Steady NS Reynolds continuation | Supplementary numerical pilot | Natural warm-start has non-redundant Newton paths | Mechanism supplement only; it is not physical-time autoregression |

## Allen--Cahn primary result

The fresh confirmation contains six unseen boundary families, 48 target tasks
and 768 paired query rollouts. At 256-square resolution and horizon 20:

- BL-RPNG-NLT mean relL2: **0.052226**;
- ordinary source pretraining + NLT: **0.060533**;
- lifted target-only LR scratch: **0.549589**;
- reduction versus LR scratch: **90.50%**, 95% CI
  **[89.86%, 91.08%]**;
- reduction versus ordinary pretraining: **13.72%**, 95% CI
  **[11.88%, 15.53%]**;
- task wins: **48/48** against both baselines;
- 256-square, 50-step mean relL2: **0.062473**;
- target setup speedup versus the resource-capped HR reference: **1350x**,
  95% CI **[1288x, 1437x]**.

The speedup is an engineering comparison against the measured resource-capped
HR specialist workflow. The HR specialist did not reach data or optimization
saturation, so this snapshot makes **no saturated-oracle parity claim**.

## Poisson negative control

At 256-square resolution over 480 paired query units:

- ordinary pretraining + NLT: **0.008205** mean relL2;
- Newton-path pretraining + NLT: **0.008487**;
- converged target-only LR scratch: **0.547385**;
- ordinary pretraining reduces error versus LR scratch by **98.50%**;
- the absolute path/ordinary difference is below the preregistered 5% bound.

This supports the falsifiable mechanism: lifting and reusable pretraining
remain valuable, while a linear one-step Newton path contributes no special
nonlinear trajectory information.

## Gray--Scott conclusion

The final R5 evaluation uses five development boundaries and two new frozen
initial conditions per boundary. The prespecified safe router improves the
256-square, 50-step point estimate by 8.03% versus always-final and by 18.79%
versus always-path. Conditional confidence intervals include zero, the
boundaries are not new, and only seed 42 was run. Accepted Newton corrections
in this regime are numerically close to endpoint increments; consequently the
experiment does not validate the Newton-specific mechanism.

## Transient Navier--Stokes conclusion

The accepted numerical candidate uses P2/P1 Taylor--Hood FEM, BDF1,
`dt=0.07`, `Re=800`, grad-div 100, natural previous-state Newton
initialization and divergence-free forcing. The bounded 20-step gate passed at
64 and 128 output resolution. Pressure first-accepted corrections contain
intermittent non-collinear information, whereas velocity corrections remain
almost endpoint-redundant. The 64-to-128 pressure discrepancy is large, so
low-resolution pressure paths must not supervise the target adapter.

The following hypotheses remain untested:

- masked high-resolution pressure-path pretraining beats a matched endpoint
  control on unseen-boundary LR-to-HR adaptation;
- pressure-path information improves velocity rollout through incompressible
  coupling;
- the advantage remains stable for 50-step autoregression.

## Claim limits

- All headline neural results are seed 42; seeds 0/1/2/3 remain pending.
- Allen--Cahn HR references are resource-capped and unsaturated.
- Gray--Scott does not establish a Newton-path mechanism.
- Transient NS has solver evidence only, not trained-model evidence.
- Exact boundary satisfaction is an implementation invariant; it is not, by
  itself, evidence of predictive accuracy.

Machine-readable frozen records and representative figures are indexed in
`docs/experiment_records/index.json` and `docs/figures/README.md`.
