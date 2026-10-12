# BL-RPNG-NLT theoretical contract and falsifiable predictions

This document fixes the mechanism that the experiments are intended to test.
It deliberately separates exact algebraic statements, modelling assumptions,
and empirical hypotheses.

## 1. Boundary lifting is a change of operator coordinates

Let (V_h) be a discrete trial space, (V_{h,0}) its zero-Dirichlet subspace,
and (E_h g) the discrete harmonic extension of boundary data (g). The
admissible states form the affine space

\[
V_{h,g}=E_hg+V_{h,0}.
\]

The map (T_{h,g}:V_{h,g}\to V_{h,0}), (T_{h,g}u=u-E_hg), is a bijection.
Consequently a boundary-indexed time-step operator

\[
\mathcal S^h_g:V_{h,g}\to V_{h,g}
\]

has the conjugated representation

\[
\widetilde{\mathcal S}^h(v,E_hg)
=T_{h,g}\,\mathcal S^h_g(v+E_hg),
\qquad \widetilde{\mathcal S}^h:V_{h,0}\to V_{h,0}.
\]

This identity does not imply that all boundary tasks become identical. It says
that their inputs and outputs now live in a common linear space and that the
remaining boundary dependence is an explicit condition (E_hg), rather than
an implicit change of affine solution space.

The implemented output projection (P_{h,0}) gives

\[
\widehat u_{t+1}=E_hg+P_{h,0}F_\theta(v_t,E_hg),
\]

so the discrete Dirichlet trace is exactly (g), independent of training loss,
sample size, or inference resolution. This is an implementation invariant, not
an empirical accuracy claim.

## 2. Newton paths are privileged tangent-space information

For an implicit nonlinear time step, an accepted Newton correction satisfies

\[
J_h(u^{(k)})\,\delta^{(k)}=-R_h(u^{(k)}),
\qquad \delta^{(k)}\in V_{h,0}.
\]

The zero trace follows from enforcing the target Dirichlet data before solving
for free degrees of freedom. Thus both the learned residual state and Newton
corrections lie in (V_{h,0}). The progress-selected cumulative correction
contains information about the action of a local inverse Jacobian while
remaining a deterministic target given the PDE step and accepted solver path.

The normalized log-residual progress

\[
p_k=\frac{\log r_0-\log r_k}{\log r_0-\log r_*}
\]

is invariant to multiplication of all residual norms by a positive constant.
Selecting the first accepted update with (p_k\ge 0.25) therefore avoids tying
the auxiliary target to units or a solver-specific absolute tolerance.

Newton data are *privileged*: they are used during source training but are not
required at deployment. The primary method does not claim to replace Newton
iterations inside a numerical solver. It uses their local inverse information
to shape a reusable neural representation.

## 3. Why target adaptation is NLT

At layer \(\ell\), NLT applies

\[
h'_\ell=\gamma_\ell\odot h_\ell+\beta_\ell.
\]

The empirical hypothesis is that, after multi-boundary path pretraining, a new
boundary changes the useful feature basis mainly through channel importance and
offset, rather than requiring a new spectral basis. For width (w) and (L)
layers, NLT has (2Lw) affine parameters before the output head, versus spectral
updates whose size scales with (Lw^2m_1m_2). Under the usual bounded-feature
and local-Lipschitz assumptions, this lowers the target-task estimation term
from a dependence on the full parameter dimension to an effective adapter
dimension. This is a hypothesis to be tested by NLT/full/LoRA comparisons, not
an unconditional theorem.

Target-stage path supervision is not intrinsic to the primary method. It is
promoted only if it improves a frozen validation and blind protocol by a
material amount. The current exploratory gain is too small for that promotion.

## 4. Error decomposition used by the experiments

For a fine-grid query and a fixed rollout horizon, the working decomposition is

\[
\|u-\widehat u_h\|
\lesssim
\underbrace{C_{\rm disc}h^p}_{\text{numerical discretization}}
+\underbrace{\varepsilon_A}_{\text{source representation}}
+\underbrace{L_A\varepsilon_{\rm NLT}(m)}_{\text{few-shot estimation}}
+\underbrace{\varepsilon_{\rm comm}(H,h)}_{\text{resolution commutator}}
+\underbrace{\varepsilon_{\rm roll}}_{\text{temporal accumulation}}.
\]

The terms are operationalized rather than fitted after seeing the blind test:

- the fine numerical solver and a mesh-doubling subset estimate
  (C_{\rm disc}h^p);
- ordinary versus Newton-path source pretraining isolates
  (arepsilon_A);
- the (m=1,2,4,8) curve and adapter ablations probe
  (arepsilon_{\rm NLT}(m));
- restrict-predict versus predict-restrict measures
  (arepsilon_{\rm comm});
- 20/50-step curves measure accumulation and instability.

An oracle comparison is prohibited until optimization has plateaued and two
consecutive support doublings improve less than one percent. Otherwise the
apparent oracle gap conflates method quality with an under-trained specialist.

## 5. Falsifiable mechanism predictions

1. On linear Poisson, exact Newton converges in one update. Once lifting and NLT
   are controlled, path supervision should add no material gain (frozen bound:
   at most 5%).
2. On nonlinear PDEs, path pretraining should outperform final-only pretraining
   with the same target NLT adapter. If it does not, the claimed privileged-path
   mechanism is rejected.
3. `lift_conditioned` should outperform raw boundary augmentation when
   boundary-to-interior propagation matters, while residual lifting should
   further help if affine-space alignment is the operative mechanism.
4. NLT should be most competitive for small target support. If full fine-tuning
   dominates at (m=1\)–4 without overfitting, the low-dimensional feature
   reweighting hypothesis is rejected.
5. The method must transfer to the independent UNO/U-FNO implementation after
   the FNO gate. Failure there limits the claim to one backbone.
6. Exact boundary error alone is insufficient: near-boundary, interior,
   residual, long-rollout and cost metrics must pass simultaneously.

