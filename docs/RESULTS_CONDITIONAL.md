# Results: conditional policy and softmax architecture

*Working note, written 2026-08-03, from a single seed (train_seed_idx=0,
probe_seed_idx=0). PRELIMINARY. Nothing here is paper-ready until the 3x3 seed
grid the linear Stage-1/Stage-2 results already use has been run. One seed's
frontier crossing is not evidence.*

Source files: `artifacts/stage2_conditional_regression.json` (linear,
closed-form), `..._reinforce_control.json` (linear, REINFORCE),
`..._softmax_sm.json` (ATTN-SM, REINFORCE), `results_regression_softmax.csv`.

## 1. The conditional policy beats the scalar on both linear architectures

At matched preservation cost the per-token policy dominates the single-theta
policy at every operating point measured (`cond_beats_scalar_at_matched_eps =
1.0`), and reaches chance where the scalar does not.

| arch | best scalar AUC | best cond AUC | eps ratio cond/scalar |
|---|---|---|---|
| ATTN-S | 0.527 | 0.4996 | 0.84 |
| ATTN-M | 0.511 | 0.4999 | 0.71 |

So on ATTN-M the conditional policy reaches chance for 71% of the scalar's
preservation cost. The numeric lever matched the closed-form lever to 4e-7
(`lever_max_abs_err_vs_closed_form`), so the diagnostic underneath these numbers
is sound.

## 2. The mechanism is not the same on the two architectures

This is the reason the targeting/polarisation split was worth building: the two
architectures win the same comparison for different reasons.

**ATTN-S targets, as predicted.** At its closest-to-chance point
`targeting_T = 1.74` (and 4.5-6.2x at tighter budgets), Spearman +0.45,
`polarisation_R = 0.22`. It puts flip budget on the high-lever tokens and drives
theta toward 0/1. This is the predicted quadrant: budget spent where the
membership signal is.

**ATTN-M wins by polarisation, not targeting.** At its closest-to-chance point
`targeting_T = 0.79` -- below 1 -- and Spearman -0.13. It is not concentrating on
the high-lever tokens there. Its advantage comes from polarisation
(`R = 0.36`): pushing theta to the extremes buys the same mean shift with less
flip variance, and variance is the masking channel that moves AUC without
removing anything. At looser budgets ATTN-M's T does rise above 1, so the
picture is mixed, but the headline operating point is a polarisation story.

`lever_gini ~ 0.54` on both, so the levers are genuinely unequal and targeting
was possible in principle. ATTN-M's failure to target at its best point is a
real, modest negative, not an artefact of uniform levers.

Honest one-line summary: per-token control helps on both architectures, but only
on ATTN-S is the help explained by "spend budget where the signal is".

## 3. The softmax attack barely works, so its unlearning result is not usable

From the sweep CSV, the ATTN-SM baseline (no corruption) sign-aligned residual
AUC is **0.444** across the nine seeds -- only 0.056 from chance, and on the
wrong side of it. Baseline eps is 0.0024: full and oracle are already almost
identical on the residual law. Across the entire grid the residual AUC stays in
0.44-0.51.

Driving that to 0.5 is removing a signal that was barely present, so the Stage-2
softmax numbers (`T = 0.77`, `cond_beats = 0.875`) are motion around chance and
should not be reported as an unlearning finding. This is the risk
`RUN_SOFTMAX.md` names: a negative result on ATTN-SM is only interesting if the
attack worked on it first, and here it did not.

Two things are worth keeping:

- The **loss** observable baseline is 0.545, above chance. There is membership
  signal on softmax attention -- it just does not live in the sign-aligned
  residual. The direction assumption that holds for linear attention (full model
  fits the forget query better, so its residual is smaller and its membership
  score higher) does not carry over. That is a genuine architecture-specific
  finding, and the cleanest thing this run produced.
- It confirms the linear intuition cannot be reused blindly, which is the kind
  of boundary worth stating.

## 4. The REINFORCE control is much weaker, which bounds what softmax can show

Running the linear archs under REINFORCE (to hold the estimator fixed against
the softmax comparison) shows the estimator itself costs a lot. On ATTN-M the
scalar policy under REINFORCE got stuck at AUC 0.538 and never reached chance.
The reported `eps_ratio = 2.19` there is therefore misleading -- it compares
against a scalar that simply failed, at near-zero eps.

Consequence for the writeup: the softmax-vs-linear comparison must be
softmax-REINFORCE vs linear-REINFORCE, never vs the closed-form linear numbers.
And because REINFORCE struggles even on the linear scalar, this single-seed run
cannot cleanly separate "softmax is harder to unlearn" from "REINFORCE
optimises worse". The weak softmax baseline (section 3) is the solid part,
because it comes from the sweep and does not depend on the estimator.

## 5. A metric bug found and fixed

The REINFORCE control reported `polarisation_R = 1.0e21`. R normalises flip
variance against a uniform policy at the same budget, and as thetabar -> 1 the
denominator collapses. The old code clamped it to 1e-30 and returned a huge
number; a near-deterministic policy has no variance channel to normalise
against, so R is undefined there, not enormous. `diagnose.describe_policy` now
drops probe points with `thetabar(1-thetabar) < 1e-4`, reports the surviving
fraction as `polarisation_frac_usable`, and returns NaN when none survive --
matching how `targeting_T` already handles a collapsed baseline.
`tests/verify_diagnose.py` has a regression check (theta=1 gives NaN, not 1e21).
The closed-form linear R values in section 2 keep thetabar mid-range and are
unaffected.

## What to run next

1. The 3x3 seed grid for section 1-2, so the frontier crossing is a median and
   not one seed. This is the blocker on anything paper-facing.
2. Re-run the REINFORCE jobs after the R fix so the polarisation column is
   readable rather than 1e21.
3. Optional: report the softmax result on the **loss** observable instead of the
   residual, where there is actually a baseline signal to remove. That turns
   section 3 from "the attack did not work" into a testable unlearning question
   on the observable that does separate.
