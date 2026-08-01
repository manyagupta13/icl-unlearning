# Running the conditional-policy experiment

*Working note, written 2026-08-02. This is the follow-up experiment flagged in
`report/paper.pdf` §Limitations: the conditional policy `p_theta(.|x_f, y_f)` is
implemented in `policy.py` but was never run.*

## What is being tested

Stage 2 as reported learns one flip probability for the whole forget group.
`ConditionalBernoulli` learns one per token. Two questions:

1. At matched preservation cost, does per-token control get closer to chance?
2. If so, is it because the policy concentrates its flip budget on the tokens
   with the largest lever on the hypothesis gap, as the moment argument predicts?

The second question is the point. A win with no measured targeting would mean
the explanation is wrong even though the number improved.

## The measurements

`diagnose.py` computes, for each token in the forget slice,

    L_i = sgn(y_q) . (-2/(N+1)) . ( E_s[a_i | M_full] - E_s[a_i | M_orac] )

which is exactly `d(mu1 - mu0) / d theta_i` in the Gaussian AUC of `theory.py`.
Two statistics summarise what the policy did with it:

| statistic | meaning | scalar policy gives | targeting policy gives |
|---|---|---|---|
| `targeting_T` | mean shift achieved / shift a scalar at the same average budget would achieve | 1 exactly | > 1 |
| `polarisation_R` | mean theta(1-theta) / thetabar(1-thetabar); flip variance relative to uniform | 1 exactly | < 1 |

`lever_gini` is the control: if the levers are all the same size there is nothing
to target and `T ~ 1` is the expected answer, not a failed policy.

Both statistics are 1 under a scalar policy by construction, so the scalar is the
null and any departure is something a scalar could not have done.

## Run it

Needs the cached ensembles (`artifacts/ensembles_{name}_ts{0,1,2}.pt`), so it runs
wherever those were trained. Paste into a Kaggle cell with a GPU attached:

```python
!git clone -q https://github.com/manyagupta13/icl-unlearning /kaggle/working/icl
%cd /kaggle/working/icl
!pip install -q -r requirements.txt

# ensembles live outside the repo (gitignored); point at wherever they are
!mkdir -p artifacts && cp /kaggle/input/<your-dataset>/ensembles_*.pt artifacts/

!python tests/verify_diagnose.py          # metric sanity, ~2s, no GPU

for cfg in ["configs/regression.yaml", "configs/mnist.yaml"]:
    for ts in range(3):
        !python scripts/stage2_conditional.py --config {cfg} \
            --train-seed-idx {ts} --probe-seed-idx 0 --out-suffix _ts{ts}
```

Three training seeds because a single-seed difference between two policies is not
evidence of anything. Report medians across seeds, the way Stage 1 does.

Output: `artifacts/stage2_conditional_{name}_ts{k}.json`, with three frontiers per
architecture (`fixed_grid`, `scalar`, `conditional`) and a `verdict` block.

## Reading the verdict block

- `cond_beats_scalar_at_matched_eps` — fraction of scalar operating points the
  conditional frontier dominates. Below about 0.5 the frontiers cross and no clean
  claim holds either way. That is a real outcome, not a failed run.
- `eps_ratio_cond_over_scalar` — preservation cost at each family's own closest
  approach to chance. Below 1 means cheaper unlearning.
- `targeting_T_at_best_conditional` — whether the mechanism story holds.

Four outcomes, all publishable, only one of them the expected one:

| | T ~ 1 | T > 1 |
|---|---|---|
| **no eps gain** | levers are uniform; check `lever_gini` before concluding anything | found the levers, no room left to use them |
| **eps gain** | gain came from somewhere else — the explanation is wrong | the predicted result |

## What to check before believing a positive result

- `lever_gini` well above 0. If the levers are near-uniform, `T > 1` is noise.
- The gain survives all three training seeds.
- `restart` variation: the MLP is non-convex and the script keeps the best of
  three restarts, so a result that only one restart finds is fragile.
- The conditional frontier beats `fixed_grid` too. Beating the learned scalar but
  not the no-learning grid would mean the scalar optimisation is what is broken.
