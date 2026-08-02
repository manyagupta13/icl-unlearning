# Running the softmax-attention architecture

*Working note, written 2026-08-02. Second follow-up experiment. The paper's
Limitations section says "two datasets, one architecture family" -- ATTN-S and
ATTN-M are the same forward pass under two parameterisations of `M`, so they
are not really two architectures. `ATTN-SM` is.*

## The model

One layer of softmax self-attention on the same tokens `t_i = [x_i ; y_i]`:

    q = t_q W_Q,  k_i = t_i W_K,  v_i = t_i W_V
    alpha = softmax_i( q.k_i / sqrt(D+1) )
    yhat  = ( sum_i alpha_i v_i ) . w_out

This is the smallest change from `LinearAttnICL` that breaks the closed form.
Same tokens, same single layer, same query slot with its label zeroed, same
read-out position. The only difference is that the mixing weights come from a
softmax over the context instead of the constant `1/(N+1)`. Adding depth, an
MLP block or layer norm at the same time would confound the nonlinearity with
capacity, and the nonlinearity is the whole question.

Check 4 in `tests/verify_softmax.py` makes the relationship concrete: zeroing
`W_Q` makes `alpha` exactly uniform at `1/(N+1)`, so the two families meet in
the degenerate limit rather than merely resembling each other.

## What breaks, and why that is the interesting part

The linear model's prediction is linear in the labels. That single fact is what
gives `theory.py` exact moments and `policy.py` a differentiable AUC with no
sampling. Under softmax the attention weights depend on the labels too, so
flipping `y_i` moves `alpha` as well as `v`, the expectation over a Bernoulli
flip no longer factors, and nothing survives:

| | linear archs | ATTN-SM |
|---|---|---|
| `theory.predicted_auc` | exact | raises `TypeError` |
| `policy.policy_auc` (differentiable AUC) | exact | raises `TypeError` |
| Stage 2 optimiser | closed-form gradient | REINFORCE |
| per-token lever | closed form or numeric, identical | numeric only |
| sweep column `auc_theory_residual` | filled | `NaN` |

These raise rather than degrade. A wrong AUC does not look wrong downstream, so
returning a plausible number would be the worst available behaviour.

`diagnose.token_lever_numeric` is what keeps the targeting diagnostic alive: it
measures `L_i` by flipping one token at a time instead of deriving it. On the
linear archs it reproduces the closed form to floating-point precision — that
equivalence is check 3 in `tests/verify_softmax.py`, and it is the reason a
cross-architecture comparison of `targeting_T` means anything at all.

One honest caveat, stated in the code as well: for a nonlinear model the
numeric lever is the derivative of the mean shift at `theta = 0`. A second
simultaneous flip no longer adds independently, so it is a local quantity away
from small budgets. For the linear archs no caveat applies.

## The confound you have to control for

ATTN-M optimises through the closed form; ATTN-SM has to use REINFORCE.
REINFORCE is high variance. So if ATTN-SM reaches chance less cleanly, there
are two candidate explanations and the architecture is only one of them.

**Always run the linear architecture with REINFORCE too.** That is what
`--estimator reinforce` is for on a linear config. Three runs, not two:

| run | tells you |
|---|---|
| ATTN-M, closed form | the paper's existing result |
| ATTN-M, REINFORCE | the cost of the estimator, architecture held fixed |
| ATTN-SM, REINFORCE | the cost of the estimator plus the architecture |

The architecture effect is the third minus the second. Reporting the third
minus the first would attribute the estimator's variance to softmax attention.

## Run it

```python
!git clone -q https://github.com/manyagupta13/icl-unlearning /kaggle/working/icl
%cd /kaggle/working/icl
!pip install -q -r requirements.txt

!python tests/verify_softmax.py       # CPU, seconds -- run this first
!python tests/verify_diagnose.py

# ATTN-SM needs its own ensembles; the linear ones do not exist for it
!python scripts/train_ensembles.py --config configs/regression_softmax.yaml
!python scripts/run_auc_sweep.py    --config configs/regression_softmax.yaml

# the three-run comparison
!python scripts/stage2_conditional.py --config configs/regression_softmax.yaml \
    --estimator reinforce --reinforce-steps 1500 --out-suffix _sm
!python scripts/stage2_conditional.py --config configs/regression.yaml \
    --estimator reinforce --reinforce-steps 1500 --out-suffix _reinforce_control
!python scripts/stage2_conditional.py --config configs/regression.yaml \
    --out-suffix _closedform
```

Then the same three for `configs/mnist_softmax.yaml` / `configs/mnist.yaml`.

Training cost should be close to the linear archs — three `(D+1)x(D+1)`
projections plus a read-out vector per shadow, which at `D=4` is 80 parameters.
The softmax over `N+1=32` context positions is the only added work.

## What to check before believing anything

- `lever_max_abs_err_vs_closed_form` in the linear runs' JSON. If that is not
  at floating-point noise, the numeric lever is wrong and every cross-arch
  comparison built on it is void.
- Training actually converged. `check_ensemble_health` now covers ATTN-SM by
  taking the per-shadow norm over all parameters concatenated, but softmax
  attention can also fail by saturating — near-one-hot `alpha` at
  initialisation — which a norm check will not catch. Compare the final query
  MSE against the linear archs on the same config; if it is much worse, the
  ensemble is undertrained and its AUC is not evidence about anything.
- The baseline AUC before any corruption. If ATTN-SM's baseline sits near 0.5,
  there is no membership signal to remove and the whole sweep is measuring
  noise. The linear archs give 0.535 and 0.625 on synthetic; something in that
  neighbourhood is what makes the comparison meaningful.

That last one is the real risk. A negative result on ATTN-SM is only
interesting if the attack worked on it in the first place.
