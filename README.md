# In-context unlearning by prompt corruption

Can you make a model forget a group of training examples by **corrupting that
group inside the prompt**, without touching the weights — and does a membership
adversary actually stop working, or does the corruption just hide the signal?

**Short answer: it mostly hides it.** The membership signal in this model lives
in the *first moment* of the residual. Variance-based corruptions (Gaussian
noise on inputs or labels) cannot reach it, so they drive AUC toward chance only
by inflating spread, and pay two to three orders of magnitude in preservation
error to do so. A corruption that moves the mean — a label flip — reaches chance
6× to 21× more cheaply, and the optimal flip probability is available in closed
form.

📄 **[`report/paper.pdf`](report/paper.pdf)** — the full write-up: method,
9 numbered results, all figures.

---

## Results at a glance

| | synthetic (Gaussian groups) | MNIST (digits 1/3/8) |
|---|---|---|
| Baseline membership AUC | 0.465 / 0.375 | 0.404 / 0.434 |
| Cost to reach chance, Gaussian noise | 762–1699× | 4375–7838× |
| Cost to reach chance, stochastic flip | **116–234×** | **279–337×** |
| Learned flip, AUC achieved | 0.4926 ± 0.0143 | **0.5013 ± 0.0005** |

Two architectures (ATTN-S, ATTN-M), 512 shadow models per hypothesis, medians
over 3 training seeds × 3 probe seeds. "Cost" is ε at the closest approach to
chance divided by ε at its own minimum.

---

## The one design decision that matters

**Shadow ensembles are trained once and reused for the entire sweep.**

The corruption is a *prompt-time* edit, so it never enters training:

```
train ensembles   (expensive, once)   →  artifacts/ensembles_*.pt
        ↓
sweep corruption  (cheap, forward passes only)  →  artifacts/results_*.csv
        ↓
optimise policy   (seconds, reuses the same cache)  →  artifacts/stage2_*.json
```

A 21-point sweep costs 21 forward passes over a frozen probe, not 21
retrainings. `H1` shadows train on all groups; `H0` shadows retrain on the
retain groups only. Both are frozen before any corruption is applied. If you
find yourself retraining inside the sweep loop, something is wrong.

**The second design decision:** all groups share one labelling rule
(`y = ω·x`, with `ω` drawn per sequence and shared across groups). Groups
differ *only* in the shape of their input covariance. Without this, corrupting
the forget group would simply delete the answer, and a drop in AUC would mean
nothing.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. train the shadow ensembles (GPU, a few minutes)
python scripts/train_ensembles.py  --config configs/regression.yaml

# 2. sweep every corruption arm (no training, seconds)
python scripts/run_auc_sweep.py    --config configs/regression.yaml

# 3. figures
python scripts/plot_auc_vs_var.py  --config configs/regression.yaml
python scripts/plot_tradeoff.py    --config configs/regression.yaml

# 4. Stage 2 -- learn the flip probability, both gradient estimators
python scripts/stage2_optimise.py  --config configs/regression.yaml --compare-reinforce
```

Swap `configs/regression.yaml` for `configs/mnist.yaml` to run the whole thing
on real MNIST features. The config selects the experiment; the scripts are
config-agnostic.

Seed sweeps use `--train-seed-idx {0,1,2} --probe-seed-idx {0,1,2}`.

---

## What is measured

**Removal** — an adversary sees the sign-aligned residual `sign(y)(ŷ−y)` at a
frozen probe point and must decide whether it came from the full model or the
retrain oracle. Reported as Mann–Whitney AUC, so 0.5 means the adversary is
guessing. Both hypotheses are scored on the *same* corrupted prompt with the
same noise draw, so detecting the edit cannot be mistaken for detecting
membership.

> **AUC below 0.5 is not a bug.** `membership_auc(H1, H0)` is directional. The
> full model fits the forget group better, so it ranks *below* the oracle and
> the baseline sits under 0.5. An AUC of 0.375 is a strong signal — inverted,
> it is 0.625. Unlearning means driving AUC *up to* 0.5, not down.

> The observable must be **sign-aligned**. Ranking raw `ŷ` makes the
> probe-averaged AUC cancel to ~0.5 regardless of the true signal (0.79 on
> positive-label probe points, 0.23 on negative ones). See `docs/NOTES.md` §4b.

**Preservation** — `ε = KL(p_oracle ‖ p_unlearned)` on the residual law, fit as
a univariate Gaussian. AUC depends only on the *standardised* gap and is
invariant to rescaling both populations together; ε is not. That asymmetry is
what makes the two tradeable, and is why the cost ratios are not tautological.

**Masking vs removal** — a shared-noise control broadcasts *one* corruption
draw across all shadows, holding variance inflation fixed. The gap between that
and the independent-draw run separates hiding the signal from removing it.

---

## Corruption families

| name | edit | parameter | channel |
|---|---|---|---|
| `C1` | `y_f → y_f + ε`, `ε ~ N(0, σ²)` | σ² | variance |
| `C2` | `x_f → x_f + ε`, `ε ~ N(0, σ²I)` | σ² | variance |
| `C3` | both, independent draws | σ² | variance |
| `flip` | `y_f → (1−2t)·y_f` | t | mean |
| `bern` | `y_f → (1−2B)·y_f`, `B ~ Bern(θ)` | θ | mean + variance |
| `whiten` | `x_f →` covariance-matched to retain groups | mixing | — |

`flip` carries a mean shift with no variance; `bern` has the same mean plus
variance `4θ(1−θ)y²`. Since predicted AUC is `Φ((μ₁−μ₀)/√(v₁+v₀))`, a zero-mean
corruption can only inflate the denominator and can never cross 0.5 — which is
the whole argument, and is visible in the sweeps.

---

## Layout

```
src/icl_unlearning/
  data.py        MixtureSpec (Gaussian groups), MnistSpec (real images),
                 build_spec() factory, sequence and probe construction
  models.py      ATTN-S (factored) / ATTN-M (merged) parameterisations
  train.py       shadow-ensemble training, S shadows in parallel
  corrupt.py     the six corruption families
  audit.py       membership AUC, bootstrap CI, residual-law fit, alpha/eps, MMD
  sweep.py       the sweep loop + shared-noise masking control
  theory.py      closed-form AUC(σ²) and exact Bernoulli-flip moments
  policy.py      Stage 2 policies, differentiable AUC, REINFORCE estimator

scripts/
  train_ensembles.py     step 1 -- the only expensive step
  run_auc_sweep.py       step 2 -- all arms, all seeds -> results CSV
  plot_auc_vs_var.py     the headline AUC vs Var(ε) figure
  plot_tradeoff.py       removal/preservation curves and frontiers
  stage2_optimise.py     learn the flip probability (closed form + REINFORCE)
  stage2_poc.py          numpy-only proof of concept, no GPU needed
  mnist_pr_probe.py      go/no-go check on MNIST digit geometry (CPU, ~30 s)
  oracle_control_attack.py   control: is the attack reading membership?
  check_null.py              null test for the identity-spectrum configs
  check_spectral_scaling.py  numerical check of the spectral asymptotics
  make_configs.py            generate the config-sweep tier
  run_all_experiments.py     batch runner, resumable
  prereg_table.py            pre-registration numbers for PREDICTIONS.md
  make_figures.py            diagnostic figure grid, single seed

configs/     regression.yaml (synthetic), mnist.yaml, classification.yaml
tests/       correctness checks -- see below
report/      paper.tex + paper.pdf + fig/
docs/        NOTES.md (running lab notebook), PREDICTIONS.md (pre-registration),
             MNIST_DESIGN.md (why digits 1/3/8 and what could go wrong)
```

---

## Tests

```bash
python tests/verify_theory.py     # closed-form AUC(σ²) vs measurement
python tests/verify_bern.py       # Bernoulli-flip moments, 4e5-trial Monte Carlo
python tests/verify_algebra.py    # read-out algebra
python tests/verify_auc.py        # AUC estimator against a known-answer case
python tests/verify_c3.py         # C3 independence
python tests/verify_oracle_attack.py
python tests/test_sanity.py
python tests/verify_batch_preflight.py   # config/compile check, no GPU needed
```

`verify_bern.py` is the one that matters for Stage 2: it confirms
`E[Δŷ] = −2θ/(N+1)·Σaᵢ` and `Var[Δŷ] = 4θ(1−θ)/(N+1)²·Σaᵢ²` to 3–4 significant
figures. Those moments are exact — `(1−2B)² = 1`, so the flip leaves the
label–label block of the context vector untouched — which is what makes the
Stage 2 objective differentiable without sampling.

---

## Two bugs found during this work

Both were caught by cross-checking Stage 2 against the Stage 1 sweep, and both
are fixed. Recorded because they changed conclusions:

1. **Stage 2 measured the wrong corruption.** `empirical_auc_and_eps` treated
   the flip *probability* θ as if it were a realised draw, computing
   `y(1−2θ)` — the deterministic `flip` arm, not the stochastic one. Caught
   because its grid search reproduced the `flip` curve exactly (0.4751) instead
   of the `bern` curve (0.4821). All reported Stage 2 numbers are post-fix.

2. **The MNIST sweep crashed on bookkeeping.** `run_auc_sweep.py` hardcoded an
   eigenbasis field when writing result metadata; MNIST specs have no
   eigenbasis, so it raised `KeyError` *after* computing every result.

---

## Known limitations

- **Group-level, not example-level.** The adversary detects whether a
  distribution was trained on, not whether a specific example was.
- **Stage 2 does not converge on the synthetic task.** Cancelling the baseline
  mean gap there requires a flip fraction > 1, which a probability cannot
  deliver, so the optimiser saturates against the boundary. This is a
  structural result rather than a tuning failure — see Result 9 in the paper.
- Two datasets, one architecture family. The conditional policy
  `p_θ(·|x_f, y_f)` is implemented in `policy.py` but not reported.

`docs/NOTES.md` is a running lab notebook and contains superseded reasoning
alongside current; the paper and this README are the current statements.
