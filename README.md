# In-context unlearning by prompt corruption

Can a model be made to forget a group of training examples by corrupting that
group inside the prompt, without touching the weights? And if a membership
attack stops working, has the information gone, or is it just hidden?

Mostly hidden. The membership signal sits in the first moment of the residual.
Gaussian noise on the inputs or labels cannot reach it, so it drives the attack
toward chance by inflating variance, and pays two to three orders of magnitude
in preservation error to do so. A label flip moves the mean, reaches chance 6×
to 21× more cheaply, and the optimal flip probability has a closed form.

Full write-up in [`report/paper.pdf`](report/paper.pdf).

## Results

| | synthetic | MNIST (digits 1/3/8) |
|---|---|---|
| baseline AUC | 0.535 / 0.625 | 0.596 / 0.566 |
| cost to reach chance, Gaussian noise | 762–1699× | 4375–7838× |
| cost to reach chance, stochastic flip | 116–234× | 279–337× |
| learned flip, AUC achieved | 0.5074 ± 0.0143 | 0.4987 ± 0.0005 |

Two architectures (ATTN-S, ATTN-M), 512 shadow models per hypothesis, medians
over 3 training seeds × 3 probe seeds. Cost is ε at the closest approach to
chance divided by ε at its own minimum.

## How it works

Ensembles are trained once and reused for the whole sweep. The corruption is a
prompt-time edit, so it never enters training:

```
train ensembles   (expensive, once)     ->  artifacts/ensembles_*.pt
sweep corruption  (forward passes only) ->  artifacts/results_*.csv
optimise policy   (reuses the cache)    ->  artifacts/stage2_*.json
```

A 21-point sweep is 21 forward passes over a frozen probe, not 21 retrainings.
H1 shadows train on all groups, H0 shadows on the retain groups only, and both
are frozen before any corruption is applied.

All groups share one labelling rule (`y = ω·x`, ω drawn per sequence), so they
differ only in the shape of their input covariance. Without this, corrupting
the forget group would delete the answer rather than the membership evidence,
and a drop in AUC would mean nothing.

## Running it

```bash
pip install -r requirements.txt

python scripts/train_ensembles.py  --config configs/regression.yaml   # GPU, minutes
python scripts/run_auc_sweep.py    --config configs/regression.yaml   # seconds
python scripts/plot_auc_vs_var.py  --config configs/regression.yaml
python scripts/plot_tradeoff.py    --config configs/regression.yaml
python scripts/stage2_optimise.py  --config configs/regression.yaml --compare-reinforce
```

Swap in `configs/mnist.yaml` for the MNIST version. The config picks the
experiment; the scripts don't care which. Seed sweeps use
`--train-seed-idx {0,1,2} --probe-seed-idx {0,1,2}`.

## What is measured

**Removal.** An adversary sees the sign-aligned residual `sign(y)(ŷ−y)` at a
frozen probe point and decides whether it came from the full model or the
retrain oracle. Reported as Mann–Whitney AUC. Both hypotheses are scored on the
same corrupted prompt with the same noise draw, so detecting the edit can't be
mistaken for detecting membership.

The score is oriented so the member scores higher, which is the usual
membership-inference convention: a working attack reads AUC > 0.5, chance is
0.5, and unlearning means driving AUC down to 0.5. The full model fits
forget-group queries better, so its residual is smaller and the membership
score is the negated residual — see `audit.membership_score`.

The residual has to be sign-aligned. Ranking raw `ŷ` gives per-point AUCs on
opposite sides of 0.5 for positive and negative query labels, which cancel to
about 0.5 when averaged, however strong the real signal is. See `docs/NOTES.md`
§4b.

**Preservation.** `ε = KL(p_oracle ‖ p_unlearned)` on the residual law, fit as
a univariate Gaussian. AUC depends only on the standardised gap and is
invariant to rescaling both populations; ε is not. That asymmetry is what makes
the two tradeable against each other.

**Masking vs removal.** A shared-noise control broadcasts one corruption draw
across all shadows, holding variance inflation fixed. The gap between that and
the independent-draw run separates hiding the signal from removing it.

## Corruption families

| name | edit | parameter | channel |
|---|---|---|---|
| `C1` | `y_f → y_f + ε`, `ε ~ N(0, σ²)` | σ² | variance |
| `C2` | `x_f → x_f + ε`, `ε ~ N(0, σ²I)` | σ² | variance |
| `C3` | both, independent draws | σ² | variance |
| `flip` | `y_f → (1−2t)·y_f` | t | mean |
| `bern` | `y_f → (1−2B)·y_f`, `B ~ Bern(θ)` | θ | mean + variance |
| `whiten` | `x_f → C_r^½ C_f^-½ x_f`, interpolated | mixing | — |

`flip` has a mean shift and no variance. `bern` has the same mean plus variance
`4θ(1−θ)y²`. Predicted AUC is `Φ((μ₁−μ₀)/√(v₁+v₀))`, so a zero-mean corruption
can only inflate the denominator and can never cross 0.5. `whiten` maps the
forget inputs onto the retain covariance and is included as a control: it tests
whether matching the input distribution is enough on its own. It isn't.

## Layout

```
src/icl_unlearning/
  data.py      MixtureSpec (Gaussian groups), MnistSpec (real images), probes
  models.py    ATTN-S (factored M) / ATTN-M (merged M)
  train.py     ensemble training, S shadows in parallel
  corrupt.py   the six corruption families
  audit.py     membership AUC, bootstrap CI, residual-law fit, alpha/eps, MMD
  sweep.py     sweep loop + shared-noise masking control
  theory.py    closed-form AUC(σ²), exact Bernoulli-flip moments
  policy.py    Stage 2 policies, differentiable AUC, REINFORCE estimator

scripts/       train_ensembles, run_auc_sweep, plot_*, stage2_optimise
               plus controls and diagnostics (oracle_control_attack,
               check_null, check_spectral_scaling, mnist_pr_probe)
configs/       regression.yaml, mnist.yaml, classification.yaml
tests/         correctness checks
report/        paper.tex, paper.pdf, fig/
docs/          working notes, kept dated — see the note at the top of each
```

## Tests

```bash
python tests/verify_theory.py     # closed-form AUC(σ²) vs measurement
python tests/verify_bern.py       # Bernoulli-flip moments, 4e5-trial Monte Carlo
python tests/verify_algebra.py    # read-out algebra
python tests/verify_auc.py        # AUC estimator on a known-answer case
python tests/verify_c3.py
python tests/verify_oracle_attack.py
python tests/test_sanity.py
python tests/verify_batch_preflight.py   # config/compile check, no GPU
```

`verify_bern.py` is the one that matters for Stage 2. It confirms
`E[Δŷ] = −2θ/(N+1)·Σaᵢ` and `Var[Δŷ] = 4θ(1−θ)/(N+1)²·Σaᵢ²` to 3–4 significant
figures. Those moments are exact because `(1−2B)² = 1` leaves the label–label
block of the context vector untouched, which is what makes the Stage 2
objective differentiable without sampling.

## Limitations

Group-level unlearning, not example-level: the adversary detects whether a
distribution was trained on, not whether a specific example was.

Stage 2 doesn't converge on the synthetic task. Cancelling the baseline mean gap
there needs a flip fraction above 1, which a probability can't deliver, so the
optimiser saturates against the bound. This is structural, not a tuning
problem — Result 9 in the paper.

Two datasets, one architecture family. The conditional policy
`p_θ(·|x_f, y_f)` is in `policy.py` but not reported.
