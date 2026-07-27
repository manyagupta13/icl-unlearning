# icl-unlearning

Membership-inference AUC sweeps for in-context unlearning in linear-attention
in-context learners.

Sweeps corruption strength against membership AUC across the full cross product:

|                | regression | classification |
|----------------|------------|----------------|
| **ATTN-S**     | ✓          | ✓              |
| **ATTN-M**     | ✓          | ✓              |

with AUC reported on **both** observables (loss `ℓ = (ŷ−y)²` and signed output `ŷ`),
because the two disagree and the disagreement is the point.

---

## The one design decision that matters

**Shadow ensembles are trained once and reused for the entire sweep.**

In-context unlearning is a *prompt-time* edit — no weight update. So the
corruption parameter never enters training. The pipeline is:

```
train ensembles  (expensive, once)   →   ensembles.pt
      ↓
sweep corruption (cheap, forward passes only)   →   results.csv
```

Concretely: `H1` shadows are trained on all groups, `H0` shadows are retrained
on retain groups only. Both are frozen. Each sweep point re-runs the *frozen*
models on a re-corrupted probe. A 40-point sweep costs 40 forward passes over
the probe, not 40 retrainings.

Budget it that way. If you find yourself retraining inside the sweep loop,
something is wrong.

---

## Layout

```
icl-unlearning/
├── configs/
│   ├── regression.yaml         # D, N, group spectra, optim, sweep grids
│   └── classification.yaml
├── src/icl_unlearning/
│   ├── data.py                 # covariance-defined mixture, sequences, frozen probe
│   ├── models.py               # ATTN-S (factored) / ATTN-M (merged)
│   ├── train.py                # ensemble training, S shadows in parallel
│   ├── corrupt.py              # C1 / C2 / flip-strength / whiten-retain
│   ├── audit.py                # membership AUC + distributional (KL, MMD)
│   └── sweep.py                # orchestration
├── scripts/
│   ├── train_ensembles.py      # → artifacts/ensembles_{task}.pt
│   ├── run_auc_sweep.py        # → artifacts/results_{task}.csv
│   └── make_figures.py         # → figures/*.pdf
├── tests/test_sanity.py
└── requirements.txt
```

---

## Usage

```bash
pip install -r requirements.txt

python scripts/train_ensembles.py --config configs/regression.yaml
python scripts/run_auc_sweep.py   --config configs/regression.yaml
python scripts/make_figures.py    --config configs/regression.yaml

python scripts/train_ensembles.py --config configs/classification.yaml
python scripts/run_auc_sweep.py   --config configs/classification.yaml
```

Everything is keyed by `(task, arch, hypothesis)`; artifacts are cached, so
re-running the sweep with a new grid does not retrain.

---

## Implementation notes

### The forward pass is O(N·D), not O(N·D²)

With `M = W_Q W_Kᵀ`, the model is

```
ŷ = (1/(N+1)) · t_qᵀ M (XᵀX) e_{D+1}
```

but `(XᵀX) e_{D+1} = Xᵀy = Σᵢ tᵢ yᵢ =: u` (the query label slot is zero), so

```
ŷ = (1/(N+1)) · t_qᵀ M u
```

Never materialise `XᵀX`. This is implemented in `models.py::context_vector`.
It matters once you scale `N` past a few hundred.

### ATTN-S vs ATTN-M

Both have the same *function class* — the difference is parameterisation, and
therefore the training dynamics:

- **ATTN-M** learns `M` directly. Roughly linear dynamics.
- **ATTN-S** learns `W_Q, W_K` with `M = W_Q W_Kᵀ`. The product parameterisation
  gives multiplicative dynamics and the saddle-to-saddle staircase.

If you want the staircase to be visible, use **plain SGD with small init**.
Adam flattens it. `train.py` exposes `optim: sgd|adam` for exactly this reason.

### Numerical gotcha

ATTN-S with momentum diverges easily: effective LR is `lr/(1−β)`, and the
product parameterisation compounds it. `lr=0.05, momentum=0.9` goes to NaN
within a few hundred steps. Use `lr≈0.005, momentum=0.9`, and keep `grad_clip`
on — it is enabled by default in the config and guards the `W_Q W_Kᵀ` blow-up
specifically.

### AUC conventions

- Reported **raw**, not symmetrised. `AUC → 0.5` is the success target;
  `AUC < 0.5` means the attacker's statistic is *inverted*, which is not
  success — see `audit.py::symmetrised_auc` if you want the corrected version.
- Computed **per probe point** over the shadow axis, then averaged. The probe
  is frozen across shadows so cross-model variance reflects membership only.
- Each shadow draws its **own** corruption noise. This is realistic, but note
  it means stochastic corruptions (C1/C2) inflate within-ensemble spread, which
  lowers AUC *mechanically* without removing anything. `audit.py` also returns
  `spread_h1` so you can separate masking from removal. Watch this on the C1 arm.

---

## What to expect (predictions worth falsifying)

1. **Output-AUC saturates at 1.0** across the whole grid, for every method.
   If so, the output observable carries no gradient information anywhere —
   which is the empirical fact that kills AUC as a training signal.
2. **C1 (label noise) is flat in σ²** on the loss observable. Zero-mean
   perturbation leaves `E[ω'] = ω`; a context-aggregating read-out averages it
   out. This arm is a predicted null.
3. **C2 (input noise) is not flat.** It shifts the in-context covariance,
   `Λ̂_f → Λ̂_f + σ²I`, which interacts with the spectral structure that sets
   learnability.
4. **Flip strength `t` is non-monotone with a dead zone at `t = 0.5`,** where
   `E[ỹ] = 0` and the edit becomes zero-mean. Grid `t` finely near 0.5.
5. **ATTN-S vs ATTN-M should differ in the sharpness of the transition,** not
   its location — same function class, different implicit bias.

---

## Scaling knobs

`D`, `N`, `n_shadows`, and the group spectra are all config-level. On GPU the
whole shadow ensemble trains as one batched tensor `[S, D+1, D+1]`, so
`n_shadows: 100 → 512` costs almost nothing until you are memory-bound on the
sequence tensor `[S, B, N+1, D+1]`. That tensor is the thing that will OOM you;
lower `batch_per_shadow` before lowering `n_shadows`.
