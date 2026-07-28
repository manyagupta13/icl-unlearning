# icl-unlearning

Membership-inference AUC sweeps for in-context unlearning in linear-attention
in-context learners.

Task is **in-context linear regression** (`y = xᵀβ`), swept over two
parameterisations of the same function class, **ATTN-S** and **ATTN-M**.

AUC is reported on **both** observables (loss `ℓ = (ŷ−y)²` and the sign-aligned
residual `sign(y)·(ŷ−y)`), because the two disagree and the disagreement is the
point.

> The sign-aware observable must be **sign-aligned**. Ranking raw `ŷ` makes the
> probe-averaged AUC cancel to ~0.5 regardless of the true signal — measured at
> 0.79 on positive-label probe points and 0.23 on negative ones, averaging to
> nothing. See `NOTES.md` §4b.
>
> **AUC below 0.5 is not an error.** `membership_auc(H1, H0)` is directional;
> a real, reproducible effect can sit on either side of 0.5 and the first real
> sweep's did. What to check is whether it moves monotonically toward 0.5 as
> corruption grows (success) and whether the confidence band is tight enough
> to be distinguishable from 0.5 in the first place. See `NOTES.md` §4d.

> **Read `NOTES.md` before running anything.** It documents an algebra error in
> the "what to expect" predictions below (C1 and C2 have their roles swapped),
> the closed-form AUC(σ²) curve available in this model, and what is still
> missing statistically. Predictions 2 and 3 in this README are retained
> verbatim as the *original* hypotheses; `NOTES.md` §0 explains why they are
> wrong.

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
│   └── regression.yaml         # D, N, group spectra, optim, sweep grids
├── src/icl_unlearning/
│   ├── data.py                 # covariance-defined mixture, sequences, frozen probe
│   ├── models.py               # ATTN-S (factored) / ATTN-M (merged)
│   ├── train.py                # ensemble training, S shadows in parallel
│   ├── corrupt.py              # C1 / C2 / flip-strength / whiten-retain
│   ├── audit.py                # membership AUC + bootstrap CI + KL/MMD
│   └── sweep.py                # orchestration
├── scripts/
│   ├── train_ensembles.py      # → artifacts/ensembles_{name}_ts{0,1,...}.pt
│   ├── run_auc_sweep.py        # → artifacts/results_{name}.csv (all seed combos)
│   ├── plot_auc_vs_var.py      # → figures/auc_{C1,C2,C3}.pdf, eps_{...}.pdf
│   └── make_figures.py         # → figures/*.pdf (diagnostic grid, single seed)
├── tests/test_sanity.py
├── NOTES.md                    # critique: what is missing for publishable results
└── requirements.txt
```

---

## Usage

```bash
pip install -r requirements.txt

python scripts/train_ensembles.py --config configs/regression.yaml
python scripts/run_auc_sweep.py   --config configs/regression.yaml
python scripts/plot_auc_vs_var.py --config configs/regression.yaml
python scripts/make_figures.py    --config configs/regression.yaml
```

`train_ensembles.py` trains `train.n_train_seeds` independent ensembles
(default 3) and `run_auc_sweep.py` sweeps every one of them against
`probe.n_probe_seeds` probe draws (default 3) — 9 combinations at the
defaults, all cheap since training is ~15s/ensemble at `n_shadows=512` and the
sweep is forward passes only. `plot_auc_vs_var.py` automatically reports the
min–max spread across those 9 runs instead of just the within-run bootstrap
CI. See `NOTES.md` §4e for why both matter.

Everything is keyed by `(name, arch, hypothesis)`; artifacts are cached, so
re-running the sweep with a new grid does not retrain. To run a variant
(different `D`, different forget group, different spectra), copy
`configs/regression.yaml`, change `name:`, and the artifacts will not collide.

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

- **Matched context is the estimand.** `auc_matched_*` scores H1 and H0 on the
  *same* corrupted probe with the same noise draw. The legacy `auc_*` scores H0
  on the clean probe, which lets a distinguisher win by detecting the edit
  rather than the membership — and that confound grows with corruption
  strength. Both are emitted; plot the matched one.
- Reported **raw**, not symmetrised. `AUC → 0.5` is the success target;
  `AUC < 0.5` means the attacker's statistic is *inverted*, which is not
  success — see `audit.py::symmetrised_auc` if you want the corrected version.
- Computed **per probe point** over the shadow axis, then averaged. The probe
  is frozen across shadows so cross-model variance reflects membership only.
- **Bootstrap CIs** over the shadow and probe axes come out as
  `auc_matched_*_lo/_hi`. At `n_shadows: 100` these are wide. Use 512 for
  anything you intend to publish.
- Each shadow draws its **own** corruption noise, which inflates within-ensemble
  spread and lowers AUC *mechanically* without removing anything. The
  `shared_noise=True` arm repeats the sweep with one noise draw broadcast across
  shadows; `masking_* = AUC(shared) − AUC(per-shadow)` is the size of that
  artefact. Per `NOTES.md` §0, expect the C2 arm to be **entirely** masking.

---

## What to expect (predictions worth falsifying)

> Predictions 2 and 3 are **already falsified on paper** — see `NOTES.md` §0.
> The label enters the context vector quadratically, so C1 carries an `O(σ²)`
> mean shift and C2 is the zero-mean arm, exactly opposite to what is claimed
> below. Kept verbatim as the original hypotheses.

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
