# What stands between this repo and a publishable AUC-vs-σ² result

Written against the state of the repo after the classification strip. Ordered
by how likely each item is to be the thing a referee kills you on.

---

## 0. A correction to the predictions in the README

Before any experimental design questions: **README predictions 2 and 3 are
backwards**, and the algebra is short enough to check by hand. This matters
because prediction 2 ("C1 is a predicted null") is currently the most
falsifiable claim in the repo, and it is falsified on paper.

Write the read-out with `c = Mᵀ t_q ∈ R^{D+1}`, split as `c = [c_x ; c_y]`, so

```
ŷ = (1/(N+1)) · cᵀu ,        u = Σᵢ tᵢ yᵢ ,     tᵢ = [xᵢ ; yᵢ]
```

The last coordinate of `u` is `Σᵢ yᵢ²`. **The label enters `u` quadratically.**
That is the step the README misses.

### C1 (label noise) is *not* zero-mean

`y_f → y_f + ε`, `ε ~ N(0, σ²)` — and because the label is also the last
coordinate of the token, both factors move:

```
Δu = Σ_{i∈f} [ xᵢ εᵢ ;  2 yᵢ εᵢ + εᵢ² ]
```

so, conditional on the probe,

```
E[Δŷ]   = (1/(N+1)) · c_y · n_f · σ²                                    ← linear in σ², NOT zero
Var[Δŷ] = (1/(N+1))² · [ σ² Σ_{i∈f} (c_xᵀxᵢ + 2 c_y yᵢ)²  +  2 c_y² n_f σ⁴ ]
```

The `εᵢ²` term contributes a deterministic drift of `n_f σ²` in the
label–label coordinate. C1 induces a **bias in ŷ that grows linearly in σ²**.
The README's "a context-aggregating read-out averages it out" argument only
applies to the `x_i ε_i` block, not to the `Σ y²` block.

### C2 (input noise) *is* zero-mean

`x_f → x_f + η`. The query token is outside the forget slice, so

```
Δu = [ Σ_{i∈f} ηᵢ yᵢ ;  0 ]
E[Δŷ]   = 0
Var[Δŷ] = (1/(N+1))² · σ² · ‖c_x‖² · Σ_{i∈f} yᵢ²
```

Exactly linear in `σ²`, exactly zero-mean.

**So the roles are swapped relative to the README.** C2 is the pure
variance-inflation arm — which makes it the cleanest possible demonstration of
the masking-vs-removal distinction, because *by construction* it removes
nothing in expectation and can only lower AUC by adding spread. C1 is the arm
with an actual systematic effect.

This is worth foregrounding rather than fixing quietly. "The obvious
zero-mean-noise-does-nothing intuition is wrong, and here is which term breaks
it" is a better contribution than the original prediction being confirmed.

### The theory curve you should overlay

Both arms give Gaussian (C2) or near-Gaussian (C1: Gaussian + scaled χ²₁, with
`n_f = 11` terms) marginals, so with matched contexts

```
AUC_output(σ²) ≈ Φ( (μ₁(σ²) − μ₀(σ²)) / sqrt(v₁(σ²) + v₀(σ²)) )
```

with `μ, v` from the expressions above, evaluated with the actual trained
`M_full` and `M_oracle`. **Plot this as a solid line under the empirical
points.** A sweep figure where measurement lands on an independently derived
curve is a different class of result from a sweep figure alone — it turns the
plot from a benchmark into evidence that you understand the mechanism. It also
doubles as an end-to-end correctness check on the pipeline.

Getting the loss observable in closed form is messier (non-central χ²) but
`E[ℓ]` and `Var[ℓ]` are still elementary, and a Gaussian-approximation AUC is
usually accurate enough to be worth drawing.

---

## 1. Blocking issues (fixed in this pass)

### 1a. H0 and H1 saw different contexts — fixed

The original `sweep_point` scored H1 on the corrupted probe and H0 on the
**clean** probe. A distinguisher can then score above 0.5 purely by detecting
that the context was edited, with no membership information involved at all.
Worse, the confound *grows with σ²*, which is exactly the axis you are
plotting. Any AUC-vs-σ² curve from that comparison is uninterpretable.

Now both hypotheses are scored on the same corrupted probe with the same noise
draw (`auc_matched_*`). The legacy quantity is still emitted as `auc_*` and
drawn faintly in the figures — the gap between the two curves is a direct
measurement of how large the confound was, which is itself worth a panel.

### 1b. Training was not reproducible — fixed

```python
seed = tr["seed"] + hash((arch, hyp)) % 10_000    # was
```

`hash()` on strings and tuples is salted by `PYTHONHASHSEED`, randomised per
process. Verified: three separate interpreters gave 6116 / 2233 / 1785 for the
same input. Every training run used different seeds and none of it could be
reproduced. Replaced with a blake2b-derived offset, with a regression test.

### 1c. No error bars — fixed

`membership_auc` returned a point estimate backed by `S = 100` shadows per
hypothesis. The standard error on a single-probe-point AUC at n₁=n₀=100 is
roughly 0.03–0.06; averaging over P=64 probe points shrinks it, but the probe
points share one frozen probe draw so the errors are correlated and the naive
`1/√(64·100)` is far too optimistic.

Added `membership_auc_ci`, a bootstrap over **both** the shadow axis and the
probe axis (probe points are i.i.d. draws, so resampling them is legitimate and
propagates "we only drew one probe" into the interval). The sweep now writes
`auc_matched_*_lo/_hi` and the figures draw them as bands.

**Do not publish an AUC curve without these bands.** With `n_shadows: 100`
you should expect the band to be wide enough that several of the README's
qualitative predictions are not resolvable, which is information you want
before you spend GPU hours, not after.

### 1d. Masking vs removal was diagnosable but not diagnosed — fixed

The README correctly identifies that per-shadow corruption noise inflates
within-ensemble spread and lowers AUC mechanically, and suggests watching
`spread_h1`. But spread is a covariate, not a control: you cannot read a
number off it and subtract.

Added a proper control. `corrupt(..., shared_noise=True)` draws the corruption
once and broadcasts it across the shadow axis (common random numbers), which
holds the variance-inflation channel fixed while applying the identical edit.
Then

```
masking = AUC(shared noise) − AUC(per-shadow noise)
```

is the portion of the AUC drop attributable to spread rather than to removal,
and it is now a column in the results CSV and a dotted line in the figures.

Given §0, the expectation is that **the entire C2 curve is masking** — that
arm has zero mean effect, so anything it does to AUC has to come through
variance. If that is what the control shows, it is the cleanest result in the
paper and it directly indicts AUC as an unlearning metric.

---

## 2. Statistical power — the next thing to fix, and it is cheap

### `n_shadows: 100` is too few

AUC error is the binding constraint on every claim in this repo, and shadows
are the only thing that reduces it. Per the README's own scaling note the
ensemble trains as one batched `[S, D+1, D+1]` tensor, so **512 shadows costs
almost nothing** and halves the interval width. Lower `batch_per_shadow` if the
`[S, B, N+1, D+1]` tensor becomes the constraint. Budget 512 minimum for final
runs; 1024 if the transitions you care about are shallow.

### One probe draw, one training seed

Everything currently reported is conditional on `probe.seed = 777` and
`train.seed = 1`. A referee will ask what happens under a different draw, and
"the bootstrap covers it" is only half true — the bootstrap over probe points
handles probe sampling noise, but nothing handles training-seed variation in
the *ensemble mean*.

Run ≥3 training seeds and ≥3 probe seeds, then report the across-seed spread
alongside the within-run bootstrap. If those two disagree, the within-run
interval is the wrong one.

### Grid resolution

The original C1/C2 grids were 10 points across 4 decades. You cannot locate a
transition, let alone characterise its sharpness, at that spacing. The config
now uses ~21 log-spaced points per decade-range and finer sampling around
`t = 0.5` for the flip arm. The sweep is forward passes only — this is free.

---

## 3. Design gaps that limit what you can claim

### The spectral story rests on a single point

The framing is that spectral geometry (participation ratio) governs
unlearnability. But there is one forget group (`z3`), one `D = 4`, and three
hand-written spectra. That supports an anecdote, not a claim.

To make it a result: parameterise the spectrum by a single continuous knob —
e.g. eigenvalues `∝ exp(−γk)` with `γ` sweeping from flat to sharply
anisotropic — and plot **AUC-vs-σ² transition location (or slope) against PR**.
That converts "these three groups behaved differently" into a curve with a
functional form, which is a publishable claim in a way that three bar heights
are not. Also sweep `forget ∈ {z1, z2, z3}` so the forget group is not
confounded with "the flattest one".

### `D = 4`, `N = 31` invite a scaling question you cannot currently answer

Fine as a working regime, but expect to be asked whether the phenomenon
survives `D = 16, 32` and `N = 128, 512`. The `O(N·D)` forward pass in
`models.py` already makes this affordable; you mainly need the runs. At minimum
show the headline curve at two `(D, N)` settings.

### The ATTN-S vs ATTN-M prediction has no test statistic

README prediction 5 says the architectures "differ in the sharpness of the
transition, not its location". Nothing in the code measures sharpness or
location. Fit a two-parameter sigmoid in `log σ²` to each AUC curve, report
midpoint and slope with bootstrap CIs, and test the two claims separately.
Right now the prediction cannot be confirmed or refuted from the outputs.

### The x-axis is not comparable across corruption families

`σ²` for C1 and `σ²` for C2 are different physical quantities, so panels cannot
be read against each other. Add a mechanism-neutral budget — the induced
residual-variance change, or the induced KL, both available in closed form from
§0 — and produce a **collapse plot** where C1/C2/C3 curves fall on one master
curve when plotted against it. Collapse plots are disproportionately convincing
and you have the analytics to attempt one.

### `alpha_eps` pools two variance sources

`fit_residual_law` flattens `[S, P]` into one vector and fits a single
Gaussian, so shadow-to-shadow variance and probe-to-probe variance are mixed
into one number that feeds `α` and `ε`. For the frontier plot to mean anything
these should be separated — fit per-probe-point and aggregate, or fit per
shadow, but state which. Also the univariate Gaussian fit is an assumption
worth checking with a QQ plot in an appendix, especially given the χ² term C1
introduces.

---

## 4. Concretely, the figure you asked about

For "AUC vs noise variance", the panel that will survive review:

- **x**: `σ²`, log scale. Include `σ² = 0` via `symlog` (already configured).
- **y**: matched-context AUC, raw, `[0.4, 1.0]`.
- One line per observable (loss, output), each with its bootstrap band.
- Dotted line: the shared-noise control. The vertical gap to the solid line is
  labelled *masking*.
- Solid thin line: the closed-form prediction from §0.
- Horizontal reference at 0.5, and a shaded null band showing the CI width you
  would get at AUC = 0.5 with your `S` — so a reader can see immediately which
  deviations are resolvable.
- Both architectures as rows, sharing the y-axis.

If the output-AUC really does pin at 1.0 everywhere (README prediction 1),
say so in the caption and give it its own sentence in the abstract — "the
sign-aware observable is uninformative as a training signal everywhere in the
corruption space" is the sharpest claim available here, and it is the one that
motivates the distributional criterion.

---

## 4b. Post-mortem on the first Kaggle figure

The first real sweep produced a figure that was unreadable. Four separate
causes, two cosmetic and two substantive. All four are now fixed, but the
substantive ones invalidate the first run's numbers — **re-run after pulling.**

### (i) `AUC(output)` was pinned near 0.5 by sign cancellation — a real bug

The flat orange line sitting at ~0.44 in every panel, unmoved by any
corruption, was not a finding. It was structurally incapable of moving.

At probe point `p` the query label `y_q(p)` is fixed, and an under-fitting
oracle shrinks its prediction toward zero. So for `y_q > 0` the oracle sits
*below* the full model (per-point AUC > 0.5) and for `y_q < 0` it sits *above*
(per-point AUC < 0.5). Averaging raw per-point AUC over a probe with mixed-sign
queries cancels the two halves against each other.

Measured on a synthetic case with the oracle crippled to 55% shrinkage — an
enormous membership signal by construction:

| aggregation | AUC |
|---|---|
| raw `ŷ`, averaged over probe (**what the code did**) | 0.50 – 0.55 |
|  ⤷ restricted to `y_q > 0` | 0.79 |
|  ⤷ restricted to `y_q < 0` | 0.23 |
| sign-aligned `sign(y)·(ŷ−y)` (**fix**) | 0.76 – 0.82 |
| `AUC(loss)` for comparison | 0.25 – 0.31 |

The observable is now `sign(y)·(ŷ−y)`, renamed `residual`. This also repairs a
mismatch between `audit.py`'s docstring and its code: the nesting argument
`AUC*(ℓ) ≤ AUC*(r)` is stated for `r = ŷ − y`, but the code was ranking raw
`ŷ`, for which `ℓ` is not even a function of the observable, so the argument
did not apply. `observables_raw` keeps the old pair so the cancellation stays
measurable, and it is plotted in the new diagnostics figure.

**Related bug:** `symmetrised_auc` was being applied to the already-averaged
AUC. `max(mean_p a_p, 1 − mean_p a_p) ≠ mean_p max(a_p, 1 − a_p)`, and only the
latter undoes cancellation. Fixed via `symmetrised_auc_per_probe`. Note the
per-point version is biased *upward* under the null (~0.54 at S=100, P=64), so
`null_auc_level()` is provided to calibrate it — this is why sign-alignment,
which needs no `max`, is the better primary metric.

### (ii) The shared-noise masking control was pure noise — my design error

The jagged dotted line swinging between 0.4 and 0.9 between adjacent grid
points was an artefact of the control itself. With a *single* shared draw,
every shadow sees the identical corruption, so the resulting AUC is a function
of that one realisation and its sampling variance is enormous.

Fixed by averaging over `n_shared_reps: 16` independent shared draws. The
control is only meaningful as an expectation over draws.

### (iii) `symlog` was rendering a negative axis branch — cosmetic

`ax.set_xscale("symlog", linthresh=1e-3)` draws the symmetric negative side
even though every `σ²` is ≥ 0. That is the source of the
`−10⁰10⁻¹10⁻²10⁻³0 10⁻³…` tick pile-up. Now `xlim` is clamped to `[0, max]`
with explicit decade ticks and a `0` label.

### (iv) `ylim = (0.4, 1.02)` squashed all the data — cosmetic

Everything lived in 0.42–0.58, i.e. the bottom 15% of each panel, while 80% of
the figure was empty. Now autoscaled to the data plus the null line. The five
overlapping lines per panel are also split: the main figure shows matched-context
AUC plus the masking control; the confounded clean-context curve and the
un-aligned `ŷ` curve move to `figures/diagnostics_{name}.pdf`.

### What the figure was actually telling you

Strip the artefacts and one real message remains: **every confidence band
covered 0.5 across the entire grid.** At `n_shadows: 100` nothing in that sweep
was statistically resolvable, which is exactly what §2 predicted. Even after
the sign-alignment fix, do not read any structure off a run at S=100. Go to 512
before interpreting anything.

---

## 4c. Two more bugs, surfaced only once `n_shadows` actually went to 512

Both of these were latent at `n_shadows: 100` and only became visible on the
first real run at the corrected default. Both are now fixed. If you re-run
this at even larger `S`, re-read this section — neither fix eliminates a
size-dependence in principle, they just move where it bites.

### (i) Gradient clipping got looser as the ensemble got bigger

`train_ensemble` clipped with
`torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip * S)` — a single
norm computed across **all S shadows stacked together**. With roughly
independent per-shadow gradients the aggregate norm scales like `√S · g`,
while the threshold scales like `grad_clip · S`. The ratio between them grows
with `S`, so the clip becomes strictly *less* able to catch an individual
diverging shadow as the ensemble grows — the safety net gets looser exactly
when a bigger ensemble makes an outlier shadow more likely to occur.

Verified numerically (`tests/verify_bugs.py`): with one outlier shadow whose
raw gradient norm is 50 against a target `grad_clip=5`, the global clip lets
**49.9–50.0** through at every `S` from 10 to 2000 — it essentially never
engages for the outlier, regardless of ensemble size. A first real run at
S=512 hit this: one ATTN-S oracle shadow diverged to loss ≈4400 while the
other 511 trained normally, and the resulting `M_oracle` silently carried a
corrupted row into every downstream AUC/eps/MMD computation.

Fixed with `clip_grad_norm_per_shadow_`: the norm is computed per leading-axis
slice (per shadow) and each shadow is scaled independently, so `max_norm`
means what it says regardless of `S`. Same test shows the outlier held at
exactly `5.0` at every `S`.

Also added `check_ensemble_health`, run automatically after every
`train_ensemble` call. It raises if any shadow's `M` is non-finite, and warns
if a shadow's Frobenius norm exceeds 20x the ensemble median — the observed
divergence was large but *finite* (loss ≈4400, not NaN), so an `isfinite()`
check alone would have passed silently. **If you see this warning after
pulling the fix, do not proceed to the sweep** — investigate before trusting
the ensemble; the fix should prevent it, but not necessarily under every
config.

### (ii) `mmd2` OOM'd because it never chunked or subsampled

`mmd2` builds one `torch.cdist` matrix over the full residual population.
At `S=512, P=64`, that population is `2·S·P = 65536` points, and the matrix is
`65536² × 8 bytes = 32.0 GiB` — an exact match to the CUDA OOM this repo
actually hit (`Tried to allocate 32.00 GiB`). It was invisible at `S=100`
(`12800² × 8 bytes ≈ 1.2 GiB`).

Fixed by subsampling to `max_n=2000` points per side (config: `sweep.mmd_max_n`)
before building the distance matrix, with a fixed seed for reproducibility.
MMD is a population statistic, so subsampling is a legitimate unbiased
estimator, not an approximation of a different quantity — just a noisier one.
Verified with a synthetic 20k-point population (`tests/test_sanity.py` /
`test_mmd2_subsamples_instead_of_oom`): the call no longer scales with input
size and is deterministic given the seed.

If you push `n_shadows` higher than 512, re-check this arithmetic —
`mmd_max_n` bounds the *population* fed to `cdist`, but if you also raise `P`
(the probe size) the memory scales the same way and you may need to lower
`mmd_max_n` further.

---

## 5. What was verified, and how

`torch` could not be installed in the environment these changes were made in,
so the test suite has not been executed — **run `pytest tests/ -q` yourself
before trusting the refactor.** What *was* checked:

- **AST consistency.** All 11 source files compile, and every call site of the
  refactored functions (`make_sequences`, `make_probe`, `train_ensemble`,
  `per_group_mse`, `corrupt`, `sweep_point`, `run_sweep`, and the new audit
  functions) matches its new signature in arity and keyword names. No lingering
  `task` / `classification` references anywhere including the config.

- **The §0 algebra, by Monte Carlo.** A from-scratch NumPy reimplementation of
  the forward pass and the corruptions, at 400k trials per point:

  | σ² | C1 mean shift (emp) | C1 (theory) | C2 mean shift (emp) |
  |----|---------------------|-------------|---------------------|
  | 0.01 | 0.00231 | 0.00228 | 0.00001 |
  | 0.1  | 0.02283 | 0.02281 | −0.00003 |
  | 1.0  | 0.22802 | 0.22808 | 0.00004 |
  | 4.0  | 0.91337 | 0.91230 | −0.00045 |

  C1's shift matches the closed form to 4 significant figures and is linear in
  σ²; C2's is zero throughout. Variances matched theory equally well. The
  correction in §0 is not a conjecture.

- **The vectorised AUC.** Transcribed to NumPy and checked against the
  `P(A>B) + ½P(A=B)` definition across five shape/shift configurations
  (including unequal `n₁ ≠ n₀` and an inverted case): max error 0.0 exactly.

- **The sign-cancellation diagnosis** (`tests/diagnose_flat_output_auc.py`),
  numbers in §4b(i). Reproduced as two unit tests.

- **The clip-scaling and OOM arithmetic** (`tests/verify_bugs.py`), numbers in
  §4c. The OOM figure (32.000 GiB) matches the traceback's 32.00 GiB exactly;
  the outlier-survives-global-clip figure (49.9/50 at every S) and the
  per-shadow-clip figure (exactly 5.0 at every S) are both reproduced as unit
  tests in `test_sanity.py`.

- **The rebuilt figure**, rendered against a synthetic CSV with the real column
  set and grids: axes now read `0 10⁻³ 10⁻² … 10²` with no negative branch and
  no overlapping ticks, and the y-range fits the data.

- **Bootstrap calibration.** Under the null (identical distributions,
  `S = 100`, `P = 64`), the 95% interval covered 0.5 in 199/200 replications.
  That is *conservative* rather than nominal — resampling both axes widens the
  interval somewhat. Erring wide is the right direction here, but note it if a
  referee asks, and consider reporting the shadow-only interval alongside.

  Worth internalising from the same experiment: with two genuinely identical
  distributions at `S = 100`, the measured AUC came out at **0.488**. A 0.01–0.02
  deviation from 0.5 in your sweeps is noise, not a finding.

---

## 6. Quick checklist before a final run

- [x] `n_shadows` = 512 (shipped default — was the source of the two bugs in §4c)
- [x] `pytest tests/ -q` passes on GPU (confirmed: 14/14, then 18/18 after §4c's tests)
- [x] no `check_ensemble_health` warnings in the training log — **check this on
      every run**, not just once; it is not guaranteed eliminated by the fix
- [ ] ≥3 training seeds × ≥3 probe seeds
- [ ] `forget` swept over all three groups
- [ ] continuous PR knob implemented
- [ ] closed-form overlay implemented and matching within CI
- [ ] sigmoid fits for transition midpoint/slope, with CIs
- [ ] collapse plot attempted
- [ ] artifacts regenerated after the seeding fix (anything cached from before
      it is not reproducible and should be deleted)
