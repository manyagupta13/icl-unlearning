# Porting the audit to MNIST: design note

**Status: not built. This is the design and the go/no-go criterion.** Written
2026-07-29, while the 16-config synthetic batch is still running.

Goal, in the user's words: *an MNIST classifier where we make it forget one
group (one digit) in context, and measure AUC.*

---

## 0. The constraint that decides the design

**Retain-group tokens must stay informative about the forget-group query.**

In the current regression setup, `beta ~ N(0, I_D)` is drawn per task and
**shared across all three groups**. Groups differ only in the input covariance
`Lambda_g`. So on a probe whose query comes from the forget group, the query is
*still partially answerable from retain tokens alone* — corrupting the forget
tokens degrades the estimate rather than deleting the answer. That is what
makes the resulting AUC a membership measurement.

The obvious MNIST design destroys this. Take the standard in-context
classification setup — context of `(image, label)` pairs with the class→label
map permuted per task, query is a digit-3 image, predict its label. The label
of digit 3 is recoverable **only** from digit-3 context tokens. Corrupt those
and both H1 and H0 collapse to chance *together*:

```
AUC -> 0.5   because the answer was deleted, not because membership was removed
```

You would get a clean, monotone, entirely meaningless curve. `NOTES.md` §1d's
shared-noise control would correctly report the whole thing as masking, and
`NOTES.md` §4f(ii)'s warning about difficulty-vs-membership confounds applies
with full force.

**So the per-task rule must be shared across groups, and the digit classes may
supply only the input distributions.**

## 1. The proposed setup

```
beta ~ N(0, I_D)              fresh per task, SHARED across all digit classes
x    ~ MNIST class c, PCA'd into a shared D-dim feature space
y    = sign(x . beta)         binary in-context classification, labels ±1
groups  = digit classes (a subset, e.g. 3 of them)
forget  = one digit
```

Genuine classification; forget group is a real digit; structure is a
one-for-one map onto the existing pipeline.

### What survives unchanged

- **Shadow economics.** A shadow is still one matrix `M [D+1, D+1]`, so the
  whole ensemble is still one batched tensor and still trains in ~15s at
  S=512. This is the single most important property of the repo and it is
  preserved. (Contrast: per-shadow CNNs would be 512 x 2 hypotheses x 2 archs
  x 3 seeds = **6144 network trainings per config**.)
- **`audit.py` entirely.** Membership AUC, matched-context convention,
  bootstrap CIs, the shared-noise masking control, `alpha`/`eps`, MMD.
- **`corrupt.py`** — C1 (label noise) still perturbs the ±1 labels, C2 (input
  noise) still perturbs the feature vectors, `flip` becomes exactly the ICUL
  label flip it is named after, `whiten` still maps forget-class inputs toward
  the retain-class covariance.
- **`theory.py`, and this is the surprising one.** The §0 moments never assumed
  `y` was Gaussian — they are evaluated against the *actual* probe `x_i, y_i`.
  With `y = ±1` the C2 variance term `sigma^2 ||c_x||^2 sum_f y_i^2` has
  `sum_f y_i^2 = n_f` **exactly, with no sampling noise**. The closed-form
  overlay gets *cleaner* than it is in the Gaussian case.
- **`PREDICTIONS.md` P1, P10, P11, P12** carry over essentially verbatim. P1
  (`sigma2*(C1)/sigma2*(C2) = tr(Lambda_f^-1)/D`) is still parameter-free.

### What changes

- **PR stops being a knob.** In the synthetic configs `spectrum_for_pr(D, pr)`
  sets participation ratio to order. MNIST hands you whatever the digits have.
  The `pr_*` sweep becomes observational: you pick forget digits along the
  observed PR range rather than dialling PR. The `nd_*` sweep survives — PCA
  dimension `D` and context length `N` are both still free.
- **`sign(x . beta)` is not linear in `x`.** The *model* is still linear (it
  regresses on ±1 targets under squared loss), but the target-generating
  process now has a threshold. The Bayes-optimal read-out is no longer
  `Lambda^-1`, so the `rho` shrinkage in `PREDICTIONS.md` §2 gets larger and
  more `D`-dependent. **The `B` coefficients and hence P1 are unaffected** —
  they depend on the trained `M`, not on how the targets were generated — but
  the `M_xx ~ rho Lambda^-1` argument used to *motivate* them weakens. Measure
  `||c_x||^2` directly from the trained `M` rather than relying on the ansatz.
- **Trace normalisation.** `data.py` trace-normalises every group so all groups
  carry equal signal energy and `E[y^2] = 1` at every `D`. Applying that to
  MNIST means rescaling each digit class's features. Defensible, but it is a
  modification of the data and must be stated. Without it, `E[y^2]` differs per
  digit and the noise channels are confounded with class energy.

---

## 2. The fork I cannot resolve without data: centering

`data.py`: *"Groups share mean 0 and mixture weight; they differ ONLY in
covariance spectrum. Spectral geometry is therefore the sole axis of
variation."*

MNIST digit classes differ **mostly in their means**. That is largely what
makes a 3 a 3. Two options, both with a real cost:

| | what you keep | what you pay |
|---|---|---|
| **center per class** | theory.py applies unchanged; spectral geometry stays the sole axis | digit identity reduces to covariance *shape*; classes may become nearly indistinguishable, and then there is no membership signal to find |
| **pooled center** | digits stay digits; strong group separation | group means now differ, which is a channel §0's algebra does **not** model — the moments assume corruption is the only thing shifting the mean |

`scripts/mnist_pr_probe.py` computes both and prints the mean-offset magnitude
`||mu_c - mu_pooled||` alongside, so the size of the untreated channel is
visible rather than assumed.

If pooled centering shows a much larger PR span than per-class, that is a
warning, not a green light: it means the separation is mean-driven and the
spectral story is not what is doing the work.

---

## 3. Go/no-go criterion

```bash
python scripts/mnist_pr_probe.py          # CPU, ~30s, no GPU
```

Reports per-digit PR at D = 4, 8, 16, 32 under both centerings, plus
`tr(Lambda^-1)` (which sets the C2 noise channel — `PREDICTIONS.md` §2).

**Proceed if** the PR span across digits is >= ~1.5x, comparable to the
synthetic config's 1.76x at D=4. Then pick three digits spanning the range,
with the flattest as the forget group — the direct analogue of `z3`.

**Stop if** every digit lands at the same PR. The port would then be measuring
class means, not spectral geometry, and the honest framing of any result would
be "we changed the input distribution", which does not extend this repo's
claim.

The probe has been exercised end-to-end on synthetic digit-like data; it has
**not** been run on real MNIST (no dataset access in the environment it was
written in). The verdict is genuinely unknown — I am not predicting which way
it goes.

---

## 4. If it is a go: build order

1. `src/icl_unlearning/mnist_data.py` — `MnistMixtureSpec` exposing the same
   interface `train.py` / `sweep.py` already consume (`.D`, `.N`, `.names`,
   `.pr(g)`, and a sampler replacing `make_sequences`' Gaussian draw with a
   draw from the class's real feature bank). Nothing downstream should need to
   know the difference.
2. `configs/mnist_cls.yaml` — same schema, `data.source: mnist`, digit list,
   PCA `D`, centering choice.
3. Re-run `tests/verify_algebra.py`'s Monte Carlo against ±1 labels. §0's
   moments *should* hold; that is a claim, and it is cheap to check.
4. Only then: the null control. The MNIST analogue of `rot_mid_identity` is
   **three copies of the same digit as the three groups** — full and oracle
   then train on the same distribution and AUC must sit at chance. Same gate,
   same reasoning as `PREDICTIONS.md` P8.
5. Pre-register before running, same as `PREDICTIONS.md`.

---

## 5. Sequencing

The synthetic batch is running now and `PREDICTIONS.md` §8 is unscored. `NOTES.md`
§6 still has five unticked boxes. The MNIST port shares the audit code with the
synthetic pipeline, so **any bug the null control catches there is a bug the
MNIST work would inherit** — which is a concrete reason to let the batch finish
first, not just a tidiness argument.

Run the probe now (it is CPU-only and costs nothing). Build after the batch
reports.
