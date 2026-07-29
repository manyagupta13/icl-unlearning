# Pre-registration: what the three follow-up experiments should show

**Written 2026-07-29, before any of the 16 generated configs were run.**

NOTES.md §4g asks for the strong version of the N/D experiment: *"Predict the
D-dependence analytically, write it down, then run."* This is that document.
Nothing here was fitted to a result. Every number was computed from the config
alone, by `scripts/prereg_table.py`, whose spectral asymptotics are checked
numerically in `scripts/check_spectral_scaling.py`.

The rule this document exists to enforce: **if a prediction below fails, say
so in the write-up.** A pre-registration that gets quietly edited after the
run is worth less than no pre-registration at all.

---

## 0. Notation

From NOTES.md §0 and `theory.py`, with `c = M^T t_q = [c_x ; c_y]` and
`t_q = [x_q ; 0]`:

```
yhat = (1/(N+1)) c^T u ,     u = sum_i t_i y_i
```

Write the predicted matched-context AUC at probe point `p` as

```
AUC_p(s) = Phi( ( mu_1(s) - mu_0(s) ) / sqrt( v_1(s) + v_0(s) ) ),    s = sigma^2
```

and split each hypothesis's variance into the part that exists at `s = 0` and
the part the corruption adds:

```
v_h(s) = A_h  +  B_h * s   (+ higher order in s for C1)
```

- **`A`** — shadow-to-shadow spread of `yhat` at zero corruption.
- **`Delta` = mu_1 - mu_0 at s = 0** — the membership gap.
- **`B`** — the noise-channel coefficient. **This is the part that is
  computable in closed form, and it is where all of the D-dependence lives.**

Define the transition location `sigma^2_* = A / B`: the corruption strength at
which the added variance matches the pre-existing spread, i.e. where the curve
starts moving.

---

## 1. What is held fixed by construction (and why it matters)

Two design choices in `data.py` / `make_configs.py` remove confounds that would
otherwise make any D-sweep uninterpretable. Both are load-bearing for
everything below.

**Trace normalisation.** Every group's spectrum sums to 1, so with
`beta ~ N(0, I_D)` and `x ~ N(0, Lambda)`:

```
E[y^2] = E[beta^T Lambda beta] = tr(Lambda) = 1     for every D
```

Label scale is therefore D-independent. Without this, `sum_f y_i^2` would carry
its own D-dependence and confound the noise channels.

**PR held at a fixed fraction of D.** `FRACS = {z1: 0.30, z2: 0.46, z3: 0.78}`,
`PR = 1 + f(D-1)`. Verified numerically: with `gamma = g/D`,

```
PR/D  ->  (2/g) tanh(g/2)          and       tr(Lambda^-1)  ->  kappa(g) * D^2
                                             kappa(g) = (e^g - 1)^2 e^-g / g^2
```

Both confirmed to 5-6 digits at D = 512 (`check_spectral_scaling.py` section A/B).
So **`tr(Lambda^-1)` grows as `D^2`** along this sweep. For the forget group
(f = 0.78, `g_inf` = 1.89, `kappa` = 1.34) the asymptote is already accurate at
D = 4 — `tr(Lambda^-1)/D^2` runs 1.23, 1.28, 1.31, 1.32 at D = 4, 8, 16, 32. For
z1 (f = 0.30, `g_inf` = 6.65) it is *not* converged at small D, so **use the
exact `tr(Lambda^-1)` from the table, not the `kappa D^2` asymptote**, whenever
a retain group is involved.

---

## 2. The two noise channels, derived

Assume the trained read-out approximates the estimator it is being asked to
learn: `sum_i x_i y_i ~ N Lambda beta`, so recovering `yhat = x_q^T beta`
requires the input block of `M` to act as `M_xx ~ rho ((N+1)/N) Lambda^-1`,
with `rho <= 1` an unknown shrinkage from finite N. Then `c_x = M_xx^T x_q`.

**C2 (input noise, `x_f += eta`).** From `theory.py`:
`Var = s * ||c_x||^2 * sum_{i in f} y_i^2 / (N+1)^2`. Taking expectations with
`x_q ~ N(0, Lambda_f)`:

```
E ||c_x||^2       = rho^2 ((N+1)/N)^2 * E[x_q^T Lambda^-2 x_q]
                  = rho^2 ((N+1)/N)^2 * tr(Lambda_f^-1)
E sum_f y_i^2     = n_f

  =>   B_C2  =  rho^2 * n_f * tr(Lambda_f^-1) / N^2
```

**C1 (label noise, `y_f += eps`).** Leading variance term is
`s * sum_{i in f} (c_x . x_i + 2 c_y y_i)^2 / (N+1)^2`. Now the quadratic form
contracts against `Lambda`, not `Lambda^-1`:

```
E (c_x . x_i)^2   = E[x_q^T M_xx Lambda M_xx^T x_q]
                  = rho^2 ((N+1)/N)^2 * E[x_q^T Lambda^-1 x_q]
                  = rho^2 ((N+1)/N)^2 * D

  =>   B_C1  ~  rho^2 * n_f * D / N^2         (for D >> 1, where the c_y term is subleading)
```

Both quadratic-form identities verified by Monte Carlo at 400k draws,
`check_spectral_scaling.py` section D: `E[x^T Lam^-1 x] = D` to 0.1%, and
`E[x^T Lam^-2 x] = tr(Lam^-1)` to 0.1%, at D = 4…32.

**The ratio is parameter-free.** `rho`, `n_f` and `N` all cancel:

```
sigma^2_*(C1) / sigma^2_*(C2)  =  B_C2 / B_C1  =  tr(Lambda_f^-1) / D  ~  kappa(g) * D
```

This is the single strongest claim in the document, because it survives *any*
assumption about training. It says the label-noise arm and the input-noise arm
sit at systematically different places on the `sigma^2` axis, by a factor that
**grows linearly in D**.

---

## 3. What is NOT predicted

`A` (shadow spread) and `Delta` (membership gap) depend on SGD noise, `lr`,
`steps` and `init_scale`, not just on `(D, N, Lambda)`. I am not going to
pretend to a scaling law for them; the honest move is to **measure them at
`sigma^2 = 0` and predict everything else conditionally**. Concretely:

> **The `sigma^2 = 0` row of each sweep, plus the closed-form `B`, determines
> the entire rest of the curve with zero fitted parameters.**

That is the real content of `theory.py`, and it is what the overlay in
`plot_auc_vs_var.py` already tests. This document's job is to say what that
overlay should look like *as D and N move*, before it is drawn.

---

## 4. Predictions — N/D sweep (`nd_*`)

Computed coefficients (from `prereg_table.py`; `B` relative to `nd_D4_N31`):

| config | D | N | N/D | n_f | PR(z3) | tr(Λ⁻¹) | B_C2 rel | B_C1 rel | σ²*(C1)/σ²*(C2) |
|---|---|---|---|---|---|---|---|---|---|
| nd_D4_N31   |  4 |  31 | 7.75 | 11 |  3.34 |    19.6 |  1.00 | 1.00 |  4.9 |
| nd_D8_N31   |  8 |  31 | 3.88 | 11 |  6.46 |    81.8 |  4.17 | 2.00 | 10.2 |
| nd_D16_N31  | 16 |  31 | 1.94 | 11 | 12.70 |   334.3 | 17.02 | 4.00 | 20.9 |
| nd_D16_N63  | 16 |  63 | 3.94 | 21 | 12.70 |   334.3 |  7.87 | 1.85 | 20.9 |
| nd_D32_N63  | 32 |  63 | 1.97 | 21 | 25.18 |  1352.6 | 31.84 | 3.70 | 42.3 |
| nd_D32_N127 | 32 | 127 | 3.97 | 43 | 25.18 |  1352.6 | 16.04 | 1.86 | 42.3 |

**P1 (parameter-free, the headline).** At every `(D, N)`, C1's transition sits
to the *right* of C2's by exactly `tr(Lambda_f^-1)/D`: a factor of **4.9, 10.2,
20.9, 42.3** at D = 4, 8, 16, 32. Test by fitting a sigmoid in `log sigma^2` to
each arm (NOTES §3 asks for this anyway) and taking the midpoint ratio.
*Falsified if* the measured ratio is flat in D, or off by more than ~2x.

**P2 (D at fixed N).** Along `nd_D4_N31 -> nd_D8_N31 -> nd_D16_N31`, `B_C2`
grows 1 : 4.2 : 17.0 while `B_C1` grows only 1 : 2 : 4. So **the C2 curve moves
left roughly four times faster than the C1 curve** as D grows. If `A` were
D-independent, C2's transition would fall by 17x from D = 4 to D = 16.

**P3 (N at fixed D).** `nd_D16_N31 -> nd_D16_N63` and
`nd_D32_N63 -> nd_D32_N127` both roughly double N. `B ∝ n_f/N^2 ∝ 1/N`, so
**both `B_C1` and `B_C2` should halve** (measured: 17.0 → 7.9 and 31.8 → 16.0,
i.e. 0.46x and 0.50x). Transitions move right by ~2x, at fixed A.

**P4 (matched N/D).** `nd_D8_N31` (N/D = 3.88), `nd_D16_N63` (3.94) and
`nd_D32_N127` (3.97) sit at essentially the same N/D. If ICL difficulty is set
by N/D alone, their `sigma^2 = 0` AUC should agree; their `B_C2` nonetheless
differs by 4x between consecutive pairs. **So matched N/D predicts a matched
intercept but NOT a matched transition location.** That dissociation is the
cleanest thing this sweep can show, and it is the reason the sweep includes
both N/D-matched and D-matched pairs.

**P5 (memory / cost, not physics).** The sweep tensor is `[S, P, N+1, D+1]`
float32. At `nd_D32_N127` that is 512·64·128·33·4 = **554 MB per copy**, ~26x
the D=4 config, with 2-3 copies live inside `sweep_point`. It should fit a T4
but will not fit alongside anything else. If it OOMs, drop `probe.P` to 32
before dropping `n_shadows` — P enters the AUC error as `1/sqrt(P)` only
through correlated probe points, `S` enters directly.

---

## 5. Predictions — PR sweep (`pr_*`), D = 4, N = 31

| config | PR(z3) | tr(Λ⁻¹) | B_C2 rel | σ²*(C2) rel | σ²*(C1)/σ²*(C2) |
|---|---|---|---|---|---|
| pr_1p45 | 1.45 | 237.9 | 1.000 |  1.00 | 59.5 |
| pr_1p90 | 1.90 |  64.7 | 0.272 |  3.68 | 16.2 |
| pr_2p35 | 2.35 |  35.9 | 0.151 |  6.63 |  9.0 |
| pr_2p80 | 2.80 |  25.5 | 0.107 |  9.34 |  6.4 |
| pr_3p25 | 3.25 |  20.4 | 0.086 | 11.69 |  5.1 |
| pr_3p70 | 3.70 |  17.4 | 0.073 | 13.69 |  4.3 |

**P6.** `sigma^2_*(C2)` rises **monotonically and by ~13.7x** across this
sweep, purely from `1/tr(Lambda_f^-1)`. This is a large, clean effect that owes
nothing to training.

**P7 (V-shape).** The retain groups are pinned at PR 1.90 and 2.38. At
`pr_1p90` and `pr_2p35` the forget group is spectrally *almost identical to a
retain group*, so there is little for a membership test to find:
**`|AUC(sigma^2=0) - 0.5|` should be V-shaped in PR(z3), minimised near
PR ~ 2.1, and largest at `pr_1p45` and `pr_3p70`.** It may also change sign
across the minimum. This is the test that separates "membership" from
"difficulty" — NOTES §4f(ii) shows the per-example attack is pure difficulty
confound, and a V-shape here is what says the model-axis AUC is not.

---

## 6. Predictions — rotation control (`rot_*`), the one that decides the headline

**P8 (`rot_*_identity` is a null, and therefore a pipeline test).** In
`rot_mid_identity` / `rot_flat_identity` all three groups have the *same*
spectrum *and* the same (identity) basis — they are literally the same
distribution. `full` trains on {z1,z2,z3}, `oracle` on {z1,z2}, but those are
indistinguishable data streams. So:

> **AUC must sit at chance.**

If it does not, something is coupling the two hypotheses — the first place to
look is `stable_offset(arch, hyp)` in `train_ensembles.py` and the shared
`torch.manual_seed(seed)` inside `train_ensemble`. **Run this pair first and do
not interpret any other config until it passes.** NOTES §6 already puts the
rotation pair first; this is the sharper reason why.

**How to test it — this matters, and the obvious version is wrong.** The first
implementation required every per-row bootstrap CI at `param = 0` to cover 0.5,
and it failed on correct data. Three compounding errors:

- at `param = 0` all six corruption modes are the *identity* edit, so
  `none/C1/C2/C3/flip/whiten` are one measurement recorded six times;
- 9 seed combos x 2 archs x 6 duplicated modes = 108 simultaneous 95%
  intervals, all required to cover — probability `0.95^108 = 0.4%` under a
  perfect null;
- the 9 `(train, probe)` combos reuse 3 trained ensembles, so they are not 9
  independent draws and a `df = 8` test overstates significance.

`scripts/check_null.py` does it properly: deduplicate to `mode = none`, average
over probe seeds *within* each training seed, t-test the resulting cluster
means against 0.5 at `df = n_train_seeds - 1`, and additionally require
`|mean - 0.5| <= 0.02` as a hard bound (with 3 seeds the t-test has almost no
power, so a gross failure could otherwise slip through).

**Measured, 2026-07-29 run:** ATTN-M `0.5026 +0.0026`, ATTN-S `0.5040 +0.0040`;
clustered `t = 2.41` and `2.31` against a `df = 2` critical value of `9.93`.
**P8 PASSES.** Both deviations are positive, which is worth a sentence and more
training seeds eventually, but both sit far inside the 0.01–0.02 noise floor
NOTES §5 measured directly.

**P9 (`rot_mid` / `rot_flat`).** Same spectrum, different random rotations.
`Lambda_train` for full and oracle are both near-isotropic averages, so the
preconditioner mismatch that NOTES §4d identifies as the source of the baseline
AUC largely cancels. **Expect `|AUC(sigma^2=0) - 0.5|` to be substantially
smaller than in the shipped `regression` config.** If instead it survives at
close to full size, then the baseline AUC is *not* preconditioner mismatch, and
NOTES §4d's reading of the completed run is wrong — which is a more interesting
outcome than the expected one, and must be reported as such.

---

## 7. Predictions that apply to every config

**P10 (C2 never crosses chance; C1 does).** NOTES §0 establishes that C2 is
*exactly* zero-mean: `E[d yhat] = 0` at every `sigma^2`. So C2 can only move
the denominator, and `Phi(Delta/sqrt(A + B s))` is monotone toward 0.5 and
**never crosses it**. C1 carries `E[d yhat] = c_y n_f s/(N+1)`, linear in
`sigma^2`, so its numerator moves too: **C1 should cross 0.5 and continue past
it** once `Gamma s > |Delta|`, where `Gamma = |c_y^(1) - c_y^(0)| n_f/(N+1)`.
Note `n_f/(N+1) ~ 1/3` at every config here, so the crossing location is set by
`|Delta|/Gamma` and is not strongly D- or N-dependent.

*This is the qualitative signature that distinguishes the two arms, and it
needs no scaling assumption at all.* If C2 crosses 0.5 anywhere, either the
matched-context pairing has broken or the noise is not being applied where the
algebra assumes.

**P11 (C2 is all masking).** Because C2 removes nothing in expectation, NOTES
§1d's control should show `masking_C2 ~ the entire AUC drop`, i.e.
`auc_shared_residual` for C2 stays flat at its `sigma^2 = 0` value while
`auc_matched_residual` falls. Predicted to hold at **every** D. If it holds at
D = 4 but degrades at D = 32, that is a finding about the control, not about
unlearning.

**P12 (theory overlay).** `auc_theory_residual` should track
`auc_matched_residual` within the seed-spread band for C1 and C2 at every
config. NOTES §4f(iii) reports agreement to 3e-4 at D = 4. **A divergence that
appears only at larger D localises the error**: the derivation assumes nothing
D-specific, so a D-dependent gap points at the pipeline (most likely the
`M_xx ~ Lambda^-1` regime breaking down as N/D → 2, where the read-out must
shrink hard and `rho << 1`).

---

## 8. Scoring this document

After the runs, fill in:

| # | claim | predicted | measured | verdict |
|---|---|---|---|---|
| P1 | σ²*(C1)/σ²*(C2) = tr(Λ⁻¹)/D | 4.9 / 10.2 / 20.9 / 42.3 | | |
| P2 | B_C2 grows ~4x faster than B_C1 in D | 17.0 vs 4.0 at D=16 | | |
| P3 | B halves when N doubles at fixed D | 0.46x, 0.50x | | |
| P4 | matched N/D ⇒ matched intercept, different σ²* | — | | |
| P6 | σ²*(C2) rises 13.7x across the PR sweep | monotone | | |
| P7 | \|AUC−0.5\| V-shaped in PR(z3), min near 2.1 | — | | |
| P8 | rot_*_identity sits at chance everywhere | 0.5 ± CI | | |
| P9 | rot_* baseline AUC much closer to 0.5 | — | | |
| P10 | C2 never crosses 0.5; C1 does | — | | |
| P11 | C2 drop is ~all masking, at every D | — | | |
| P12 | theory overlay holds at every D | — | | |
