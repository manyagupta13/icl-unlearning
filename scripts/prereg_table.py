#!/usr/bin/env python
"""
Emit the pre-registration numbers for PREDICTIONS.md: the exactly-computable
noise-channel coefficients for every generated config.

    B_C2 / sigma^2  ~  rho^2 n_f tr(Lam_f^-1) / N^2      (input-noise channel)
    B_C1 / sigma^2  ~  rho^2 n_f D            / N^2      (label-noise channel)
    ratio  B_C2/B_C1 = tr(Lam_f^-1)/D                    <- rho, n_f, N all cancel

rho (the read-out shrinkage) and A (the shadow-spread variance) are NOT
predicted here -- they are measured at sigma^2 = 0. Everything below is
parameter-free given the config.
"""
import numpy as np

def exp_spectrum(D, gamma):
    k = np.arange(D, dtype=np.float64)
    e = np.exp(-float(gamma) * k)
    return e / e.sum()

def pr_of_gamma(D, g):
    e = exp_spectrum(D, g)
    return float(e.sum() ** 2 / (e ** 2).sum())

def gamma_for_pr(D, pr, tol=1e-12):
    lo, hi = 0.0, 1.0
    while pr_of_gamma(D, hi) > pr:
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if pr_of_gamma(D, mid) > pr:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

def spec(D, pr):
    return exp_spectrum(D, gamma_for_pr(D, pr))


FRACS = {"z1": 0.30, "z2": 0.46, "z3": 0.78}

def counts_for(N, k=3):
    per = N // k
    return per, N - per * (k - 1)


print("=" * 92)
print("N/D SWEEP  (nd_*)   forget group z3, PR = 1 + 0.78(D-1)")
print("=" * 92)
print(f"{'config':>14} {'D':>4} {'N':>5} {'N/D':>6} {'n_f':>5} "
      f"{'PR(z3)':>8} {'tr(Lam^-1)':>12} {'B_C2 rel':>10} {'B_C1 rel':>10} "
      f"{'C1/C2 sigma2*':>14}")
ref2 = ref1 = None
nd = [(4, 31), (8, 31), (16, 31), (16, 63), (32, 63), (32, 127)]
for D, N in nd:
    pr = 1.0 + FRACS["z3"] * (D - 1.0)
    lam = spec(D, pr)
    tri = float((1.0 / lam).sum())
    _, n_f = counts_for(N)
    b2 = n_f * tri / N ** 2
    b1 = n_f * D / N ** 2
    if ref2 is None:
        ref2, ref1 = b2, b1
    print(f"{'nd_D%d_N%d' % (D, N):>14} {D:4d} {N:5d} {N/D:6.2f} {n_f:5d} "
          f"{pr:8.3f} {tri:12.2f} {b2/ref2:10.3f} {b1/ref1:10.3f} "
          f"{tri/D:14.2f}")

print()
print("  B_* rel = coefficient relative to nd_D4_N31.  sigma2_* = A / B, so a")
print("  LARGER B means the transition sits at SMALLER sigma^2 (curve moves left).")
print("  'C1/C2 sigma2*' = tr(Lam^-1)/D = how much further right C1's transition")
print("  sits than C2's, at the same (D,N). Parameter-free.")

print()
print("=" * 92)
print("PR SWEEP  (pr_*)   D = 4, N = 31, n_f = 11; retain PRs 1.90 / 2.38")
print("=" * 92)
print(f"{'config':>12} {'PR(z3)':>8} {'gamma':>9} {'tr(Lam^-1)':>12} "
      f"{'B_C2 rel':>10} {'sigma2*_C2 rel':>15} {'C1/C2':>8}")
base = None
for p3 in (1.45, 1.90, 2.35, 2.80, 3.25, 3.70):
    lam = spec(4, p3)
    tri = float((1.0 / lam).sum())
    if base is None:
        base = tri
    print(f"{('pr_%.2f' % p3).replace('.', 'p'):>12} {p3:8.2f} "
          f"{gamma_for_pr(4, p3):9.5f} {tri:12.3f} {tri/base:10.3f} "
          f"{base/tri:15.3f} {tri/4:8.2f}")
print()
print("  Retain groups sit at PR 1.90 and 2.38, so the MEMBERSHIP signal should")
print("  be smallest where PR(z3) is between them (~2.1) and grow either side.")
print("  Independently, the C2 TRANSITION LOCATION should rise monotonically")
print("  with PR(z3) as 1/tr(Lam^-1) -- a ~6.9x span across this sweep.")

print()
print("=" * 92)
print("ROTATION  (rot_*)   D = 4, N = 31, all three groups share one spectrum")
print("=" * 92)
for tag, frac in (("mid", 0.55), ("flat", 0.85)):
    pr = 1.0 + frac * 3.0
    lam = spec(4, pr)
    print(f"  rot_{tag:<5} PR = {pr:.2f} (all groups)   "
          f"tr(Lam^-1) = {float((1/lam).sum()):8.3f}   "
          f"eigs = {[round(v, 4) for v in lam]}")
print()
print("  rot_*_identity: identical spectrum AND identical (identity) basis means")
print("  the three groups are the SAME DISTRIBUTION. 'full' and 'oracle' then")
print("  train on indistinguishable data -> this is a NULL. AUC must sit at")
print("  chance within CI at every sigma^2. If it does not, the pipeline has a")
print("  bug (most likely seed coupling between the full and oracle streams),")
print("  and no other number in the batch can be trusted.")
