#!/usr/bin/env python
"""
Numerical check of the spectral asymptotics used in the D-dependence
prediction (PREDICTIONS.md sections 2-3).

Pure numpy, mirrors data.py's exp_spectrum / gamma_for_pr exactly
(float64 there too, so agreement should be ~1e-12).

Claims checked
--------------
  (A)  PR(gamma)/D  ->  (2/g) tanh(g/2)   with gamma = g/D
  (B)  tr(Lambda^-1) ->  kappa(g) * D^2,  kappa(g) = (e^g - 1)^2 e^-g / g^2
  (C)  tr(Lambda^-1) at the config's actual PR fractions, D = 4..64
  (D)  E[x^T Lambda^-1 x] = D exactly for x ~ N(0, Lambda)  (the C1 channel)
"""
import numpy as np

# ---------------------------------------------------------------- data.py port

def exp_spectrum(D, gamma):
    k = np.arange(D, dtype=np.float64)
    e = np.exp(-float(gamma) * k)
    return e / e.sum()


def pr_of_gamma(D, gamma):
    e = exp_spectrum(D, gamma)
    return float(e.sum() ** 2 / (e ** 2).sum())


def gamma_for_pr(D, target_pr, tol=1e-12, max_iter=300):
    if not (1.0 < target_pr < D):
        raise ValueError(f"PR must be in (1,{D}), got {target_pr}")
    lo, hi = 0.0, 1.0
    while pr_of_gamma(D, hi) > target_pr:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("no bracket")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if pr_of_gamma(D, mid) > target_pr:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def spectrum_for_pr(D, pr):
    return exp_spectrum(D, gamma_for_pr(D, pr))


# ------------------------------------------------------------- the asymptotics

def pr_frac_asymptote(g):
    """lim_{D->inf} PR/D at gamma = g/D."""
    return (2.0 / g) * np.tanh(g / 2.0) if g > 1e-12 else 1.0


def g_for_frac(f, lo=1e-9, hi=1e3):
    """Invert (2/g)tanh(g/2) = f."""
    if f >= 1.0:
        return 0.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if pr_frac_asymptote(mid) > f:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def kappa(g):
    """lim tr(Lambda^-1)/D^2 at gamma = g/D."""
    if g < 1e-9:
        return 1.0
    return (np.expm1(g) ** 2) * np.exp(-g) / (g ** 2)


# --------------------------------------------------------------------- checks

def main():
    FRACS = {"z1": 0.30, "z2": 0.46, "z3": 0.78}     # from make_configs.py
    rng = np.random.default_rng(0)

    print("=" * 78)
    print("(A)+(B)  asymptotics at fixed g = gamma*D")
    print("=" * 78)
    print(f"{'g':>6} {'PR/D emp (D=512)':>18} {'(2/g)tanh(g/2)':>16} "
          f"{'trLinv/D^2 (D=512)':>20} {'kappa(g)':>12}")
    for g in (0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
        D = 512
        lam = exp_spectrum(D, g / D)
        pr_d = (lam.sum() ** 2 / (lam ** 2).sum()) / D
        tri = (1.0 / lam).sum() / D ** 2
        print(f"{g:6.2f} {pr_d:18.6f} {pr_frac_asymptote(g):16.6f} "
              f"{tri:20.6f} {kappa(g):12.6f}")

    print()
    print("=" * 78)
    print("(C)  tr(Lambda^-1) at the config's PR fractions, PR = 1 + f(D-1)")
    print("=" * 78)
    for name, f in FRACS.items():
        g_inf = g_for_frac(f)
        print(f"\n  group {name}   fraction f={f}   g_inf={g_inf:.4f}   "
              f"kappa={kappa(g_inf):.4f}")
        print(f"    {'D':>4} {'PR':>8} {'gamma':>9} {'gamma*D':>8} "
              f"{'tr(Lam^-1)':>14} {'/D^2':>10} {'ratio vs D=4':>13}")
        base = None
        for D in (4, 8, 16, 32, 64):
            pr = 1.0 + f * (D - 1.0)
            lam = spectrum_for_pr(D, pr)
            gam = gamma_for_pr(D, pr)
            tri = float((1.0 / lam).sum())
            if base is None:
                base = tri
            print(f"    {D:4d} {pr:8.3f} {gam:9.5f} {gam*D:8.4f} "
                  f"{tri:14.4f} {tri/D**2:10.4f} {tri/base:13.2f}")

    print()
    print("=" * 78)
    print("(D)  the two quadratic forms that set the C1 / C2 noise channels")
    print("     x ~ N(0, Lambda):  E[x^T Lam^-1 x] = D      (C1 channel)")
    print("                        E[x^T Lam^-2 x] = tr(Lam^-1)  (C2 channel)")
    print("=" * 78)
    print(f"{'D':>4} {'PR':>8} {'E[xLinv x] emp':>16} {'D':>6} "
          f"{'E[xLinv2 x] emp':>17} {'tr(Lam^-1)':>13}")
    for D in (4, 8, 16, 32):
        pr = 1.0 + FRACS["z3"] * (D - 1.0)
        lam = spectrum_for_pr(D, pr)
        n = 400_000
        x = rng.standard_normal((n, D)) * np.sqrt(lam)
        q1 = float((x ** 2 / lam).sum(1).mean())
        q2 = float((x ** 2 / lam ** 2).sum(1).mean())
        print(f"{D:4d} {pr:8.3f} {q1:16.4f} {D:6d} "
              f"{q2:17.2f} {float((1/lam).sum()):13.2f}")

    print()
    print("=" * 78)
    print("(E)  predicted transition-location scaling, normalised to D=4,N=31")
    print("     C2:  sigma2_*  ~  3 A N / (tr(Lam^-1) (N/(N+1))^-2 ... )  -> ~ 1/N")
    print("     C1:  sigma2_*  ~  3 A N / D                              -> ~ D/N")
    print("     using A ~ (D/N)^2 (ridgeless excess risk); see PREDICTIONS.md 4")
    print("=" * 78)
    print(f"{'D':>4} {'N':>5} {'N/D':>6} {'tr(Lam^-1)':>12} "
          f"{'s2*_C2 (rel)':>13} {'s2*_C1 (rel)':>13} {'ratio C1/C2':>12}")
    ref = None
    for D, N in ((4, 31), (8, 31), (16, 31), (16, 63), (32, 63), (32, 127)):
        pr = 1.0 + FRACS["z3"] * (D - 1.0)
        lam = spectrum_for_pr(D, pr)
        tri = float((1.0 / lam).sum())
        n_f = N - 2 * (N // 3)
        A = (D / N) ** 2                      # shadow-spread proxy
        # sigma2_* = A (N+1)^2 / (channel), channel_C2 = ((N+1)/N)^2 tri n_f
        s2_C2 = A * N ** 2 / (tri * n_f)
        s2_C1 = A * N ** 2 / (D * n_f)
        if ref is None:
            ref = (s2_C2, s2_C1)
        print(f"{D:4d} {N:5d} {N/D:6.2f} {tri:12.2f} "
              f"{s2_C2/ref[0]:13.3f} {s2_C1/ref[1]:13.3f} {s2_C1/s2_C2:12.2f}")


if __name__ == "__main__":
    main()
