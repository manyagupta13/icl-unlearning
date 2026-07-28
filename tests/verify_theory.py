"""
Check theory.predicted_auc's assembly, in NumPy (no torch here).

The MOMENTS were already verified against Monte Carlo in verify_algebra.py.
What is new in theory.py is turning them into an AUC:

    AUC_p = Phi( (mu1 - mu0) / sqrt(v1 + v0) ),   averaged over p

This script reimplements that assembly from scratch and compares it to a
Monte-Carlo AUC drawn from the same Gaussian model, to confirm the formula
(and the variance-addition step: shadow spread + per-shadow noise) is right.
"""
import numpy as np
from math import erf, sqrt

rng = np.random.default_rng(0)


def Phi(z):
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def auc_per_probe(h1, h0):
    n1, n0 = h1.shape[0], h0.shape[0]
    allv = np.concatenate([h1, h0], axis=0)
    order = np.argsort(allv, axis=0)
    ranks = np.empty_like(allv)
    idx = np.arange(1, n1 + n0 + 1, dtype=allv.dtype)
    np.put_along_axis(ranks, order, np.broadcast_to(idx[:, None], allv.shape), axis=0)
    return (ranks[:n1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


S, P = 4000, 48
print(f"S={S} shadows, P={P} probe points\n")
print(f"{'case':28s} {'predicted':>11s} {'monte carlo':>13s} {'abs err':>9s}")
print("-" * 66)

for case, (d_mu, shadow_sd1, shadow_sd0, noise_sd1, noise_sd0) in {
    "no noise, clear gap":      (0.30, 0.25, 0.25, 0.00, 0.00),
    "noise swamps the gap":     (0.30, 0.25, 0.25, 1.20, 1.20),
    "asymmetric noise":         (0.20, 0.30, 0.18, 0.60, 0.35),
    "tiny gap":                 (0.02, 0.40, 0.40, 0.10, 0.10),
    "negative gap (AUC<0.5)":   (-0.35, 0.25, 0.25, 0.20, 0.20),
}.items():
    # per-probe-point true means (random, so the average over p is non-trivial)
    mu1 = rng.normal(0.0, 0.4, size=P) + d_mu
    mu0 = mu1 - d_mu

    # --- closed form -------------------------------------------------------
    v1 = shadow_sd1 ** 2 + noise_sd1 ** 2
    v0 = shadow_sd0 ** 2 + noise_sd0 ** 2
    pred = Phi((mu1 - mu0) / np.sqrt(v1 + v0)).mean()

    # --- monte carlo: shadow spread AND per-shadow noise, added ------------
    h1 = (mu1[None, :] + shadow_sd1 * rng.normal(size=(S, P))
          + noise_sd1 * rng.normal(size=(S, P)))
    h0 = (mu0[None, :] + shadow_sd0 * rng.normal(size=(S, P))
          + noise_sd0 * rng.normal(size=(S, P)))
    mc = auc_per_probe(h1, h0).mean()

    print(f"{case:28s} {pred:11.4f} {mc:13.4f} {abs(pred-mc):9.4f}")

print("\nAll rows should agree to ~1e-2 (Monte-Carlo error at these S, P).")
print("Note the last row: the formula handles AUC < 0.5 with no special-casing,")
print("which is the regime the real sweep lands in.")
