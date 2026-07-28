"""
Two checks on oracle_control_attack.py, in NumPy (no torch in this sandbox):

1. The transpose trick: auc_per_probe(loss_f.T, loss_r.T) must return ONE AUC
   per shadow, computed over the example axis. Verified against a direct
   per-shadow pairwise AUC.

2. The confound itself: a model that never saw the forget group still
   separates forget-vs-retain by loss, purely because the forget group has a
   flatter spectrum and is therefore intrinsically harder. This is the thing
   the script is designed to expose.
"""
import numpy as np

rng = np.random.default_rng(0)


def auc_per_probe(h1, h0):
    n1, n0 = h1.shape[0], h0.shape[0]
    allv = np.concatenate([h1, h0], axis=0)
    order = np.argsort(allv, axis=0)
    ranks = np.empty_like(allv)
    idx = np.arange(1, n1 + n0 + 1, dtype=allv.dtype)
    np.put_along_axis(ranks, order, np.broadcast_to(idx[:, None], allv.shape), axis=0)
    return (ranks[:n1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_pairwise(a, b):
    d = a[:, None] - b[None, :]
    return (d > 0).mean() + 0.5 * (d == 0).mean()


# ---- 1. transpose semantics ------------------------------------------------
S, n = 6, 200
loss_f = rng.gamma(2.0, 0.15, size=(S, n))       # [S, n_eval]
loss_r = rng.gamma(2.0, 0.11, size=(S, n))
vec = auc_per_probe(loss_f.T, loss_r.T)          # -> [S]
ref = np.array([auc_pairwise(loss_f[s], loss_r[s]) for s in range(S)])
print("1. transpose gives one AUC per shadow over the example axis")
print(f"   shape {vec.shape} (expect ({S},))   maxerr vs pairwise = "
      f"{np.abs(vec - ref).max():.2e}")

# ---- 2. the confound -------------------------------------------------------
# Trace-normalised spectra from the config. Flatter (higher PR) = harder to
# learn in-context = higher residual error, independent of membership.
eigs = {"z1": [0.70, 0.15, 0.10, 0.05],
        "z2": [0.60, 0.20, 0.10, 0.10],
        "z3": [0.40, 0.30, 0.20, 0.10]}


def pr(e):
    e = np.array(e) / np.sum(e)
    return e.sum() ** 2 / (e ** 2).sum()


print("\n2. group difficulty confound")
for g, e in eigs.items():
    print(f"   {g}: PR = {pr(e):.2f}")

# Model the residual error as increasing with PR. The KEY point: this holds
# for the oracle too, which never trained on z3 -- difficulty is a property of
# the data distribution, not of what was memorised.
base_err = {g: 0.05 * pr(e) for g, e in eigs.items()}
MEMBERSHIP_BONUS = 0.004   # tiny genuine advantage the full model has on z3

print(f"\n   (membership advantage baked in: only {MEMBERSHIP_BONUS} lower error "
      f"on z3 for the full model)")
print(f"\n   {'model':10s} {'saw z3?':10s} {'loss z3':>9s} {'loss retain':>12s} "
      f"{'AUC(z3 vs retain)':>19s}")
for label, saw in (("full", True), ("oracle", False)):
    mu_f = base_err["z3"] - (MEMBERSHIP_BONUS if saw else 0.0)
    mu_r = 0.5 * (base_err["z1"] + base_err["z2"])
    lf = rng.gamma(4.0, mu_f / 4.0, size=(S, 4000))
    lr = rng.gamma(4.0, mu_r / 4.0, size=(S, 4000))
    a = auc_per_probe(lf.T, lr.T).mean()
    print(f"   {label:10s} {str(saw):10s} {lf.mean():9.4f} {lr.mean():12.4f} "
          f"{a:19.4f}")

print("\n   Both rows sit far from 0.5 and close to EACH OTHER. The oracle")
print("   never saw z3, so everything it scores is pure difficulty. That is")
print("   the artefact an uncontrolled forget-vs-retain attack reports.")
