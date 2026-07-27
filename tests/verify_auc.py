"""
NumPy transcription of audit.auc_per_probe (the new vectorised path) checked
against an independent O(n1*n0) pairwise-comparison AUC. Same scatter/argsort
logic as the torch version, so an error there shows up here.
"""
import numpy as np

rng = np.random.default_rng(1)


def auc_per_probe(h1, h0):
    """Transcription of the torch implementation."""
    n1, n0 = h1.shape[0], h0.shape[0]
    allv = np.concatenate([h1, h0], axis=0)
    order = np.argsort(allv, axis=0)
    ranks = np.empty_like(allv)
    idx = np.arange(1, n1 + n0 + 1, dtype=allv.dtype)
    np.put_along_axis(ranks, order, np.broadcast_to(idx[:, None], allv.shape), axis=0)
    r1 = ranks[:n1].sum(axis=0)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_pairwise(a, b):
    """Definition: P(A > B) + 0.5 P(A == B)."""
    d = a[:, None] - b[None, :]
    return (d > 0).mean() + 0.5 * (d == 0).mean()


ok = True
for trial, (n1, n0, shift) in enumerate(
        [(100, 100, 0.0), (100, 100, 1.0), (40, 120, 0.5),
         (7, 5, -2.0), (200, 50, 0.2)]):
    P = 9
    h1 = rng.normal(size=(n1, P)) + shift
    h0 = rng.normal(size=(n0, P))
    fast = auc_per_probe(h1, h0)
    ref = np.array([auc_pairwise(h1[:, p], h0[:, p]) for p in range(P)])
    err = np.abs(fast - ref).max()
    print(f"trial {trial}: n1={n1:3d} n0={n0:3d} shift={shift:+.1f}  "
          f"maxerr={err:.2e}  mean_auc={fast.mean():.4f}")
    ok &= err < 1e-9

# endpoints
a = np.arange(20.0)[:, None]
print(f"\nseparated  AUC = {auc_per_probe(a + 100, a)[0]:.6f}  (expect 1)")
print(f"inverted   AUC = {auc_per_probe(a, a + 100)[0]:.6f}  (expect 0)")

# bootstrap CI coverage: under H0 (identical distributions) a 95% interval
# should contain 0.5 about 95% of the time
def ci(h1, h0, n_boot=300, rng=None):
    S1, P = h1.shape
    S0 = h0.shape[0]
    vals = []
    for _ in range(n_boot):
        i1 = rng.integers(0, S1, S1)
        i0 = rng.integers(0, S0, S0)
        ip = rng.integers(0, P, P)
        vals.append(auc_per_probe(h1[i1][:, ip], h0[i0][:, ip]).mean())
    v = np.array(vals)
    return np.quantile(v, 0.025), np.quantile(v, 0.975)


cover = 0
TRIALS = 200
for _ in range(TRIALS):
    h1 = rng.normal(size=(100, 64))
    h0 = rng.normal(size=(100, 64))
    lo, hi = ci(h1, h0, n_boot=150, rng=rng)
    cover += (lo <= 0.5 <= hi)
print(f"\nnull coverage of the 95% bootstrap CI: {cover}/{TRIALS} "
      f"= {100*cover/TRIALS:.1f}%  (expect ~95%)")
print("PASS" if ok and 0.88 <= cover / TRIALS <= 1.0 else "CHECK")
