"""
Why is AUC(output) pinned near 0.5 and flat in every panel?

Hypothesis: it is a SIGN-CANCELLATION ARTEFACT of the aggregation, not a
property of the models.

Setup that mirrors the audit. At probe point p the query label y_q(p) is
fixed, and:
  - the full model (H1) predicts close to y_q
  - the retrain oracle (H0) never saw the forget group, so it SHRINKS toward
    zero: yhat0 ~ kappa * y_q with kappa < 1

Then for a probe point with y_q > 0, H1's yhat sits ABOVE H0's -> per-point
AUC > 0.5. For y_q < 0, H1's yhat sits BELOW H0's -> per-point AUC < 0.5.
Averaging raw per-point AUC over probe points with mixed-sign queries cancels
to 0.5 REGARDLESS of how distinguishable the two models are.

This script measures that, and checks three candidate fixes.
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


S, P = 100, 64
KAPPA = 0.55          # oracle shrinkage: how much it underfits the forget group
NOISE = 0.25

print(f"S={S} shadows, P={P} probe points, oracle shrinkage kappa={KAPPA}")
print("(a large, obvious membership signal by construction)\n")

for trial in range(3):
    y_q = rng.normal(size=P)                       # query labels, symmetric about 0
    yhat1 = y_q[None, :] + NOISE * rng.normal(size=(S, P))
    yhat0 = KAPPA * y_q[None, :] + NOISE * rng.normal(size=(S, P))

    # --- what the code currently does: raw AUC on yhat, averaged over probe --
    a_out = auc_per_probe(yhat1, yhat0)
    # --- the loss observable ------------------------------------------------
    l1, l0 = (yhat1 - y_q) ** 2, (yhat0 - y_q) ** 2
    a_loss = auc_per_probe(l1, l0)

    # --- fix 1: residual r = yhat - y as the sign-aware observable ----------
    r1, r0 = yhat1 - y_q, yhat0 - y_q
    a_r = auc_per_probe(r1, r0)

    # --- fix 2: sign-align the observable by the query label ---------------
    sg = np.sign(y_q)[None, :]
    a_align = auc_per_probe(sg * yhat1, sg * yhat0)

    # --- fix 3: symmetrise PER PROBE POINT, then average -------------------
    a_symfirst = np.maximum(a_out, 1 - a_out)

    print(f"trial {trial}")
    print(f"  raw AUC(output), averaged      = {a_out.mean():.4f}   <-- current code")
    print(f"  raw AUC(residual r), averaged  = {a_r.mean():.4f}")
    print(f"  AUC(loss), averaged            = {a_loss.mean():.4f}")
    print(f"  sign-aligned AUC(output)       = {a_align.mean():.4f}   <-- fix")
    print(f"  per-point symmetrised, avgd    = {a_symfirst.mean():.4f}   <-- fix")
    print(f"  symmetrise-AFTER-average (bug) = {max(a_out.mean(), 1-a_out.mean()):.4f}")
    pos, neg = y_q > 0, y_q < 0
    print(f"    breakdown: mean AUC on y_q>0 = {a_out[pos].mean():.4f} "
          f"({pos.sum()} pts) | on y_q<0 = {a_out[neg].mean():.4f} ({neg.sum()} pts)")
    print()
