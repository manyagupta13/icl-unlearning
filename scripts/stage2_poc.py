#!/usr/bin/env python
"""
Stage 2 proof of concept, pure numpy (torch is unavailable in this sandbox).

Purpose: produce REAL numbers for three claims that the report makes, on a
synthetic instance built to match the shipped config's geometry.

  (1) The closed-form AUC(theta) agrees with the measured AUC.
  (2) The closed-form gradient and the brief's REINFORCE gradient agree in
      expectation -- so the closed form is not a different objective.
  (3) The closed-form gradient has far lower variance and cost, which is the
      argument for using it.

Instance: D=4, N=31, n_f=11, P=64 probe points, S=512 shadows. M_full and
M_oracle are built as preconditioners Lambda_train^-1 for the group sets each
hypothesis trains on, plus per-shadow noise standing in for SGD variation.
"""
import time

import numpy as np

rng = np.random.default_rng(0)
D, N, n_f, P, S = 4, 31, 11, 64, 512
COUNTS = (10, 10, 11)


def spec(D, pr, lo=0.0, hi=1.0):
    def PR(g):
        e = np.exp(-g * np.arange(D)); e /= e.sum()
        return e.sum() ** 2 / (e ** 2).sum()
    while PR(hi) > pr:
        hi *= 2
    for _ in range(200):
        m = (lo + hi) / 2
        if PR(m) > pr: lo = m
        else: hi = m
    e = np.exp(-((lo + hi) / 2) * np.arange(D))
    return e / e.sum()


# ---- instance ---------------------------------------------------------------
L = [np.diag(spec(D, p)) for p in (1.90, 2.38, 3.34)]     # shipped spectra
H = [np.linalg.cholesky(m) for m in L]
Pf = np.linalg.inv(sum(L) / 3)
Po = np.linalg.inv(sum(L[:2]) / 2)


def make_M(base, S, jitter=0.04):
    """[S, D+1, D+1] read-outs: shared preconditioner + per-shadow SGD noise."""
    M = np.zeros((S, D + 1, D + 1))
    M[:, :D, :D] = base
    M += jitter * rng.standard_normal((S, D + 1, D + 1))
    return M


M_full, M_orac = make_M(Pf, S), make_M(Po, S)

# frozen probe
xs = [rng.standard_normal((P, c, D)) @ h.T for c, h in zip(COUNTS, H)]
xq = rng.standard_normal((P, 1, D)) @ H[2].T
x = np.concatenate(xs + [xq], axis=1)                      # [P, N+1, D]
beta = rng.standard_normal((P, D))
y = np.einsum("pnd,pd->pn", x, beta)                       # [P, N+1]
sl = slice(N - n_f, N)
yq = y[:, -1].copy()
sgn = np.where(np.sign(yq) == 0, 1.0, np.sign(yq))


def readout(M):
    """c_x [S,P,D], c_y [S,P], yhat0 [S,P] -- mirrors theory.readout_covector."""
    ylab = y.copy(); ylab[:, -1] = 0.0
    t = np.concatenate([x, ylab[:, :, None]], axis=2)      # [P,N+1,D+1]
    tq = t[:, -1, :]
    u = np.einsum("pnd,pn->pd", t, ylab)
    c = np.einsum("sde,pd->spe", M, tq)
    return c[..., :D], c[..., D], np.einsum("spe,pe->sp", c, u) / (N + 1)


CXF, _, Y0F = readout(M_full)
CXO, _, Y0O = readout(M_orac)
AF = np.einsum("spd,pid->spi", CXF, x[:, sl, :]) * y[:, sl]     # a_i, full
AO = np.einsum("spd,pid->spi", CXO, x[:, sl, :]) * y[:, sl]     # a_i, oracle


def ncdf(z):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))


def auc_closed(theta):
    """Gaussian closed form. theta scalar or [P,n_f]."""
    th = np.broadcast_to(np.asarray(theta, float), (P, n_f))
    out = []
    for Y0, A in ((Y0F, AF), (Y0O, AO)):
        shift = -2.0 * (th * A).sum(-1) / (N + 1)
        var = 4.0 * (th * (1 - th) * A ** 2).sum(-1) / (N + 1) ** 2
        centre = Y0 + shift
        out.append((sgn * (centre.mean(0) - yq),
                    centre.var(0, ddof=1) + var.mean(0)))
    (m1, v1), (m0, v0) = out
    return float(ncdf((m1 - m0) / np.sqrt(np.maximum(v1 + v0, 1e-30))).mean())


def auc_empirical(theta, reps=1):
    """Measured Mann-Whitney AUC over the shadow axis, per probe point."""
    th = np.broadcast_to(np.asarray(theta, float), (P, n_f))
    accs = []
    for _ in range(reps):
        vals = []
        for M, Y0, A in ((M_full, Y0F, AF), (M_orac, Y0O, AO)):
            B = (rng.random((S, P, n_f)) < th).astype(float)
            d = -2.0 * (B * A).sum(-1) / (N + 1)
            vals.append(sgn * (Y0 + d - yq))
        h1, h0 = vals
        a = np.empty(P)
        for p in range(P):
            allv = np.concatenate([h1[:, p], h0[:, p]])
            r = allv.argsort().argsort().astype(float) + 1
            a[p] = (r[:S].sum() - S * (S + 1) / 2) / (S * S)
        accs.append(a.mean())
    return float(np.mean(accs))


print("=" * 78)
print("(1)  closed form vs measurement,  scalar theta")
print("=" * 78)
print(f"{'theta':>7} {'AUC closed':>12} {'AUC measured':>14} {'|diff|':>9}")
for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    c, e = auc_closed(t), auc_empirical(t)
    print(f"{t:7.2f} {c:12.5f} {e:14.5f} {abs(c-e):9.5f}")

# ---- gradients --------------------------------------------------------------
def obj_closed(t):
    return (auc_closed(t) - 0.5) ** 2


def grad_closed(t, h=1e-5):
    return (obj_closed(t + h) - obj_closed(t - h)) / (2 * h)


def grad_reinforce(t, n_samples, reps=1):
    """E[ grad log Q(B) . (AUC(B)-0.5)^2 ] with a mean baseline."""
    g = []
    for _ in range(reps):
        sc, sl_ = [], []
        for _ in range(n_samples):
            B = (rng.random((P, n_f)) < t).astype(float)
            sc.append((auc_empirical(B) - 0.5) ** 2)
            sl_.append((B / max(t, 1e-8) - (1 - B) / max(1 - t, 1e-8)).sum())
        sc = np.array(sc); sl_ = np.array(sl_)
        g.append(float(((sc - sc.mean()) * sl_).mean()))
    return np.array(g)


print("\n" + "=" * 78)
print("(2)+(3)  gradient of (AUC-0.5)^2 at theta = 0.15")
print("=" * 78)
t0 = 0.15
tic = time.time(); gc = grad_closed(t0); t_cf = time.time() - tic
print(f"  closed form      : {gc:+.6e}   ({t_cf*1000:.1f} ms, deterministic)")
for ns in (8, 32, 128):
    tic = time.time(); gr = grad_reinforce(t0, ns, reps=12); dt = (time.time()-tic)/12
    print(f"  REINFORCE n={ns:<4}: {gr.mean():+.6e}  sd {gr.std():.2e}  "
          f"({dt*1000:.0f} ms/grad)  sd/|closed| = {gr.std()/abs(gc):6.2f}")

print("\n" + "=" * 78)
print("(4)  optimisation:  min_theta (AUC(theta) - 0.5)^2")
print("=" * 78)
for label, gfun, nstep in (("closed form", lambda t: grad_closed(t), 60),
                           ("REINFORCE n=32", lambda t: grad_reinforce(t, 32)[0], 60)):
    t = 0.02; lr = 0.02 if label.startswith("closed") else 2e-4
    tic = time.time()
    for _ in range(nstep):
        t = float(np.clip(t - lr * gfun(t), 1e-4, 1 - 1e-4))
    print(f"  {label:15s} -> theta* = {t:.4f}   AUC = {auc_closed(t):.5f}   "
          f"({time.time()-tic:.1f}s)")

print("\n  reference: best theta on a 41-point grid =", end=" ")
grid = [(abs(auc_closed(t) - 0.5), t) for t in np.linspace(0, 1, 41)]
best = min(grid)
print(f"{best[1]:.4f}  (AUC {auc_closed(best[1]):.5f})")
print(f"  baseline at theta=0: AUC = {auc_closed(0.0):.5f}")
