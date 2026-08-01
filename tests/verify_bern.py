#!/usr/bin/env python
"""
Monte-Carlo check of the Bernoulli-flip moments, in the style of
tests/verify_algebra.py (which did the same job for C1 and C2).

Corruption (professor's page 3):  p_theta(ytilde_f | .) ~ Bern(theta),
features unchanged.  ytilde_i = (1 - 2 B_i) y_i,  B_i ~ Bern(theta).

Claim
-----
    a_i      = y_i * (c_x . x_i)                       for i in the forget slice
    E[dyhat] = -2 theta / (N+1) * sum a_i
    Var      =  4 theta (1-theta) / (N+1)^2 * sum a_i^2

The step that makes this cleaner than C1: (1 - 2B)^2 = 1 identically, so the
label-label block sum_i y_i^2 of the context vector is UNCHANGED by a sign
flip. C1's epsilon^2 drift (NOTES.md section 0) has no analogue here.
"""
import numpy as np

rng = np.random.default_rng(0)
D, N, n_f, TRIALS = 4, 31, 11, 400_000

# one fixed probe point and one fixed read-out, as in theory.py
lam = np.array([0.4171, 0.2769, 0.1839, 0.1221]); lam /= lam.sum()
x  = rng.standard_normal((N+1, D)) * np.sqrt(lam)
b  = rng.standard_normal(D)
y  = x @ b
M  = rng.standard_normal((D+1, D+1)) * 0.3
tq = np.concatenate([x[-1], [0.0]])
c  = M.T @ tq
c_x, c_y = c[:D], c[D]
sl = slice(N - n_f, N)

def yhat(ylab):
    """Exact repo forward pass: u = sum_n [x_n ; ylab_n] * ylab_n."""
    t = np.concatenate([x, ylab[:, None]], axis=1)
    u = (t * ylab[:, None]).sum(0)
    return (c @ u) / (N + 1)

ylab0 = y.copy(); ylab0[-1] = 0.0
base  = yhat(ylab0)

a = y[sl] * (x[sl] @ c_x)
print(f"{'theta':>7} {'E[dyhat] emp':>14} {'theory':>12} "
      f"{'Var emp':>12} {'theory':>12}")
for th in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
    d = np.empty(TRIALS)
    for k in range(TRIALS):
        yl = ylab0.copy()
        B  = rng.random(n_f) < th
        yl[sl] = np.where(B, -y[sl], y[sl])
        d[k] = yhat(yl) - base
    m_t = -2*th/(N+1) * a.sum()
    v_t =  4*th*(1-th)/(N+1)**2 * (a**2).sum()
    print(f"{th:7.2f} {d.mean():14.6f} {m_t:12.6f} {d.var():12.3e} {v_t:12.3e}")

print("\nand the claim that the label-label block is untouched:")
for th in (0.25, 0.5, 1.0):
    yl = ylab0.copy(); B = rng.random(n_f) < th
    yl[sl] = np.where(B, -y[sl], y[sl])
    print(f"  theta={th}:  sum ytilde^2 - sum y^2 = "
          f"{(yl**2).sum() - (ylab0**2).sum():+.3e}")
