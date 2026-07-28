"""
Is C3's eps_1 (on x_f) independent of eps_2 (on y_f), and do they share scale?

Mirrors corrupt.py's exact call pattern:

    s = param ** 0.5
    y[..., sl]    += s * _noise(y_shape, ...)     # eps_2, shape [S,P,n_f]
    x[..., sl, :] += s * _noise(x_shape, ...)     # eps_1, shape [S,P,n_f,D]

Two separate draws from the same advancing generator. This checks (a) that they
are in fact different realisations, (b) that they have equal variance, and
(c) how the *total* injected perturbation compares across C1 / C2 / C3 at the
same nominal param.
"""
import numpy as np

S, P, n_f, D = 8, 64, 11, 4
param = 1.0
s = param ** 0.5

# --- mirror the two sequential draws from one generator --------------------
rng = np.random.default_rng(0)
eps2 = s * rng.normal(size=(S, P, n_f))        # label noise  (drawn first)
eps1 = s * rng.normal(size=(S, P, n_f, D))     # input noise  (drawn second)

print("(a) independence")
# compare eps2 against the D=0 slice of eps1 -- same shape, so if the generator
# had been reset/reused these would be identical
sl = eps1[..., 0]
print(f"    eps2 vs eps1[...,0]: identical? {np.allclose(eps2, sl)}")
print(f"    correlation = {np.corrcoef(eps2.ravel(), sl.ravel())[0,1]:+.5f} "
      f"(expect ~0 for independent draws)")

print("\n(b) equal variance, one shared knob")
print(f"    Var(eps_2 on y) = {eps2.var():.4f}")
print(f"    Var(eps_1 on x) = {eps1.var():.4f}   (param = {param})")
print("    -> both tied to the SAME `param`; C3 sweeps the diagonal of the")
print("       2-D (sigma_1^2, sigma_2^2) space, not the full space")

print("\n(c) total injected perturbation energy per forget token, at same param")
for label, ex, ey in (("C1 (label only)", 0, 1),
                      ("C2 (input only)", D, 0),
                      ("C3 (both)      ", D, 1)):
    print(f"    {label}: {ex} x-components + {ey} y-component "
          f"-> E||delta||^2 = {(ex + ey) * param:.2f} * sigma^2")
print("    -> at equal nominal Var(eps), C2 injects 4x C1's energy and")
print("       C3 injects 5x. The x-axes are NOT directly comparable across")
print("       panels; see NOTES.md section 3 on a mechanism-neutral budget.")
