"""
Closed-form AUC(sigma^2) for C1 and C2.

The model is linear in the context, so the effect of Gaussian context noise on
the read-out is available in closed form. That makes the headline sweep figure
falsifiable: an independently derived curve laid under the measured points is
a different class of evidence from a sweep alone, and it doubles as an
end-to-end correctness check on the pipeline.

Setup (see NOTES.md section 0). With c = M^T t_q split as c = [c_x ; c_y],

    yhat = (1/(N+1)) c^T u ,   u = sum_i t_i y_i ,   t_i = [x_i ; y_i]

The last coordinate of u is sum_i y_i^2 -- THE LABEL ENTERS QUADRATICALLY.
That is the step that makes C1 and C2 behave oppositely.

C1  y_f -> y_f + eps,  eps ~ N(0, s)      [both factors of t_i y_i move]
    E[dyhat]   = c_y * n_f * s / (N+1)                        <- linear in s, NOT zero
    Var[dyhat] = [ s * sum_{i in f} (c_x.x_i + 2 c_y y_i)^2
                   + 2 c_y^2 n_f s^2 ] / (N+1)^2

C2  x_f -> x_f + eta,  eta ~ N(0, s I)    [query token is outside the forget slice]
    E[dyhat]   = 0
    Var[dyhat] = s * ||c_x||^2 * sum_{i in f} y_i^2 / (N+1)^2

Both were confirmed by Monte Carlo against a from-scratch reimplementation
(tests/verify_algebra.py): C1's shift matches to 4 significant figures and
grows linearly in s; C2's is ~1e-5 at every s tested.

bern  y_f -> (1 - 2B) y_f,  B ~ Bern(t)   [the learned-policy parameterisation]
    Write a_i = y_i * (c_x . x_i) over the forget slice. Then
    E[dyhat]   = -2 t / (N+1) * sum_i a_i
    Var[dyhat] =  4 t (1-t) / (N+1)^2 * sum_i a_i^2

    The step that makes this cleaner than C1: (1 - 2B)^2 = 1 identically, so a
    sign flip leaves the label-label block sum_i y_i^2 of the context vector
    EXACTLY unchanged. C1's epsilon^2 drift has no analogue here, and c_y does
    not enter at all. Confirmed by Monte Carlo at 400k trials per point
    (tests/verify_bern.py): means match to 3-4 significant figures, variances
    likewise, and the label-label block difference is identically 0.

    Note the variance is non-monotone in t: zero at t=0, maximal at t=0.5,
    zero again at t=1 where the flip becomes deterministic. So `bern` at t=1
    coincides with `flip` at t=1, and the two arms differ most at t=0.5.

C3 is deliberately NOT provided. Perturbing x and y simultaneously introduces
an eta*eps cross term in the x-block of u; it is still tractable but was not
derived or verified in NOTES.md, and shipping an unverified formula next to
two verified ones would undercut the point of the overlay.
"""
from __future__ import annotations

import math

import torch

from .data import Probe, assemble


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


@torch.no_grad()
def readout_covector(M: torch.Tensor, probe: Probe):
    """
    c = M^T t_q for every (shadow, probe point), plus the un-corrupted yhat.

    Returns
        c_x   [S, P, D]     input block of c
        c_y   [S, P]        label block of c
        yhat0 [S, P]        prediction on the CLEAN probe
    """
    S = M.shape[0]
    x = probe.x.unsqueeze(0).expand(1, *probe.x.shape)
    y = probe.y.unsqueeze(0).expand(1, *probe.y.shape)
    X, ylab, _ = assemble(x, y)              # [1, P, N+1, D+1]
    X, ylab = X[0], ylab[0]                  # [P, N+1, D+1], [P, N+1]

    tq = X[:, -1, :]                         # [P, D+1]
    u = torch.einsum("pnd,pn->pd", X, ylab)  # [P, D+1]

    # c[s, p, e] = sum_d M[s, d, e] tq[p, d]
    c = torch.einsum("sde,pd->spe", M, tq)   # [S, P, D+1]
    D = probe.x.shape[-1]
    c_x, c_y = c[..., :D], c[..., D]

    N = probe.x.shape[1] - 1
    yhat0 = torch.einsum("spe,pe->sp", c, u) / (N + 1)
    return c_x, c_y, yhat0


@torch.no_grad()
def noise_moments(M: torch.Tensor, probe: Probe, mode: str, sigma2: float):
    """
    Mean shift and variance of yhat induced by the corruption, per (shadow,
    probe point). Returns (shift [S,P], var [S,P]).
    """
    c_x, c_y, _ = readout_covector(M, probe)
    sl = probe.forget_slice
    N = probe.x.shape[1] - 1
    n_f = sl.stop - sl.start
    s = float(sigma2)
    denom = (N + 1) ** 2

    x_f = probe.x[:, sl, :]          # [P, n_f, D]
    y_f = probe.y[:, sl]             # [P, n_f]

    if mode == "C1":
        # c_x . x_i + 2 c_y y_i , per (shadow, probe, forget token)
        lin = torch.einsum("spd,pid->spi", c_x, x_f) + 2.0 * c_y[..., None] * y_f
        shift = c_y * n_f * s / (N + 1)
        var = (s * (lin ** 2).sum(-1) + 2.0 * (c_y ** 2) * n_f * s * s) / denom
    elif mode == "C2":
        shift = torch.zeros_like(c_y)
        var = s * (c_x ** 2).sum(-1) * (y_f ** 2).sum(-1) / denom
    elif mode == "bern":
        # a_i = y_i * (c_x . x_i); the label-label block is untouched because
        # (1 - 2B)^2 = 1, so c_y never enters.
        a = torch.einsum("spd,pid->spi", c_x, x_f) * y_f
        shift = -2.0 * s * a.sum(-1) / (N + 1)
        var = 4.0 * s * (1.0 - s) * (a ** 2).sum(-1) / denom
    else:
        raise ValueError(f"no closed form for mode {mode!r} -- see module docstring")
    return shift, var


@torch.no_grad()
def predicted_auc(M_full: torch.Tensor, M_oracle: torch.Tensor, probe: Probe,
                  mode: str, sigma2: float, observable: str = "residual") -> float:
    """
    Gaussian prediction of the matched-context membership AUC.

    At a fixed probe point the score distribution over the shadow axis has two
    variance sources: shadow-to-shadow variation in M, and the per-shadow
    corruption draw. They add, because each shadow draws its own noise.

        AUC_p = Phi( (mu1 - mu0) / sqrt(v1 + v0) )

    then averaged over probe points, matching audit.membership_auc's
    aggregation. The observable is the SIGN-ALIGNED residual s*(yhat - y),
    s = sign(y_q); the alignment flips the mean and leaves the variance alone.

    Only `observable="residual"` is supported: the loss observable is a
    non-central chi-square whose AUC has no comparably clean closed form.
    """
    if observable != "residual":
        raise ValueError("closed form is derived for the sign-aligned residual only")

    yq = probe.y[:, -1]                                   # [P]
    sgn = torch.sign(yq)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)

    stats = []
    for M in (M_full, M_oracle):
        _, _, yhat0 = readout_covector(M, probe)          # [S, P]
        shift, var = noise_moments(M, probe, mode, sigma2)
        centre = yhat0 + shift                            # [S, P]
        mu = sgn * (centre.mean(dim=0) - yq)              # [P]
        v = centre.var(dim=0, unbiased=True) + var.mean(dim=0)
        stats.append((mu, v))

    (mu1, v1), (mu0, v0) = stats
    z = (mu1 - mu0) / torch.sqrt((v1 + v0).clamp_min(1e-30))
    return float(_normal_cdf(z).mean())
