"""
Audits.

Two families, deliberately kept side by side because they can disagree:

1. Membership inference (per-example, fixed probe).
   H0: retrained oracle (never saw the forget group)
   H1: fully trained, then edited in context
   Success is AUC -> 0.5.

   Computed on TWO observables:
     loss   l = (yhat - y)^2      sign-blind
     output yhat                  sign-aware
   Since l is sigma(r)-measurable with r = yhat - y, and r -> l discards the
   sign, AUC*(l) <= AUC*(r) always. An edit that inverts rather than removes
   sits exactly in the gap.

2. Distributional (population-level).
   alpha = KL(p1 || p) removal, eps = KL(p2 || p) preservation, over the
   forget-population residual law. Plus a nonparametric MMD^2 that transfers
   to settings where the Gaussian fit does not hold.

Conventions
  - AUC is computed per probe point across the shadow axis, then averaged.
  - AUC is returned RAW. AUC < 0.5 means the attacker's statistic is inverted,
    which is not success. Use `symmetrised_auc` if you want the corrected
    quantity.
"""
from __future__ import annotations

import torch


# ------------------------------------------------------------ membership AUC

def _auc_1d(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Mann-Whitney AUC with tie correction. pos, neg are 1-D."""
    n1, n0 = pos.numel(), neg.numel()
    allv = torch.cat([pos, neg])
    order = torch.argsort(allv)
    sv = allv[order]

    ranks = torch.empty_like(sv)
    ranks[order] = torch.arange(1, allv.numel() + 1, device=allv.device,
                                dtype=allv.dtype)
    # average ranks within tie groups
    uniq, inv, cnt = torch.unique(sv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = torch.zeros_like(uniq).scatter_add_(
            0, inv, torch.arange(1, allv.numel() + 1, device=allv.device,
                                 dtype=allv.dtype))
        avg = sums / cnt
        ranks[order] = avg[inv]

    r1 = ranks[:n1].sum()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_per_probe(score_h1: torch.Tensor, score_h0: torch.Tensor) -> torch.Tensor:
    """
    Vectorised Mann-Whitney AUC along the shadow axis, one value per probe point.

        score_* : [S, P]   ->   [P]

    No tie correction: the observables are continuous, so ties occur with
    probability zero. `_auc_1d` keeps the tie-corrected reference path and the
    tests check the two agree.
    """
    n1, n0 = score_h1.shape[0], score_h0.shape[0]
    allv = torch.cat([score_h1, score_h0], dim=0)                 # [n1+n0, P]
    order = torch.argsort(allv, dim=0)
    ranks = torch.empty_like(allv)
    idx = torch.arange(1, n1 + n0 + 1, device=allv.device, dtype=allv.dtype)
    ranks.scatter_(0, order, idx.unsqueeze(1).expand_as(allv).contiguous())
    r1 = ranks[:n1].sum(dim=0)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def membership_auc(score_h1: torch.Tensor, score_h0: torch.Tensor) -> float:
    """
    score_* : [S, P]  (shadow models x probe points)
    Returns the per-probe AUC averaged over probe points.
    """
    return float(auc_per_probe(score_h1, score_h0).mean())


def membership_auc_ci(score_h1: torch.Tensor, score_h0: torch.Tensor,
                      n_boot: int = 200, alpha: float = 0.05,
                      seed: int = 0, resample_probe: bool = True):
    """
    Bootstrap CI for the probe-averaged membership AUC.

    Two sources of noise are resampled:
      - the shadow axis (S draws per hypothesis), independently for H1 and H0
      - the probe axis, if `resample_probe` -- probe points are i.i.d. draws
        from the probe distribution, so resampling them propagates the
        "we only drew one probe" uncertainty into the interval

    Without this the AUC-vs-strength curves are point estimates with S=100
    behind each one, and a 0.03 wiggle is indistinguishable from an effect.

    Returns (auc, lo, hi).
    """
    S1, P = score_h1.shape
    S0 = score_h0.shape[0]
    dev = score_h1.device
    g = torch.Generator(device="cpu").manual_seed(seed)

    point = float(auc_per_probe(score_h1, score_h0).mean())

    i1 = torch.randint(0, S1, (n_boot, S1), generator=g).to(dev)
    i0 = torch.randint(0, S0, (n_boot, S0), generator=g).to(dev)
    ip = (torch.randint(0, P, (n_boot, P), generator=g).to(dev)
          if resample_probe else None)

    vals = []
    for b in range(n_boot):
        h1, h0 = score_h1[i1[b]], score_h0[i0[b]]
        if ip is not None:
            h1, h0 = h1[:, ip[b]], h0[:, ip[b]]
        vals.append(auc_per_probe(h1, h0).mean())
    v = torch.stack(vals)
    lo = float(torch.quantile(v, alpha / 2))
    hi = float(torch.quantile(v, 1 - alpha / 2))
    return point, lo, hi


def symmetrised_auc(a: float) -> float:
    """max(a, 1-a): folds an inverted attacker back onto the informative side."""
    return max(a, 1.0 - a)


def observables(yhat: torch.Tensor, yq: torch.Tensor):
    """-> dict of observable name -> [S, P] score tensor."""
    return {"loss": (yhat - yq) ** 2, "output": yhat}


def spread(score: torch.Tensor) -> float:
    """
    Mean within-ensemble std across probe points. Diagnostic: a stochastic
    corruption can lower AUC purely by inflating this, which is masking, not
    removal. Report it next to every AUC.
    """
    return float(score.std(dim=0).mean())


# ----------------------------------------------------------- distributional

def gaussian_kl(p_mean, p_var, q_mean, q_var, eps: float = 1e-12) -> float:
    """KL(P || Q) for univariate Gaussians."""
    p_var = max(float(p_var), eps)
    q_var = max(float(q_var), eps)
    return float(
        0.5 * (torch.log(torch.tensor(q_var / p_var))
               + (p_var + (p_mean - q_mean) ** 2) / q_var - 1.0)
    )


def fit_residual_law(yhat: torch.Tensor, yq: torch.Tensor):
    """Univariate Gaussian fit of the residual r = yhat - y over the population."""
    r = (yhat - yq).reshape(-1)
    return float(r.mean()), float(r.var(unbiased=True))


def alpha_eps(p1, p2, p):
    """
    p* are (mean, var) tuples of the residual law.
      p1 = fully-trained model, p2 = retrain oracle, p = unlearned model
      alpha = KL(p1||p)  removal      (larger better)
      eps   = KL(p2||p)  preservation (smaller better)
      base  = KL(p1||p2) intrinsic gap
      frontier: eps_min(alpha) = (sqrt(alpha) - sqrt(base))^2 for alpha >= base
    """
    alpha = gaussian_kl(*p1, *p)
    eps = gaussian_kl(*p2, *p)
    base = gaussian_kl(*p1, *p2)
    eps_min = (max(alpha, base) ** 0.5 - base ** 0.5) ** 2
    return {"alpha": alpha, "eps": eps, "base": base,
            "eps_min": eps_min, "gap": eps - eps_min}


def mmd2(x: torch.Tensor, y: torch.Tensor, bandwidth: float | None = None) -> float:
    """
    Unbiased MMD^2 with an RBF kernel and median-heuristic bandwidth.
    x, y are 1-D samples of the observable. Transfers to non-Gaussian settings
    where the closed-form KL is unusable.
    """
    x = x.reshape(-1, 1).double()
    y = y.reshape(-1, 1).double()
    if bandwidth is None:
        z = torch.cat([x, y])
        d = torch.cdist(z, z)
        med = d[d > 0].median()
        bandwidth = float(med) if med > 0 else 1.0

    def k(a, b):
        return torch.exp(-torch.cdist(a, b) ** 2 / (2 * bandwidth ** 2))

    n, m = x.shape[0], y.shape[0]
    kxx = k(x, x).fill_diagonal_(0).sum() / (n * (n - 1))
    kyy = k(y, y).fill_diagonal_(0).sum() / (m * (m - 1))
    kxy = k(x, y).mean()
    return float(kxx + kyy - 2 * kxy)
