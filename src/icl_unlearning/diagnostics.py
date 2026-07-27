"""
Signal diagnostics for the membership audit.

The sweep is only interpretable if the audit has signal to begin with. Before
any edit, the fully-trained ensemble and the retrain oracle saw DIFFERENT
training data, so a membership attacker should separate them almost perfectly:
baseline AUC(output) ~ 1.0. If that control sits near 0.5 the whole grid is
noise around a dead baseline, and every curve in it is uninterpretable.

Three things are worth knowing when that happens, and they answer different
questions:

  bootstrap_auc_ci     is the AUC I measured distinguishable from 0.5 at all?
                       Resamples SHADOW MODELS -- they are the inferential
                       unit, not probe points. Probe points are frozen and
                       shared, so resampling them would understate the error.

  gap_to_spread        WHY is the AUC what it is. AUC is a monotone function of
                       the standardised mean gap between the two arms; d ~ 0.1
                       cannot produce AUC 0.9 no matter how many shadows you
                       train. Separates "no effect" from "effect buried in
                       ensemble variance".

  ensemble_separation  is the failure upstream of the observable? Compares the
                       ensembles in WEIGHT space. If the two mean read-out
                       matrices coincide, the two hypotheses converged to the
                       same function and no observable can ever separate them
                       -- that is a data-model problem, not an audit problem.

Together they split the failure into two diagnosable cases:
    ratio < 1  ->  dead baseline, the models are the same
    ratio > 1  ->  weights differ but outputs do not, i.e. the difference lives
                   in directions the probe never excites (undertrained, or too
                   little spectral contrast between groups)
"""
from __future__ import annotations

import torch

from . import audit


# --------------------------------------------------------------- vectorised AUC

def _auc_columns(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """
    Mann-Whitney AUC per probe point, tie-corrected, vectorised over columns.

        pos [S1, P], neg [S0, P]  ->  [P]

    Mathematically identical to `audit._auc_1d` applied column by column; it
    exists because the bootstrap calls this a few hundred times and the Python
    loop over P dominates otherwise. Ties matter here even for continuous
    scores: resampling with replacement duplicates rows by construction.
    """
    n1, n0 = pos.shape[0], neg.shape[0]
    v = torch.cat([pos, neg], dim=0).double()
    n, P = v.shape

    order = v.argsort(dim=0)
    sv = v.gather(0, order)

    idx = torch.arange(n, device=v.device).unsqueeze(1).expand(n, P)
    start = torch.ones_like(sv, dtype=torch.bool)
    start[1:] = sv[1:] != sv[:-1]          # start[i]: i opens a tie run
    end = torch.ones_like(start)
    end[:-1] = start[1:]                   # end[i]:   i closes a tie run

    # first/last index of the tie run each sorted element belongs to.
    # Run boundaries are increasing, so a running max/min recovers them without
    # a scatter-reduce (which is not available on every torch we run under).
    first = torch.cummax(torch.where(start, idx, torch.zeros_like(idx)),
                         dim=0).values
    cand = torch.where(end, idx, torch.full_like(idx, n - 1))
    last = cand.flip(0).cummin(dim=0).values.flip(0)

    avg_rank = (first + last).double() / 2.0 + 1.0     # 1-based, tie-averaged
    ranks = torch.empty_like(sv)
    ranks.scatter_(0, order, avg_rank)

    r1 = ranks[:n1].sum(dim=0)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ------------------------------------------------------------------- bootstrap

@torch.no_grad()
def bootstrap_auc_ci(score_h1: torch.Tensor, score_h0: torch.Tensor,
                     n_boot: int = 300, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """
    Percentile bootstrap CI on the membership AUC.

        score_h1, score_h0 : [S, P]  (shadow models x probe points)

    Shadow models are resampled with replacement, independently in each arm.
    The probe is frozen and shared across shadows -- it is not resampled, so
    the interval covers training randomness only, which is the randomness the
    audit's null is stated over.

    Returns {auc, lo, hi, width, n_boot}. `auc` is the raw point estimate from
    `audit.membership_auc`, not the bootstrap mean.
    """
    S1, S0 = score_h1.shape[0], score_h0.shape[0]
    point = audit.membership_auc(score_h1, score_h0)

    g = torch.Generator().manual_seed(seed)
    reps = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        i1 = torch.randint(0, S1, (S1,), generator=g).to(score_h1.device)
        i0 = torch.randint(0, S0, (S0,), generator=g).to(score_h0.device)
        reps[b] = _auc_columns(score_h1[i1], score_h0[i0]).mean().cpu()

    lo = float(torch.quantile(reps, alpha / 2.0))
    hi = float(torch.quantile(reps, 1.0 - alpha / 2.0))
    return {"auc": point, "lo": lo, "hi": hi, "width": hi - lo, "n_boot": n_boot}


# --------------------------------------------------------------- effect size

def gap_to_spread(score_h1: torch.Tensor, score_h0: torch.Tensor,
                  eps: float = 1e-12) -> float:
    """
    Standardised separation between the two arms, averaged over probe points:

        d_p = |mean_s h1[s,p] - mean_s h0[s,p]| / (0.5 * (std_s h1 + std_s h0))

    This is what actually drives AUC -- for roughly Gaussian arms
    AUC = Phi(d / sqrt(2)), so d ~ 2.7 is needed for AUC ~ 0.97 and d ~ 0.1
    caps you near 0.53. Reporting it next to an AUC says whether a low value
    means "no difference" (small numerator) or "difference drowned in ensemble
    variance" (large denominator, e.g. a stochastic corruption inflating spread
    without removing anything).
    """
    m1, m0 = score_h1.mean(dim=0), score_h0.mean(dim=0)
    s1, s0 = score_h1.std(dim=0), score_h0.std(dim=0)
    denom = (0.5 * (s1 + s0)).clamp_min(eps)
    return float(((m1 - m0).abs() / denom).mean())


# ------------------------------------------------------------- weight space

def ensemble_separation(M_full: torch.Tensor, M_oracle: torch.Tensor) -> dict:
    """
    Weight-space separation of the two ensembles.

        M_full, M_oracle : [S, D+1, D+1]  frozen effective read-out matrices

        between = ||E[M_full] - E[M_oracle]||_F
        within  = sqrt( 0.5 * (E||M_full - E[M_full]||_F^2
                             + E||M_oracle - E[M_oracle]||_F^2) )
        ratio   = between / within

    ratio << 1 means the two hypotheses learned the same function and the
    membership question has no answer -- fix the data model (more dimensions,
    starker spectral contrast) rather than the attacker. ratio >> 1 with a flat
    output-space AUC means the difference sits in directions the probe does not
    excite.
    """
    mu1, mu0 = M_full.mean(dim=0), M_oracle.mean(dim=0)
    between = float(torch.linalg.norm((mu1 - mu0).reshape(-1)))

    def _within(M, mu):
        return float(((M - mu).reshape(M.shape[0], -1) ** 2).sum(dim=1).mean())

    within = float((0.5 * (_within(M_full, mu1) + _within(M_oracle, mu0))) ** 0.5)
    return {"between": between, "within": within,
            "ratio": between / max(within, 1e-12)}
