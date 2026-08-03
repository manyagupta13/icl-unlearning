"""
What did the conditional policy actually learn?

Stage 2's scalar policy has one knob: a single flip probability theta applied to
every forget token. The conditional policy has one knob per token,
theta_i = sigmoid(MLP([x_i ; y_i])). If it beats the scalar, the interesting
question is not "by how much" but "by doing what", and there are only two things
it can be doing.

THE LEVER
---------
Take the Gaussian AUC of theory.py. Under independent per-token flips the
mean of yhat moves by

    E[dyhat] = -2/(N+1) sum_i theta_i a_i ,   a_i = y_i (c_x . x_i)

and a_i is computed separately under each hypothesis, because c_x comes from M.
What the attacker actually sees is the GAP between the two hypotheses, so the
quantity a single token can move is the difference, sign-aligned the same way
audit.py aligns the residual:

    L_i = sgn(y_q) . (-2/(N+1)) . ( E_s[a_i | M_full] - E_s[a_i | M_orac] )

L_i is this token's lever on mu1 - mu0. Spending flip budget on a token with
|L_i| near zero buys nothing: it perturbs both hypotheses equally and the AUC
does not move. This is the quantity the conditional policy can see and the
scalar policy cannot, so it is the one to measure against.

`token_lever` computes L_i from the closed form and needs a linear read-out.
`token_lever_numeric` measures the same quantity by flipping one token at a
time, and works for any architecture including softmax attention. On the linear
archs the two agree to floating-point precision, which is what licenses using
the numeric one where no closed form exists.

TWO MECHANISMS, BOTH FALSIFIABLE
--------------------------------
1. TARGETING. Put budget on high-|L_i| tokens. Measured by

       T = (sum_i theta_i L_i) / (thetabar . sum_i L_i)

   the mean shift achieved, divided by the shift a scalar policy spending the
   SAME average budget thetabar would achieve. T = 1 is no targeting at all;
   T > 1 means each unit of flip probability is buying more removal. This is a
   ratio of achieved-to-uniform, so it is directly the thing that would make a
   conditional policy worth having.

2. POLARISATION. Push theta_i toward 0 or 1 rather than sitting at intermediate
   values. The flip variance is 4 theta_i (1 - theta_i) a_i^2, maximal at
   theta = 1/2 and zero at both ends, so a polarised policy buys the same mean
   shift with less variance. Variance inflation is the masking channel -- it
   moves AUC toward chance without removing anything (Stage 1, the shared-noise
   control), so a policy that reaches chance with LESS variance is doing more
   genuine removal. Measured by

       R = mean_i[theta_i (1 - theta_i)] / (thetabar (1 - thetabar))

   which is <= 1 by Jensen, equals 1 exactly when theta is uniform, and falls
   toward 0 as the policy polarises.

Both metrics are 1 under a scalar policy by construction. That is deliberate:
it makes the scalar the null, and any departure from 1 is the conditional
policy doing something a scalar could not.

A WARNING ON READING THESE
--------------------------
T > 1 with the AUC unchanged means the policy found the levers but had no room
left to use them. T ~ 1 with the AUC improved means the gain came from somewhere
other than targeting, and the explanation is wrong. Both are informative; only
T > 1 AND a lower eps at matched AUC supports the targeting story.
"""
from __future__ import annotations

import torch

from .data import Probe, assemble
from .models import apply_frozen
from .theory import readout_covector


def _rank(v: torch.Tensor) -> torch.Tensor:
    """Ascending ranks along the last axis. Ties get arbitrary distinct ranks;
    with continuous a_i exact ties are measure-zero and not worth the code."""
    idx = v.argsort(dim=-1)
    out = torch.empty_like(v)
    ar = torch.arange(v.shape[-1], device=v.device, dtype=v.dtype)
    out.scatter_(-1, idx, ar.expand_as(v).contiguous())
    return out


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along the last axis -> [...]"""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = (a.pow(2).sum(-1) * b.pow(2).sum(-1)).clamp_min(1e-30).sqrt()
    return num / den


@torch.no_grad()
def token_lever(M_full: torch.Tensor, M_oracle: torch.Tensor,
                probe: Probe) -> torch.Tensor:
    """
    L_i, the per-token lever on the hypothesis gap. Returns [P, n_f].

    See the module docstring. This is the difference of the per-hypothesis
    influences a_i = y_i (c_x . x_i), averaged over shadows, sign-aligned by
    sgn(y_q) and scaled by -2/(N+1) so that

        d(mu1 - mu0) / d theta_i  =  L_i

    exactly, with mu1, mu0 as defined in theory.predicted_auc.
    """
    sl = probe.forget_slice
    N = probe.x.shape[1] - 1
    x_f, y_f = probe.x[:, sl, :], probe.y[:, sl]

    yq = probe.y[:, -1]
    sgn = torch.sign(yq)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)

    bar = []
    for M in (M_full, M_oracle):
        c_x, _, _ = readout_covector(M, probe)                 # [S, P, D]
        a = torch.einsum("spd,pid->spi", c_x, x_f) * y_f       # [S, P, n_f]
        bar.append(a.mean(dim=0))                              # [P, n_f]

    return sgn[:, None] * (-2.0 / (N + 1)) * (bar[0] - bar[1])


@torch.no_grad()
def token_lever_numeric(M_full, M_oracle, probe: Probe,
                        chunk: int = 4) -> torch.Tensor:
    """
    The same L_i, measured instead of derived. Works for ANY architecture.
    Returns [P, n_f].

    Flip token i's label deterministically, on its own, and record how far the
    prediction moves under each hypothesis:

        L_i = sgn(y_q) . ( E_s[yhat_full(flip i)] - E_s[yhat_full]
                          - E_s[yhat_orac(flip i)] + E_s[yhat_orac] )

    For the linear archs this is EXACTLY the analytic lever, not an
    approximation: flipping y_i changes the context vector by [-2 x_i y_i ; 0]
    -- the label-label block is untouched because y_i^2 is even -- so the
    prediction moves by -2 a_i / (N+1), which is the analytic expression term
    for term. tests/verify_softmax.py checks the two agree to floating-point
    precision, which is what licenses using this version on the softmax model.

    The honest caveat: for a nonlinear architecture this is the derivative of
    the mean shift with respect to theta_i AT theta = 0, because a second
    simultaneous flip no longer adds independently. It stays the right notion
    of "how much can this token move the gap", but it stops being exact away
    from small budgets, and a targeting statistic built on it is a local one.
    For the linear archs no such caveat applies.

    Cost is n_f + 1 forward passes per hypothesis. `chunk` bounds how many
    single-token variants are batched at once, since each one materialises a
    full [S, P, N+1, D+1] context.
    """
    sl = probe.forget_slice
    n_f = sl.stop - sl.start
    S = M_full.shape[0]

    yq = probe.y[:, -1]
    sgn = torch.sign(yq)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)

    def mean_pred(M, y):
        X, ylab, _ = assemble(probe.x.unsqueeze(0).expand(S, *probe.x.shape), y)
        return apply_frozen(M, X, ylab, probe.x.shape[1] - 1).mean(dim=0)

    y0 = probe.y.unsqueeze(0).expand(S, *probe.y.shape)
    base = [mean_pred(M, y0) for M in (M_full, M_oracle)]        # each [P]

    cols = []
    for start in range(0, n_f, chunk):
        stop = min(start + chunk, n_f)
        for j in range(start, stop):
            y = y0.clone()
            y[:, :, sl.start + j] *= -1.0
            d_full = mean_pred(M_full, y) - base[0]
            d_orac = mean_pred(M_oracle, y) - base[1]
            cols.append(sgn * (d_full - d_orac))

    return torch.stack(cols, dim=-1)                              # [P, n_f]


@torch.no_grad()
def describe_policy(theta: torch.Tensor, lever: torch.Tensor,
                    n_buckets: int = 5) -> dict:
    """
    Targeting and polarisation statistics for a realised theta.

        theta  [P, n_f]   flip probabilities the policy emitted
        lever  [P, n_f]   from token_lever

    All statistics are computed within each probe point and then averaged over
    probe points, matching how audit.py aggregates AUC. Averaging the ratios
    rather than ratioing the averages is the conservative choice: it stops one
    probe point with a near-zero denominator from dominating.
    """
    theta = theta.detach().to(lever.dtype)
    if theta.shape != lever.shape:
        theta = theta.expand_as(lever)

    tbar = theta.mean(-1)                                       # [P]
    denom = tbar * lever.sum(-1)                                # [P]

    # Probe points where the uniform-policy shift is ~0 have no meaningful
    # "what would a scalar have got" baseline; drop them rather than report a
    # ratio against noise.
    scale = lever.abs().sum(-1).clamp_min(1e-30)
    ok = denom.abs() > 1e-6 * scale
    targeting = torch.where(ok, (theta * lever).sum(-1) / denom.masked_fill(~ok, 1.0),
                            torch.full_like(denom, float("nan")))

    # Polarisation R = mean_i theta_i(1-theta_i) / [thetabar(1-thetabar)], the
    # flip variance relative to a uniform policy at the same budget. It is only
    # defined when there IS a variance channel to normalise against: as
    # thetabar -> 0 or 1 the whole forget set becomes (near-)deterministic, the
    # denominator collapses, and the ratio blows up to meaningless magnitudes
    # (a 1e21 was observed in a REINFORCE run that drove thetabar to ~1). That
    # is not a polarised policy -- it is a policy with no budget left to
    # polarise -- so those probe points are dropped, exactly as `targeting`
    # drops points with no shift baseline. The threshold 1e-4 corresponds to
    # thetabar within ~1e-4 of 0 or 1; well clear of any genuine mid-range
    # policy, whose thetabar(1-thetabar) is order 0.01-0.25.
    pol_prod = tbar * (1.0 - tbar)                              # [P], <= 1/4
    pol_ok = pol_prod > 1e-4
    pol_num = (theta * (1.0 - theta)).mean(-1)
    polarisation = torch.where(
        pol_ok, pol_num / pol_prod.masked_fill(~pol_ok, 1.0),
        torch.full_like(pol_prod, float("nan")))

    absL = lever.abs()
    rho_signed = _corr(_rank(theta), _rank(lever))
    rho_abs = _corr(_rank(theta), _rank(absL))

    # Mean theta by |lever| bucket, lowest to highest. The scalar policy gives a
    # flat row here; a targeting policy gives an increasing one.
    n_f = theta.shape[-1]
    k = min(n_buckets, n_f)
    order = absL.argsort(dim=-1)
    th_sorted = theta.gather(-1, order)
    edges = [round(j * n_f / k) for j in range(k + 1)]
    buckets = [float(th_sorted[..., edges[j]:edges[j + 1]].mean())
               for j in range(k)]

    frac = float(ok.float().mean())
    return {
        "theta_mean": float(theta.mean()),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "theta_std_within_probe": float(theta.std(dim=-1).mean()),
        "targeting_T": float(torch.nanmean(targeting)),
        "targeting_frac_usable": frac,
        "polarisation_R": float(torch.nanmean(polarisation)),
        "polarisation_frac_usable": float(pol_ok.float().mean()),
        "spearman_theta_lever": float(rho_signed.mean()),
        "spearman_theta_abs_lever": float(rho_abs.mean()),
        "theta_by_abs_lever_bucket": buckets,
        "lever_abs_mean": float(absL.mean()),
        "lever_gini": float(_gini(absL)),
    }


@torch.no_grad()
def _gini(v: torch.Tensor) -> torch.Tensor:
    """
    Gini coefficient of |lever| within each probe point, averaged.

    Context for the targeting number: if the levers are all the same size
    (Gini ~ 0) there is nothing for a conditional policy to target and T ~ 1 is
    the expected result, not a failure. If they are highly unequal (Gini -> 1)
    and T is still ~1, the policy genuinely did not find them.
    """
    n = v.shape[-1]
    s = v.sort(dim=-1).values
    w = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    num = ((2 * w - n - 1) * s).sum(-1)
    den = (n * s.sum(-1)).clamp_min(1e-30)
    return (num / den).mean()
