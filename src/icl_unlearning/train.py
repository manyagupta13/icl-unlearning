"""
Shadow-ensemble training. All S shadows train in parallel as one batched
tensor; each shadow has its own init and its own data stream, so ensemble
variation reflects training randomness the way the audit assumes.

Two hypotheses:
    "full"   trained on every group           (saw the forget group)
    "oracle" retrained on retain groups only  (never saw it)

Optimiser note: use SGD (+ small init) if you want the saddle-to-saddle
staircase to be visible in ATTN-S. Adam flattens it.

Stability note: ATTN-S is a product parameterisation, so momentum bites twice.
Effective LR is lr/(1-beta); lr=0.05 with beta=0.9 diverges to NaN within a few
hundred steps. lr~0.005 with grad clipping is the safe default -- PROVIDED the
clipping is per-shadow. See `clip_grad_norm_per_shadow_`.
"""
from __future__ import annotations

import time

import torch

from .data import MixtureSpec, make_sequences
from .models import FrozenSoftmax, build_model, is_linear_arch


def clip_grad_norm_per_shadow_(params, max_norm: float) -> torch.Tensor:
    """
    Clip each shadow's gradient independently, to `max_norm`.

    `torch.nn.utils.clip_grad_norm_` computes ONE norm across every parameter
    it is given -- here, that means across all S shadows stacked together.
    With S roughly-independent shadows the aggregate norm scales like sqrt(S)
    while a `grad_clip * S` threshold scales like S, so the ratio between
    threshold and aggregate norm GROWS with S: the clip gets looser precisely
    as the ensemble gets bigger. A single diverging shadow barely moves an
    aggregate norm computed over hundreds of others, so it sails through
    un-clipped. Confirmed empirically: at S=512 one ATTN-S oracle shadow blew
    up to loss ~4400 while training was otherwise stable.

    This clips per shadow instead: every parameter tensor is [S, ...], so the
    norm is computed per leading-axis slice and each shadow is scaled by its
    own factor. `max_norm` then means what the README says it means,
    independent of ensemble size.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    S = grads[0].shape[0]
    sq = torch.zeros(S, device=grads[0].device, dtype=torch.float64)
    for g in grads:
        sq += g.reshape(S, -1).double().pow(2).sum(dim=1)
    norms = sq.sqrt()
    scale = (max_norm / (norms + 1e-6)).clamp(max=1.0).to(grads[0].dtype)
    for g in grads:
        g.mul_(scale.reshape([S] + [1] * (g.dim() - 1)))
    return norms.to(grads[0].dtype)


def train_ensemble(spec: MixtureSpec, groups: list[str], arch: str,
                   S: int = 100, batch_per_shadow: int = 8, steps: int = 6000,
                   lr: float = 5e-3, momentum: float = 0.9, optim: str = "sgd",
                   grad_clip: float = 5.0, init_scale: float = 0.05,
                   seed: int = 0, device="cuda", dtype=torch.float32,
                   log_every: int = 500, trace_every: int = 1):
    """
    Returns
        M     frozen read-out: [S, D+1, D+1] for the linear archs, a
              FrozenSoftmax for ATTN-SM. Either is accepted by
              models.apply_frozen and by everything downstream of it.
        trace [n_logged]     query-MSE trace (for the staircase plot)
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    model = build_model(arch, S, spec.D, spec.N, init_scale=init_scale,
                        device=device, dtype=dtype)

    if optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optim == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(optim)

    trace, t0 = [], time.time()
    for t in range(steps):
        X, ylab, yq = make_sequences(spec, groups, S, batch_per_shadow,
                                     gen, device, dtype)
        yhat = model(X, ylab)
        # mean over batch, summed over shadows: each shadow gets its own grad
        per_shadow = ((yhat - yq) ** 2).mean(dim=1)
        loss = per_shadow.sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            clip_grad_norm_per_shadow_(model.parameters(), grad_clip)
        opt.step()

        if t % trace_every == 0:
            trace.append(float(per_shadow.mean()))
        if log_every and t % log_every == 0:
            print(f"    [{arch}] step {t:6d}  loss {per_shadow.mean():.4f}",
                  flush=True)

    with torch.no_grad():
        M = (model.M.detach().clone() if is_linear_arch(arch)
             else model.frozen())

    print(f"    [{arch}] done in {time.time()-t0:.0f}s  "
          f"final {sum(trace[-20:])/max(1,len(trace[-20:])):.4f}", flush=True)
    check_ensemble_health(M, arch)
    return M, torch.tensor(trace)


def check_ensemble_health(M, arch: str, ratio_thresh: float = 20.0):
    """
    Dispatch for architectures whose frozen state is several tensors rather
    than one matrix. The per-shadow norm is taken over all parameters
    concatenated, which is the analogue of ||M|| and catches the same failure:
    a shadow that diverged during training but stayed finite.
    """
    if not isinstance(M, torch.Tensor):
        S = M.shape[0]
        flat = torch.cat([v.reshape(S, -1) for v in M.params.values()], dim=1)
        return _check_norms(flat, arch, ratio_thresh)
    return _check_norms(M.reshape(M.shape[0], -1), arch, ratio_thresh)


def _check_norms(M: torch.Tensor, arch: str, ratio_thresh: float = 20.0):
    """
    Flag shadows whose read-out matrix diverged. Per-shadow clipping should
    prevent this, but this is the check that would have caught the previous
    failure immediately (loss ~4400 on one shadow, silently corrupting the
    oracle's AUC and MMD downstream) instead of surfacing three steps later as
    an out-of-memory error with no obvious connection to training.

    Flags on Frobenius-norm outliers, not just non-finite values: the observed
    divergence was large but finite, so an isfinite() check alone would have
    missed it.
    """
    if not torch.isfinite(M).all():
        bad = (~torch.isfinite(M).reshape(M.shape[0], -1).all(dim=1)).nonzero().flatten()
        raise RuntimeError(
            f"[{arch}] {len(bad)}/{M.shape[0]} shadow(s) have non-finite M "
            f"(indices {bad.tolist()[:10]}...). Training diverged. If this "
            f"recurs after the per-shadow grad-clip fix, lower lr further.")

    norms = torch.linalg.norm(M.reshape(M.shape[0], -1), dim=1)
    med = norms.median()
    ratio = (norms / med.clamp_min(1e-12))
    bad = (ratio > ratio_thresh).nonzero().flatten()
    if len(bad):
        print(f"    [{arch}] WARNING: {len(bad)}/{M.shape[0]} shadow(s) have "
              f"||M|| > {ratio_thresh}x the ensemble median (median={med:.3g}, "
              f"max={norms.max():.3g}). These shadows likely diverged during "
              f"training and will distort spread-based diagnostics and any "
              f"metric computed over the raw population (e.g. mmd2). "
              f"Indices: {bad.tolist()[:20]}", flush=True)


@torch.no_grad()
def per_group_mse(spec: MixtureSpec, M: torch.Tensor, arch: str,
                  S: int, gen: torch.Generator, device, n_eval: int = 256,
                  dtype=torch.float32) -> dict[str, float]:
    """
    Diagnostic: per-group query MSE. Expect the ordering to track participation
    ratio -- flatter spectrum, higher PR, higher residual error.
    """
    from .models import apply_frozen
    out = {}
    for g in spec.names:
        X, ylab, yq = make_sequences(spec, [g], S, n_eval, gen, device, dtype)
        yhat = apply_frozen(M, X, ylab, spec.N)
        out[g] = float(((yhat - yq) ** 2).mean())
    return out
