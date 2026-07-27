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
hundred steps. lr~0.005 with grad clipping is the safe default.
"""
from __future__ import annotations

import time

import torch

from .data import MixtureSpec, make_sequences
from .models import LinearAttnICL


def train_ensemble(spec: MixtureSpec, groups: list[str], arch: str, task: str,
                   S: int = 100, batch_per_shadow: int = 8, steps: int = 6000,
                   lr: float = 5e-3, momentum: float = 0.9, optim: str = "sgd",
                   grad_clip: float = 5.0, init_scale: float = 0.05,
                   seed: int = 0, device="cuda", dtype=torch.float32,
                   log_every: int = 500, trace_every: int = 1):
    """
    Returns
        M     [S, D+1, D+1]  frozen effective read-out matrices
        trace [n_logged]     query-MSE trace (for the staircase plot)
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    model = LinearAttnICL(arch, S, spec.D, spec.N, init_scale=init_scale,
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
                                     task, gen, device, dtype)
        yhat = model(X, ylab)
        # mean over batch, summed over shadows: each shadow gets its own grad
        per_shadow = ((yhat - yq) ** 2).mean(dim=1)
        loss = per_shadow.sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip * S)
        opt.step()

        if t % trace_every == 0:
            trace.append(float(per_shadow.mean()))
        if log_every and t % log_every == 0:
            print(f"    [{arch}|{task}] step {t:6d}  loss {per_shadow.mean():.4f}",
                  flush=True)

    with torch.no_grad():
        M = model.M.detach().clone()

    print(f"    [{arch}|{task}] done in {time.time()-t0:.0f}s  "
          f"final {sum(trace[-20:])/max(1,len(trace[-20:])):.4f}", flush=True)
    return M, torch.tensor(trace)


@torch.no_grad()
def per_group_mse(spec: MixtureSpec, M: torch.Tensor, arch: str, task: str,
                  S: int, gen: torch.Generator, device, n_eval: int = 256,
                  dtype=torch.float32) -> dict[str, float]:
    """
    Diagnostic: per-group query MSE. Expect the ordering to track participation
    ratio -- flatter spectrum, higher PR, higher residual error.
    """
    from .models import apply_frozen
    out = {}
    for g in spec.names:
        X, ylab, yq = make_sequences(spec, [g], S, n_eval, task, gen, device, dtype)
        yhat = apply_frozen(M, X, ylab, spec.N)
        out[g] = float(((yhat - yq) ** 2).mean())
    return out
