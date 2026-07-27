"""
Context-corruption families applied to the forget-set tokens of a frozen probe.

Every corruption is a PROMPT-TIME edit: no weight update, no retraining. That
is what lets the sweep reuse one trained ensemble across the whole grid.

Each shadow model draws its own corruption noise, which is realistic but has a
consequence worth remembering: stochastic corruptions inflate within-ensemble
spread, and that lowers membership AUC mechanically without removing anything.
Compare against `audit.spread` before reading a drop as removal.

Families
    none    identity (control; gives the un-edited AUC)
    C1      additive label noise      y_f += eps,  eps ~ N(0, s)      param = s = sigma^2
    C2      additive input noise      x_f += eps,  eps ~ N(0, s I)    param = s = sigma^2
    C3      both
    flip    tunable sign strength     y_f -> (1 - 2t) y_f             param = t
            t=0 identity, t=1 full sign-flip (parameter-free ICUL),
            t=0.5 the zero-mean dead zone
    whiten  map forget inputs toward the retain covariance
            x_f -> C_ret^{1/2} C_f^{-1/2} x_f, interpolated by param in [0,1]
            (input-space; uses EMPIRICAL covariances, never the true Lambda)
"""
from __future__ import annotations

import torch

from .data import Probe, assemble


def _expand(probe: Probe, S: int):
    x = probe.x.unsqueeze(0).expand(S, *probe.x.shape).clone()
    y = probe.y.unsqueeze(0).expand(S, *probe.y.shape).clone()
    return x, y


def _empirical_cov(x: torch.Tensor) -> torch.Tensor:
    """x [..., n, D] -> [D, D] pooled empirical covariance."""
    f = x.reshape(-1, x.shape[-1])
    f = f - f.mean(0, keepdim=True)
    return (f.T @ f) / max(1, f.shape[0] - 1)


def _inv_sqrt(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    e, V = torch.linalg.eigh(C.double())
    e = e.clamp_min(eps)
    return (V @ torch.diag(e.rsqrt()) @ V.T).to(C.dtype)


def _sqrt(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    e, V = torch.linalg.eigh(C.double())
    e = e.clamp_min(eps)
    return (V @ torch.diag(e.sqrt()) @ V.T).to(C.dtype)


def corrupt(probe: Probe, S: int, mode: str, param: float,
            gen: torch.Generator, retain_x: torch.Tensor | None = None):
    """
    Build per-shadow corrupted copies of the probe.

    retain_x: [*, D] sample of retain-group inputs, required for mode="whiten".
              Estimated empirically -- do NOT pass the true covariance.

    Returns (X, ylab, yq) with X [S, P, N+1, D+1].
    """
    x, y = _expand(probe, S)
    sl = probe.forget_slice
    dev, dt = x.device, x.dtype

    if mode == "none":
        pass

    elif mode in ("C1", "C3"):
        s = float(param) ** 0.5
        shp = y[:, :, sl].shape
        y[:, :, sl] += s * torch.randn(shp, generator=gen, device=dev, dtype=dt)
        if mode == "C3":
            shp = x[:, :, sl, :].shape
            x[:, :, sl, :] += s * torch.randn(shp, generator=gen, device=dev, dtype=dt)

    elif mode == "C2":
        s = float(param) ** 0.5
        shp = x[:, :, sl, :].shape
        x[:, :, sl, :] += s * torch.randn(shp, generator=gen, device=dev, dtype=dt)

    elif mode == "flip":
        y[:, :, sl] *= (1.0 - 2.0 * float(param))

    elif mode == "whiten":
        if retain_x is None:
            raise ValueError("mode='whiten' needs retain_x")
        C_f = _empirical_cov(probe.x[:, sl, :])
        C_r = _empirical_cov(retain_x)
        W = _sqrt(C_r) @ _inv_sqrt(C_f)
        I = torch.eye(W.shape[0], device=dev, dtype=dt)
        Wt = (1.0 - float(param)) * I + float(param) * W       # interpolate
        x[:, :, sl, :] = x[:, :, sl, :] @ Wt.T

    else:
        raise ValueError(f"unknown corruption mode {mode!r}")

    return assemble(x, y)


#: grids are declared in the config; this is only the registry of valid modes
MODES = ("none", "C1", "C2", "C3", "flip", "whiten")
