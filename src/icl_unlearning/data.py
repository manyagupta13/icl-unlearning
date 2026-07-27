"""
Covariance-defined Gaussian mixture and in-context sequence generation.

Groups share mean 0 and mixture weight; they differ ONLY in covariance
spectrum, each trace-normalised so all carry equal signal energy. Spectral
geometry is therefore the sole axis of variation.

All tensors carry a leading shadow-model axis S so an entire ensemble trains
as one batched op.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------- spectra

def trace_normalise(eigs: torch.Tensor) -> torch.Tensor:
    return eigs / eigs.sum()


def participation_ratio(eigs: torch.Tensor) -> float:
    """PR = tr(L)^2 / tr(L^2). PR=1 -> one dominant axis; PR=D -> flat."""
    return float(eigs.sum() ** 2 / (eigs ** 2).sum())


@dataclass
class MixtureSpec:
    """Group spectra. `eigs[g]` is the sorted eigenvalue list for group g."""
    names: list[str]
    eigs: dict[str, list[float]]
    D: int
    N: int
    basis: str = "identity"          # "identity" | "random"
    seed: int = 0
    _rot: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for g in self.names:
            assert len(self.eigs[g]) == self.D, f"group {g}: need {self.D} eigenvalues"
        if self.basis == "random":
            gen = torch.Generator().manual_seed(self.seed)
            for g in self.names:
                a = torch.randn(self.D, self.D, generator=gen)
                q, _ = torch.linalg.qr(a)
                self._rot[g] = q

    def sqrt_cov(self, g: str, device, dtype) -> torch.Tensor:
        """Returns L^{1/2} as [D, D]; diagonal unless basis='random'."""
        e = trace_normalise(torch.tensor(self.eigs[g], dtype=dtype))
        s = torch.diag(e.sqrt()).to(device)
        if self.basis == "random":
            q = self._rot[g].to(device=device, dtype=dtype)
            return q @ s @ q.T
        return s

    def pr(self, g: str) -> float:
        return participation_ratio(trace_normalise(torch.tensor(self.eigs[g])))


# ------------------------------------------------------------------- sequences

def make_sequences(spec: MixtureSpec, groups: list[str], S: int, B: int,
                   gen: torch.Generator, device, dtype=torch.float32):
    """
    One in-context task per (shadow, batch) element: N context pairs + 1 query,
    all drawn from a single group sampled uniformly from `groups`.

    Returns
        X    [S, B, N+1, D+1]  tokens [x_i ; y_i], query label slot zeroed
        ylab [S, B, N+1]       label column, query slot zeroed
        yq   [S, B]            true query label
    """
    D, N = spec.D, spec.N
    G = len(groups)

    gid = torch.randint(0, G, (S, B), generator=gen, device=device)
    half = torch.stack([spec.sqrt_cov(g, device, dtype) for g in groups])   # [G,D,D]
    Lh = half[gid]                                                         # [S,B,D,D]

    z = torch.randn(S, B, N + 1, D, generator=gen, device=device, dtype=dtype)
    x = torch.einsum("sbnd,sbed->sbne", z, Lh)

    beta = torch.randn(S, B, D, generator=gen, device=device, dtype=dtype)
    y = torch.einsum("sbnd,sbd->sbn", x, beta)

    yq = y[:, :, -1].clone()
    ylab = y.clone()
    ylab[:, :, -1] = 0.0
    X = torch.cat([x, ylab.unsqueeze(-1)], dim=-1)
    return X, ylab, yq


# ----------------------------------------------------------------------- probe

@dataclass
class Probe:
    """
    Frozen audit probe: mixed context (retain + forget tokens) with the query
    drawn from the forget group. Shared across every shadow model, so
    cross-model variation reflects membership alone.
    """
    x: torch.Tensor          # [P, N+1, D]
    y: torch.Tensor          # [P, N+1]
    forget_slice: slice      # indices of forget-group context tokens
    P: int


def make_probe(spec: MixtureSpec, counts: dict[str, int], forget: str,
               P: int, gen: torch.Generator, device,
               dtype=torch.float32) -> Probe:
    """
    counts: tokens per group in the context, e.g. {"z1": 10, "z2": 10, "z3": 11}.
            Must sum to spec.N. Forget-group tokens are placed last so the
            slice is contiguous.
    """
    D, N = spec.D, spec.N
    assert sum(counts.values()) == N, f"context counts must sum to N={N}"

    order = [g for g in spec.names if g != forget] + [forget]
    xs = []
    for g in order:
        n = counts[g]
        z = torch.randn(P, n, D, generator=gen, device=device, dtype=dtype)
        xs.append(z @ spec.sqrt_cov(g, device, dtype).T)
    # query token from the forget group
    zq = torch.randn(P, 1, D, generator=gen, device=device, dtype=dtype)
    xs.append(zq @ spec.sqrt_cov(forget, device, dtype).T)
    x = torch.cat(xs, dim=1)                                    # [P, N+1, D]

    beta = torch.randn(P, D, generator=gen, device=device, dtype=dtype)
    y = torch.einsum("pnd,pd->pn", x, beta)

    start = N - counts[forget]
    return Probe(x=x, y=y, forget_slice=slice(start, N), P=P)


def assemble(x: torch.Tensor, y: torch.Tensor):
    """[..., N+1, D], [..., N+1] -> (X, ylab, yq) with the query label hidden."""
    yq = y[..., -1].clone()
    ylab = y.clone()
    ylab[..., -1] = 0.0
    X = torch.cat([x, ylab.unsqueeze(-1)], dim=-1)
    return X, ylab, yq
