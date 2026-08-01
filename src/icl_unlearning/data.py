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


# ------------------------------------------------- parametric spectrum family
#
# Hand-written eigenvalue lists do not survive a change of D: to sweep D you
# must decide what the eigenvalues become, and that choice silently determines
# the answer. So spectra are generated from one continuous knob instead.
#
#     lambda_k  proportional to  exp(-gamma * k),   k = 0 .. D-1
#
# gamma = 0 gives a flat spectrum (PR = D, hardest); large gamma concentrates
# all mass on one axis (PR -> 1, easiest). PR is monotone decreasing in gamma
# at fixed D, so it can be inverted numerically -- which lets a group be
# specified by the quantity that actually matters (its participation ratio)
# rather than by D magic numbers.

def exp_spectrum(D: int, gamma: float) -> list[float]:
    """Trace-normalised exponential-decay spectrum, descending."""
    k = torch.arange(D, dtype=torch.float64)
    e = torch.exp(-float(gamma) * k)
    return (e / e.sum()).tolist()


def pr_of_gamma(D: int, gamma: float) -> float:
    return participation_ratio(torch.tensor(exp_spectrum(D, gamma),
                                            dtype=torch.float64))


def gamma_for_pr(D: int, target_pr: float, tol: float = 1e-10,
                 max_iter: int = 200) -> float:
    """
    Invert PR -> gamma by bisection. PR is continuous and strictly decreasing
    in gamma on (0, inf), with PR(0) = D exactly, so a bracket always exists
    for any target in (1, D).

    Raises on out-of-range targets rather than silently clipping: asking for
    PR >= D or PR <= 1 is a specification error, not something to paper over.
    """
    if not (1.0 < target_pr < D):
        raise ValueError(
            f"target PR must lie strictly in (1, D)=(1, {D}); got {target_pr}. "
            f"PR=D means a perfectly flat spectrum (gamma=0), PR=1 means all "
            f"mass on one axis (gamma=inf); neither is reachable exactly.")
    lo, hi = 0.0, 1.0
    while pr_of_gamma(D, hi) > target_pr:      # grow until PR drops below target
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError(f"could not bracket PR={target_pr} at D={D}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if pr_of_gamma(D, mid) > target_pr:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def spectrum_for_pr(D: int, target_pr: float) -> list[float]:
    """Trace-normalised spectrum at dimension D with the requested PR."""
    return exp_spectrum(D, gamma_for_pr(D, target_pr))


@dataclass
class MixtureSpec:
    """Group spectra. `eigs[g]` is the sorted eigenvalue list for group g."""
    names: list[str]
    eigs: dict[str, list[float]]
    D: int
    N: int
    basis: str = "identity"          # "identity" | "random"
    seed: int = 0
    # "regression"     y = beta . x                (continuous targets)
    # "classification" y = sign(beta . x) in {-1,+1}
    #
    # Only the target map changes. The model is unchanged: it still emits a
    # continuous yhat under squared loss, so a shadow is still one matrix and
    # the ensemble economics are identical. Two things do change and both are
    # improvements: a label FLIP becomes the literal operation it is named
    # after rather than the negation of a continuous value, and sum_i y_i^2 =
    # n_f EXACTLY, which removes the only sampling noise from C2's closed-form
    # variance (see theory.py).
    task: str = "regression"
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

    def targets(self, raw: torch.Tensor) -> torch.Tensor:
        """Map beta . x to the task's targets. sign(0) is sent to +1."""
        if self.task == "regression":
            return raw
        if self.task == "classification":
            s = torch.sign(raw)
            return torch.where(s == 0, torch.ones_like(s), s)
        raise ValueError(f"unknown task {self.task!r}")

    def sample(self, g: str, shape, gen, device, dtype) -> torch.Tensor:
        """Draw inputs from group `g`. Returns [*shape, D]."""
        z = torch.randn(*shape, self.D, generator=gen, device=device, dtype=dtype)
        return z @ self.sqrt_cov(g, device, dtype).T


# --------------------------------------------------------------- MNIST variant

@dataclass
class MnistSpec:
    """
    Same interface as MixtureSpec, but inputs are real MNIST images rather than
    Gaussian draws. Groups are digit classes; the forget group is a digit.

    Design constraint (the one that decides whether this measures anything):
    the per-sequence task vector beta is SHARED across groups, exactly as in the
    synthetic setup, so the retain tokens remain informative about a forget-group
    query. If instead each class carried its own label rule, corrupting the
    forget tokens would delete the answer outright and both hypotheses would
    collapse to chance together -- a clean curve measuring nothing.

    So digit classes supply only the INPUT DISTRIBUTION; targets are
    y = sign(beta . x) with beta fresh per sequence.

    Centering. `center="class"` subtracts each digit's own mean, which keeps
    spectral geometry the sole axis of variation and matches MixtureSpec's
    "groups share mean 0" contract. `center="pooled"` keeps the class means,
    which preserves what actually distinguishes digits but introduces a mean
    channel that theory.py's moments do not model. Run scripts/mnist_pr_probe.py
    to see the size of that channel before choosing.
    """
    names: list[str]                 # e.g. ["d1", "d3", "d8"]
    digits: list[int]                # the digit each group corresponds to
    D: int
    N: int
    forget: str = ""
    task: str = "classification"
    center: str = "class"            # "class" | "pooled"
    seed: int = 0
    banks: dict = field(default_factory=dict, repr=False)   # group -> [n_g, D]

    def __post_init__(self):
        assert len(self.names) == len(self.digits)
        if not self.banks:
            self._build()

    def _build(self):
        from .mnist import load_feature_banks
        self.banks = load_feature_banks(self.digits, self.names, self.D,
                                        self.center, self.seed)

    def sqrt_cov(self, g, device, dtype):
        """Empirical covariance square root -- used only by the whiten arm."""
        b = self.banks[g].to(device=device, dtype=dtype)
        c = (b.T @ b) / (b.shape[0] - 1)
        e, V = torch.linalg.eigh(c.double())
        return (V @ torch.diag(e.clamp_min(1e-12).sqrt()) @ V.T).to(dtype)

    def pr(self, g: str) -> float:
        b = self.banks[g]
        c = (b.T @ b) / (b.shape[0] - 1)
        e = torch.linalg.eigvalsh(c).clamp_min(0)
        return float(e.sum() ** 2 / (e ** 2).sum())

    def targets(self, raw: torch.Tensor) -> torch.Tensor:
        return MixtureSpec.targets(self, raw)

    def sample(self, g: str, shape, gen, device, dtype) -> torch.Tensor:
        """Draw real images (as PCA features) with replacement from group `g`."""
        b = self.banks[g].to(device=device, dtype=dtype)
        n = int(torch.tensor(shape).prod())
        idx = torch.randint(0, b.shape[0], (n,), generator=gen, device=device)
        return b[idx].reshape(*shape, self.D)


# ---------------------------------------------------------------- spec factory

def build_spec(cfg_data: dict):
    """
    Build the right spec from a config's `data` block. Keeps every driver
    script agnostic about whether the inputs are Gaussian or MNIST.
    """
    if cfg_data.get("source", "gaussian") == "mnist":
        return MnistSpec(names=cfg_data["groups"], digits=cfg_data["digits"],
                         D=cfg_data["D"], N=cfg_data["N"],
                         forget=cfg_data.get("forget", ""),
                         task=cfg_data.get("task", "classification"),
                         center=cfg_data.get("center", "class"),
                         seed=cfg_data.get("seed", 0))
    return MixtureSpec(names=cfg_data["groups"], eigs=cfg_data["eigs"],
                       D=cfg_data["D"], N=cfg_data["N"],
                       basis=cfg_data.get("basis", "identity"),
                       seed=cfg_data.get("seed", 0),
                       task=cfg_data.get("task", "regression"))


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

    # Draw every group's inputs, then select per (shadow, batch) by gid. Going
    # through spec.sample keeps this identical for Gaussian and MNIST specs;
    # the latter cannot be expressed as z @ L^{1/2}.
    per_group = torch.stack([spec.sample(g, (S, B, N + 1), gen, device, dtype)
                             for g in groups])                  # [G,S,B,N+1,D]
    x = per_group.gather(
        0, gid[None, :, :, None, None].expand(1, S, B, N + 1, D)).squeeze(0)

    beta = torch.randn(S, B, D, generator=gen, device=device, dtype=dtype)
    y = spec.targets(torch.einsum("sbnd,sbd->sbn", x, beta))

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
    xs = [spec.sample(g, (P, counts[g]), gen, device, dtype) for g in order]
    # query token from the forget group
    xs.append(spec.sample(forget, (P, 1), gen, device, dtype))
    x = torch.cat(xs, dim=1)                                    # [P, N+1, D]

    beta = torch.randn(P, D, generator=gen, device=device, dtype=dtype)
    y = spec.targets(torch.einsum("pnd,pd->pn", x, beta))

    start = N - counts[forget]
    return Probe(x=x, y=y, forget_slice=slice(start, N), P=P)


def assemble(x: torch.Tensor, y: torch.Tensor):
    """[..., N+1, D], [..., N+1] -> (X, ylab, yq) with the query label hidden."""
    yq = y[..., -1].clone()
    ylab = y.clone()
    ylab[..., -1] = 0.0
    X = torch.cat([x, ylab.unsqueeze(-1)], dim=-1)
    return X, ylab, yq
