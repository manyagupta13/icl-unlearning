"""
One layer of linear self-attention doing in-context linear regression.

    Q = X W_Q,  K = X W_K,  A = (1/(N+1)) Q K^T,  Yhat = A X
    yhat = Yhat[N+1, D+1]

Writing M = W_Q W_K^T this is

    yhat = (1/(N+1)) * t_q^T M (X^T X) e_{D+1}

and because the query label slot is zero, (X^T X) e_{D+1} = X^T y = sum_i t_i y_i.
So we never form X^T X:

    yhat = (1/(N+1)) * t_q^T M u,      u = sum_i t_i y_i

O(N*D) instead of O(N*D^2). Matters when you scale N.

ATTN-S and ATTN-M share this forward exactly; they differ only in how M is
parameterised, hence in training dynamics:
  ATTN-S  M = W_Q W_K^T   (factored -> multiplicative dynamics, staircase)
  ATTN-M  M               (merged   -> roughly linear dynamics)

An entire shadow ensemble lives in the leading axis S of the parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def context_vector(X: torch.Tensor, ylab: torch.Tensor) -> torch.Tensor:
    """u = sum_i t_i y_i  ->  [S, B, D+1]"""
    return torch.einsum("sbnd,sbn->sbd", X, ylab)


class LinearAttnICL(nn.Module):
    """
    arch: "ATTN-S" | "ATTN-M"
    S:    number of shadow models trained in parallel
    """

    def __init__(self, arch: str, S: int, D: int, N: int,
                 init_scale: float = 0.05, device=None, dtype=torch.float32):
        super().__init__()
        self.arch, self.S, self.D, self.N = arch, S, D, N
        self.scale = 1.0 / (N + 1)
        dp1 = D + 1
        kw = dict(device=device, dtype=dtype)

        if arch == "ATTN-S":
            self.WQ = nn.Parameter(init_scale * torch.randn(S, dp1, dp1, **kw))
            self.WK = nn.Parameter(init_scale * torch.randn(S, dp1, dp1, **kw))
        elif arch == "ATTN-M":
            # match the effective init magnitude of the factored form
            self.M_ = nn.Parameter(init_scale ** 2 * torch.randn(S, dp1, dp1, **kw))
        else:
            raise ValueError(f"unknown arch {arch!r}")

    @property
    def M(self) -> torch.Tensor:
        """Effective read-out matrix [S, D+1, D+1]."""
        if self.arch == "ATTN-S":
            return self.WQ @ self.WK.transpose(1, 2)
        return self.M_

    def forward(self, X: torch.Tensor, ylab: torch.Tensor) -> torch.Tensor:
        """X [S,B,N+1,D+1], ylab [S,B,N+1] -> yhat [S,B]"""
        tq = X[:, :, -1, :]
        u = context_vector(X, ylab)
        return self.scale * torch.einsum("sbd,sde,sbe->sb", tq, self.M, u)

    @torch.no_grad()
    def predict_frozen(self, M: torch.Tensor, X: torch.Tensor,
                       ylab: torch.Tensor) -> torch.Tensor:
        """Forward with an externally supplied M (for frozen sweep evaluation)."""
        tq = X[:, :, -1, :]
        u = context_vector(X, ylab)
        return self.scale * torch.einsum("sbd,sde,sbe->sb", tq, M, u)


class SoftmaxAttnICL(nn.Module):
    """
    One layer of SOFTMAX self-attention on the same tokens. "ATTN-SM".

        q = t_q W_Q,  k_i = t_i W_K,  v_i = t_i W_V
        alpha = softmax_i( q.k_i / sqrt(D+1) )
        yhat  = ( sum_i alpha_i v_i ) . w_out

    Deliberately the SMALLEST change from LinearAttnICL that breaks the closed
    form: same tokens t_i = [x_i ; y_i], same query slot with its label zeroed,
    same one layer, same read-out position. The only difference is that the
    mixing weights are normalised by a softmax over the context instead of by
    the constant 1/(N+1). Anything else varied at the same time (depth, MLPs,
    layer norm) would leave the nonlinearity confounded with capacity, and the
    question here is specifically what the nonlinearity does.

    Why this matters for the rest of the repo: the linear model's prediction is
    linear in the labels, which is what makes theory.py's moments exact and
    policy.py's AUC differentiable without sampling. Here the attention weights
    themselves depend on the labels, so flipping y_i moves alpha as well as v,
    the expectation over a Bernoulli flip no longer factors, and no closed form
    survives. Stage 2 on this architecture has to go through REINFORCE.
    """

    def __init__(self, S: int, D: int, N: int, init_scale: float = 0.05,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.arch, self.S, self.D, self.N = "ATTN-SM", S, D, N
        dp1 = D + 1
        self.att_scale = 1.0 / (dp1 ** 0.5)
        kw = dict(device=device, dtype=dtype)
        self.WQ = nn.Parameter(init_scale * torch.randn(S, dp1, dp1, **kw))
        self.WK = nn.Parameter(init_scale * torch.randn(S, dp1, dp1, **kw))
        self.WV = nn.Parameter(init_scale * torch.randn(S, dp1, dp1, **kw))
        # read-out starts at unit scale: with all three projections at
        # init_scale=0.05 the attention logits start near zero (so alpha is
        # near-uniform, which is the right place to start) but the values are
        # also tiny, and a small w_out on top of that gives a vanishing
        # gradient signal at step 0.
        self.wo = nn.Parameter(torch.randn(S, dp1, **kw) / (dp1 ** 0.5))

    def forward(self, X: torch.Tensor, ylab: torch.Tensor) -> torch.Tensor:
        """X [S,B,N+1,D+1], ylab [S,B,N+1] -> yhat [S,B]"""
        # X's last column is ylab by construction (data.assemble), but take the
        # labels from ylab explicitly so this cannot silently desynchronise if
        # a corruption ever edits one and not the other.
        T = torch.cat([X[..., :self.D], ylab.unsqueeze(-1)], dim=-1)
        tq = T[:, :, -1, :]
        q = torch.einsum("sbd,sde->sbe", tq, self.WQ)
        k = torch.einsum("sbnd,sde->sbne", T, self.WK)
        v = torch.einsum("sbnd,sde->sbne", T, self.WV)
        logits = torch.einsum("sbe,sbne->sbn", q, k) * self.att_scale
        alpha = torch.softmax(logits, dim=-1)
        ctx = torch.einsum("sbn,sbne->sbe", alpha, v)
        return torch.einsum("sbe,se->sb", ctx, self.wo)

    @torch.no_grad()
    def frozen(self) -> "FrozenSoftmax":
        return FrozenSoftmax({k: v.detach().clone()
                              for k, v in self.state_dict().items()},
                             D=self.D)


class FrozenSoftmax:
    """
    A frozen ATTN-SM ensemble, carrying the same interface the rest of the
    pipeline expects of a frozen `M`.

    Everything downstream of training touches an ensemble in exactly two ways:
    `apply_frozen(M, X, ylab, N)` and `M.shape[0]` for the shadow count. Giving
    this object a `shape` and letting `apply_frozen` dispatch on type means
    sweep.py, audit.py, corrupt.py and the Stage 2 scripts work unchanged on a
    nonlinear architecture, with no branching on arch scattered through them.
    The places that genuinely cannot generalise -- theory.py's closed form and
    policy.py's differentiable AUC -- raise instead of silently returning
    something wrong.
    """

    def __init__(self, params: dict, D: int):
        self.params = params
        self.D = D

    @property
    def shape(self):
        return self.params["WQ"].shape

    @property
    def device(self):
        return self.params["WQ"].device

    def to(self, device):
        return FrozenSoftmax({k: v.to(device) for k, v in self.params.items()},
                             D=self.D)

    def cpu(self):
        return self.to("cpu")

    @torch.no_grad()
    def predict(self, X: torch.Tensor, ylab: torch.Tensor) -> torch.Tensor:
        p, D = self.params, self.D
        dp1 = D + 1
        T = torch.cat([X[..., :D], ylab.unsqueeze(-1)], dim=-1)
        tq = T[:, :, -1, :]
        q = torch.einsum("sbd,sde->sbe", tq, p["WQ"])
        k = torch.einsum("sbnd,sde->sbne", T, p["WK"])
        v = torch.einsum("sbnd,sde->sbne", T, p["WV"])
        logits = torch.einsum("sbe,sbne->sbn", q, k) / (dp1 ** 0.5)
        alpha = torch.softmax(logits, dim=-1)
        ctx = torch.einsum("sbn,sbne->sbe", alpha, v)
        return torch.einsum("sbe,se->sb", ctx, p["wo"])


def frozen_to_blob(M):
    """
    Serialisable form of a frozen ensemble.

    Pickling a FrozenSoftmax directly would work, but it would make every
    cached .pt file depend on this class staying importable under the same
    name. A plain dict of tensors with a marker key keeps the artifacts
    readable by anything that can open a torch file. Linear ensembles pass
    through as the bare tensor they always were, so existing caches load
    unchanged.
    """
    if isinstance(M, torch.Tensor):
        return M.cpu()
    return {"__frozen__": "softmax", "D": M.D,
            **{k: v.cpu() for k, v in M.params.items()}}


def frozen_from_blob(obj):
    """Inverse of frozen_to_blob."""
    if isinstance(obj, torch.Tensor):
        return obj
    if not (isinstance(obj, dict) and obj.get("__frozen__") == "softmax"):
        raise TypeError(f"unrecognised frozen-ensemble blob: {type(obj)}")
    D = int(obj["D"])
    params = {k: v for k, v in obj.items()
              if k not in ("__frozen__", "D")}
    return FrozenSoftmax(params, D=D)


LINEAR_ARCHS = ("ATTN-S", "ATTN-M")
SOFTMAX_ARCHS = ("ATTN-SM",)


def is_linear_arch(arch: str) -> bool:
    return arch in LINEAR_ARCHS


def build_model(arch: str, S: int, D: int, N: int, init_scale: float = 0.05,
                device=None, dtype=torch.float32) -> nn.Module:
    if arch in LINEAR_ARCHS:
        return LinearAttnICL(arch, S, D, N, init_scale=init_scale,
                             device=device, dtype=dtype)
    if arch in SOFTMAX_ARCHS:
        return SoftmaxAttnICL(S, D, N, init_scale=init_scale,
                              device=device, dtype=dtype)
    raise ValueError(f"unknown arch {arch!r}")


@torch.no_grad()
def apply_frozen(M, X: torch.Tensor, ylab: torch.Tensor,
                 N: int) -> torch.Tensor:
    """
    Stateless forward for frozen ensembles. This is what the sweep calls:
    the corruption parameter never touches training, so we only ever need the
    frozen read-out.

        M    [S, D+1, D+1] tensor  (linear archs)  or  FrozenSoftmax
        X    [S, P, N+1, D+1]
        ylab [S, P, N+1]
    -> yhat  [S, P]

    The tensor path is byte-for-byte what it always was, so every number the
    paper reports is unaffected by the addition of the dispatch.
    """
    if not isinstance(M, torch.Tensor):
        return M.predict(X, ylab)
    tq = X[:, :, -1, :]
    u = context_vector(X, ylab)
    return torch.einsum("sbd,sde,sbe->sb", tq, M, u) / (N + 1)
