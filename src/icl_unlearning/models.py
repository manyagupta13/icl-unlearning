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


@torch.no_grad()
def apply_frozen(M: torch.Tensor, X: torch.Tensor, ylab: torch.Tensor,
                 N: int) -> torch.Tensor:
    """
    Stateless forward for frozen ensembles. This is what the sweep calls:
    the corruption parameter never touches training, so we only ever need M.
        M    [S, D+1, D+1]
        X    [S, P, N+1, D+1]
        ylab [S, P, N+1]
    -> yhat  [S, P]
    """
    tq = X[:, :, -1, :]
    u = context_vector(X, ylab)
    return torch.einsum("sbd,sde,sbe->sb", tq, M, u) / (N + 1)
