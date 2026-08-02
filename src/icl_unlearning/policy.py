r"""
Stage 2: a LEARNED in-context corruption, p_theta(xtilde_f, ytilde_f | x_f, y_f).

This implements the "Generalize" half of the brief. Stage 1 sweeps hand-designed
corruptions (C1/C2/C3/flip/bern) and plots AUC against Var(eps). Stage 2 replaces
the hand-designed corruption with one that is *optimised*.

THE PARAMETERISATION
--------------------
The brief factorises the policy as

    p_theta(xtilde_f, ytilde_f | x_f, y_f)
        = p_theta(ytilde_f | xtilde_f, x_f, y_f) . p_theta(xtilde_f | x_f, y_f)
          \_______________  _______________/     \________  ________/
                          \/                              \/
                     ~ Bern(theta)                    = delta(x_f)

i.e. the features pass through untouched and the label is flipped with
probability theta. Two policies are provided:

    ScalarBernoulli   one theta for every forget token          (the brief's
                                                                 page-3 form)
    ConditionalBernoulli  theta_i = sigmoid(MLP([x_i ; y_i]))    (the brief's
                                                                 page-2 "NN")

WHY THERE IS NO REINFORCE IN THE DEFAULT PATH
---------------------------------------------
The brief derives a policy gradient because AUC(theta) is not differentiable
through the sampling of ytilde_f:

    grad E[AUC] = E[ grad log Q_theta(y_f) . AUC(y_f) ]

That derivation is correct and `reinforce_grad` implements it. But for THIS
parameterisation it is unnecessary, because the expectation can be taken in
closed form before any sampling happens. With a_i = y_i (c_x . x_i),

    E[dyhat]   = -2/(N+1)   sum_i theta_i a_i
    Var[dyhat] =  4/(N+1)^2 sum_i theta_i (1 - theta_i) a_i^2

for INDEPENDENT per-token flips -- scalar or conditional, both covered. Feed
those two moments into the Gaussian AUC of theory.py and AUC(theta) becomes an
ordinary differentiable function of theta. No sampling, no score-function
estimator, no variance.

This is verified two ways: the moments against Monte Carlo
(tests/verify_bern.py), and the closed-form gradient against REINFORCE
(scripts/stage2_optimise.py, which reports both).

REINFORCE remains necessary if the policy also perturbs the FEATURES, since
p_theta(xtilde_f | x_f, y_f) = delta(x_f) is what kills the cross terms. That
path is kept for exactly that reason.

A CORRECTION TO THE OBJECTIVE
-----------------------------
The brief writes

    min_theta  AUC(theta) + lambda ||theta||

Stage 1 measured AUC sitting BELOW 0.5 and rising toward it as corruption grows
(0.417 at zero corruption for rot_mid; see the report). Minimising AUC directly
therefore pushes the corruption toward *more* distinguishability, not less --
AUC = 0.3 is a better attacker than AUC = 0.5, just an inverted one. The
success criterion is AUC = 0.5, so the objective must penalise distance from
chance:

    min_theta  (AUC(theta) - 1/2)^2 + lambda ||theta||

`objective="auc"` keeps the literal form for comparison; `objective="dist"`
(the default) is the corrected one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import Probe
from .theory import readout_covector


# --------------------------------------------------------------------- policies

class ScalarBernoulli(nn.Module):
    """One flip probability for every forget token. The brief's page-3 form."""

    def __init__(self, init: float = 0.05):
        super().__init__()
        p = min(max(init, 1e-4), 1 - 1e-4)
        self.logit = nn.Parameter(torch.tensor(math.log(p / (1 - p))))

    def forward(self, x_f: torch.Tensor, y_f: torch.Tensor) -> torch.Tensor:
        """-> theta broadcast to [P, n_f]"""
        return torch.sigmoid(self.logit).expand(y_f.shape)

    def penalty(self) -> torch.Tensor:
        return torch.sigmoid(self.logit).abs()

    def budget(self, x_f: torch.Tensor, y_f: torch.Tensor) -> torch.Tensor:
        """Mean flip probability. See ConditionalBernoulli.budget for why this
        exists alongside penalty()."""
        return self(x_f, y_f).mean()


class ConditionalBernoulli(nn.Module):
    """
    theta_i = sigmoid(MLP([x_i ; y_i])) -- the brief's page-2 neural policy.
    Lets the corruption spend its budget where it actually buys removal instead
    of flipping every token equally.
    """

    def __init__(self, D: int, hidden: int = 32, init_bias: float = -3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.fill_(init_bias)     # start near "do nothing"

    def forward(self, x_f: torch.Tensor, y_f: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_f, y_f.unsqueeze(-1)], dim=-1)      # [P, n_f, D+1]
        return torch.sigmoid(self.net(z).squeeze(-1))        # [P, n_f]

    def penalty(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.net.parameters()).sqrt()

    def budget(self, x_f: torch.Tensor, y_f: torch.Tensor) -> torch.Tensor:
        """
        Mean flip probability, i.e. the expected fraction of forget labels that
        get flipped.

        penalty() is NOT usable for comparing the two policies. For
        ScalarBernoulli it returns theta; here it returns the L2 norm of the MLP
        weights, which is in different units and has no fixed relationship to how
        much corruption is applied. Regularising both by `penalty` and calling
        the resulting curves comparable would be wrong -- the same lambda would
        mean different things. `budget` is the same quantity for both, so
        sweeping lambda against it traces frontiers that can be laid on the same
        axes.
        """
        return self(x_f, y_f).mean()


# ------------------------------------------------------------- closed-form AUC

def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def bern_moments(theta: torch.Tensor, a: torch.Tensor, N: int):
    """
    Mean shift and variance of yhat under independent per-token Bernoulli flips.

        theta [P, n_f] or broadcastable      a [S, P, n_f] with a_i = y_i (c_x.x_i)

    Returns (shift [S,P], var [S,P]). Works for scalar and conditional policies
    alike, which is the whole reason REINFORCE is avoidable here.
    """
    shift = -2.0 * (theta * a).sum(-1) / (N + 1)
    var = 4.0 * (theta * (1.0 - theta) * a ** 2).sum(-1) / (N + 1) ** 2
    return shift, var


def policy_auc(M_full: torch.Tensor, M_oracle: torch.Tensor, probe: Probe,
               policy: nn.Module) -> torch.Tensor:
    """
    Differentiable matched-context membership AUC under the policy.

    Mirrors theory.predicted_auc exactly -- same Gaussian approximation, same
    two variance sources (shadow spread + per-shadow corruption draw), same
    per-probe-point averaging -- but with the corruption moments coming from
    `policy` instead of a fixed sigma^2. Returns a scalar tensor with grad.

    LINEAR ARCHITECTURES ONLY. bern_moments is exact because the prediction is
    linear in the labels, so the expectation over independent flips factors. If
    the attention weights themselves depend on the labels that is false, and
    every step below would be quietly computing the wrong objective. Refusing
    is the only safe behaviour; use reinforce_grad there.
    """
    if not isinstance(M_full, torch.Tensor):
        raise TypeError(
            f"policy_auc needs a linear read-out; got {type(M_full).__name__}. "
            "The closed-form AUC assumes the prediction is linear in the "
            "labels, which fails once the attention weights depend on them. "
            "Optimise with policy.reinforce_grad instead -- see "
            "scripts/stage2_conditional.py --estimator reinforce.")
    sl = probe.forget_slice
    N = probe.x.shape[1] - 1
    x_f, y_f = probe.x[:, sl, :], probe.y[:, sl]
    theta = policy(x_f, y_f)                                  # [P, n_f]

    yq = probe.y[:, -1]
    sgn = torch.sign(yq)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)

    stats = []
    for M in (M_full, M_oracle):
        c_x, _, yhat0 = readout_covector(M, probe)            # [S,P,D], [S,P]
        a = torch.einsum("spd,pid->spi", c_x, x_f) * y_f      # [S,P,n_f]
        shift, var = bern_moments(theta, a, N)
        centre = yhat0 + shift
        mu = sgn * (centre.mean(dim=0) - yq)
        v = centre.var(dim=0, unbiased=True) + var.mean(dim=0)
        stats.append((mu, v))

    (mu1, v1), (mu0, v0) = stats
    # negated: the membership score is -residual so the member sits on
    # the high side (audit.membership_score)
    z = (mu0 - mu1) / torch.sqrt((v1 + v0).clamp_min(1e-30))
    return _normal_cdf(z).mean()


def objective_value(auc: torch.Tensor, policy: nn.Module, lam: float,
                    objective: str = "dist") -> torch.Tensor:
    """
    `dist` : (AUC - 1/2)^2 + lam ||theta||   -- chance is the target
    `auc`  : AUC + lam ||theta||             -- the brief's literal form; see
                                                the module docstring for why it
                                                is the wrong target here
    """
    if objective == "dist":
        core = (auc - 0.5) ** 2
    elif objective == "auc":
        core = auc
    else:
        raise ValueError(objective)
    return core + lam * policy.penalty()


# ------------------------------------------------------------------- REINFORCE

def reinforce_grad(M_full: torch.Tensor, M_oracle: torch.Tensor, probe: Probe,
                   policy: nn.Module, auc_fn, n_samples: int = 32,
                   baseline: float | None = None, generator=None):
    """
    The brief's score-function estimator:

        grad E[AUC] = E[ grad log Q_theta(B) . AUC(B) ]

    `auc_fn(B) -> float` scores one sampled flip mask by running the actual
    frozen forward pass, so this makes no Gaussian assumption at all. Provided
    for comparison against the closed form, and required if the policy is ever
    extended to perturb features.

    A mean baseline is subtracted -- without it the estimator is unusable, since
    AUC ~ 0.5 makes the raw score-function term enormous relative to its own
    variation.
    """
    x_f, y_f = probe.x[:, probe.forget_slice, :], probe.y[:, probe.forget_slice]
    theta = policy(x_f, y_f)

    scores, logps = [], []
    for _ in range(n_samples):
        with torch.no_grad():
            B = (torch.rand(theta.shape, generator=generator,
                            device=theta.device) < theta).to(theta.dtype)
        logp = (B * torch.log(theta.clamp_min(1e-8))
                + (1 - B) * torch.log((1 - theta).clamp_min(1e-8))).sum()
        scores.append(float(auc_fn(B)))
        logps.append(logp)

    s = torch.tensor(scores, dtype=theta.dtype, device=theta.device)
    b = s.mean() if baseline is None else baseline
    loss = ((s - b).detach() * torch.stack(logps)).mean()
    return loss, float(s.mean()), float(s.std())
