"""
Sweep orchestration.

The frozen ensembles are loaded once. For each (arch, mode, strength) we
re-corrupt the probe, run frozen forward passes, and score both observables.
No training happens in here -- that is the whole point.

Two H0 conventions are computed at every grid point, because they answer
different questions and only one of them is a membership test:

  clean-context H0    oracle on the un-edited probe. This is the original
                      convention. It is CONFOUNDED: H1 sees a corrupted context
                      and H0 does not, so a distinguisher can score above 0.5
                      by detecting the edit rather than by detecting membership.
                      As corruption strength grows this confound grows with it,
                      which is exactly the regime the sweep is probing.

  matched-context H0  oracle on the SAME corrupted probe, same noise draw.
                      Context is now identical across hypotheses, so the only
                      remaining difference is what the weights saw in training.
                      This is the membership estimand. Report this one.

Both are emitted (`auc_*` and `auc_matched_*`) so the gap between them is
visible; that gap is the size of the confound.
"""
from __future__ import annotations

import torch

from . import audit
from .corrupt import corrupt
from .data import MixtureSpec, Probe, assemble
from .models import apply_frozen


@torch.no_grad()
def sweep_point(spec: MixtureSpec, probe: Probe, M_full: torch.Tensor,
                M_oracle: torch.Tensor, mode: str, param: float,
                gen: torch.Generator, retain_x=None, n_boot: int = 200,
                boot_seed: int = 0) -> dict:
    """One grid point. Returns a flat record ready for a dataframe row."""
    S = M_full.shape[0]

    # H1: fully trained + in-context edit
    X1, yl1, yq1 = corrupt(probe, S, mode, param, gen, retain_x=retain_x)
    yhat1 = apply_frozen(M_full, X1, yl1, spec.N)

    # H0a: retrain oracle on the CLEAN probe (legacy, confounded -- see module docstring)
    x0 = probe.x.unsqueeze(0).expand(S, *probe.x.shape)
    y0 = probe.y.unsqueeze(0).expand(S, *probe.y.shape)
    X0, yl0, yq0 = assemble(x0, y0)
    yhat0 = apply_frozen(M_oracle, X0, yl0, spec.N)

    # H0b: retrain oracle on the SAME corrupted probe -- matched context.
    # Reuses (X1, yl1) so the noise draw is identical across hypotheses.
    yhat0m = apply_frozen(M_oracle, X1, yl1, spec.N)

    # Masking control: same corruption, one noise draw shared across shadows.
    # Deterministic modes are unaffected, so skip the extra work there.
    stochastic = mode in ("C1", "C2", "C3")
    if stochastic:
        Xs, yls, yqs = corrupt(probe, S, mode, param, gen, retain_x=retain_x,
                               shared_noise=True)
        yhat1s = apply_frozen(M_full, Xs, yls, spec.N)
        yhat0s = apply_frozen(M_oracle, Xs, yls, spec.N)

    obs1 = audit.observables(yhat1, yq1)
    obs0 = audit.observables(yhat0, yq0)
    obs0m = audit.observables(yhat0m, yq1)

    rec = {"mode": mode, "param": float(param)}
    for name in ("loss", "output"):
        a = audit.membership_auc(obs1[name], obs0[name])
        rec[f"auc_{name}"] = a
        rec[f"auc_{name}_sym"] = audit.symmetrised_auc(a)

        am, lo, hi = audit.membership_auc_ci(obs1[name], obs0m[name],
                                             n_boot=n_boot, seed=boot_seed)
        rec[f"auc_matched_{name}"] = am
        rec[f"auc_matched_{name}_lo"] = lo
        rec[f"auc_matched_{name}_hi"] = hi

        if stochastic:
            o1s = audit.observables(yhat1s, yqs)[name]
            o0s = audit.observables(yhat0s, yqs)[name]
            a_sh = audit.membership_auc(o1s, o0s)
        else:
            a_sh = am
        rec[f"auc_shared_{name}"] = a_sh
        # positive => part of the AUC drop is variance masking, not removal
        rec[f"masking_{name}"] = a_sh - am

        rec[f"spread_h1_{name}"] = audit.spread(obs1[name])
        rec[f"spread_h0_{name}"] = audit.spread(obs0[name])

    # distributional criterion on the forget-population residual law
    p = audit.fit_residual_law(yhat1, yq1)
    p2 = audit.fit_residual_law(yhat0, yq0)
    X_c, yl_c, yq_c = corrupt(probe, S, "none", 0.0, gen)
    p1 = audit.fit_residual_law(apply_frozen(M_full, X_c, yl_c, spec.N), yq_c)
    rec.update(audit.alpha_eps(p1, p2, p))
    rec["mmd2_to_oracle"] = audit.mmd2((yhat1 - yq1).reshape(-1),
                                       (yhat0 - yq0).reshape(-1))
    return rec


@torch.no_grad()
def run_sweep(spec: MixtureSpec, probe: Probe, ensembles: dict, grids: dict,
              seed: int = 0, device="cuda", retain_x=None,
              n_boot: int = 200) -> list[dict]:
    """
    ensembles: {(arch, hyp): M}
    grids:     {mode: [param, ...]}
    Returns a list of records with `arch` attached.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []
    archs = sorted({a for (a, _) in ensembles})
    for arch in archs:
        M_full = ensembles[(arch, "full")]
        M_oracle = ensembles[(arch, "oracle")]
        for mode, params in grids.items():
            for k, prm in enumerate(params):
                rec = sweep_point(spec, probe, M_full, M_oracle, mode, prm,
                                  gen, retain_x=retain_x, n_boot=n_boot,
                                  boot_seed=seed + 1000 * k)
                rec["arch"] = arch
                rows.append(rec)
                print(f"  {arch:7s} {mode:7s} p={prm:<8.4g} "
                      f"AUCm(loss)={rec['auc_matched_loss']:.3f}"
                      f"[{rec['auc_matched_loss_lo']:.3f},"
                      f"{rec['auc_matched_loss_hi']:.3f}] "
                      f"AUCm(out)={rec['auc_matched_output']:.3f} "
                      f"mask={rec['masking_loss']:+.3f} "
                      f"eps={rec['eps']:.4f}", flush=True)
    return rows
