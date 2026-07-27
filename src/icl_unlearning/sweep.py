"""
Sweep orchestration.

The frozen ensembles are loaded once. For each (arch, mode, strength) we
re-corrupt the probe, run frozen forward passes, and score both observables.
No training happens in here -- that is the whole point.

H0 (oracle) is evaluated on the CLEAN probe: the oracle never saw the forget
group, so there is nothing to edit. H1 is evaluated on the corrupted probe.
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
                gen: torch.Generator, retain_x=None) -> dict:
    """One grid point. Returns a flat record ready for a dataframe row."""
    S = M_full.shape[0]

    # H1: fully trained + in-context edit
    X1, yl1, yq1 = corrupt(probe, S, mode, param, gen, retain_x=retain_x)
    yhat1 = apply_frozen(M_full, X1, yl1, spec.N)

    # H0: retrain oracle on the clean probe
    x0 = probe.x.unsqueeze(0).expand(S, *probe.x.shape)
    y0 = probe.y.unsqueeze(0).expand(S, *probe.y.shape)
    X0, yl0, yq0 = assemble(x0, y0)
    yhat0 = apply_frozen(M_oracle, X0, yl0, spec.N)

    obs1 = audit.observables(yhat1, yq1)
    obs0 = audit.observables(yhat0, yq0)

    rec = {"mode": mode, "param": float(param)}
    for name in ("loss", "output"):
        a = audit.membership_auc(obs1[name], obs0[name])
        rec[f"auc_{name}"] = a
        rec[f"auc_{name}_sym"] = audit.symmetrised_auc(a)
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
              seed: int = 0, device="cuda", retain_x=None) -> list[dict]:
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
            for prm in params:
                rec = sweep_point(spec, probe, M_full, M_oracle, mode, prm,
                                  gen, retain_x=retain_x)
                rec["arch"] = arch
                rows.append(rec)
                print(f"  {arch:7s} {mode:7s} p={prm:<8.4g} "
                      f"AUC(loss)={rec['auc_loss']:.3f} "
                      f"AUC(out)={rec['auc_output']:.3f} "
                      f"eps={rec['eps']:.4f}", flush=True)
    return rows
