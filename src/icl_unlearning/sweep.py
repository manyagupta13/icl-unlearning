"""
Sweep orchestration.

The frozen ensembles are loaded once. For each (arch, mode, strength) we
re-corrupt the probe, run frozen forward passes, and score both observables.
No training happens in here -- that is the whole point.

H0 (oracle) is evaluated on the CLEAN probe: the oracle never saw the forget
group, so there is nothing to edit. H1 is evaluated on the corrupted probe.

Every AUC ships with a bootstrap CI and a gap/spread effect size, and every row
carries the NO-EDIT BASELINE AUC for its architecture. The baseline is the
control (fully-trained vs oracle, no edit applied) and it is the reference every
swept AUC has to be read against: a curve at 0.5 means removal only if the
baseline was near 1.0. It is computed ONCE PER ARCH -- it does not depend on the
corruption mode or strength, so recomputing it per grid point would be both
wasteful and a source of spurious grid-to-grid jitter.
"""
from __future__ import annotations

import torch

from . import audit, diagnostics
from .corrupt import corrupt
from .data import MixtureSpec, Probe, assemble
from .models import apply_frozen

OBSERVABLES = ("loss", "output")


def clean_forward(spec: MixtureSpec, probe: Probe, M: torch.Tensor):
    """Frozen forward on the unedited probe. -> (yhat, yq)"""
    S = M.shape[0]
    x = probe.x.unsqueeze(0).expand(S, *probe.x.shape)
    y = probe.y.unsqueeze(0).expand(S, *probe.y.shape)
    X, yl, yq = assemble(x, y)
    return apply_frozen(M, X, yl, spec.N), yq


@torch.no_grad()
def arch_baseline(spec: MixtureSpec, probe: Probe, M_full: torch.Tensor,
                  M_oracle: torch.Tensor, n_boot: int = 300,
                  seed: int = 0) -> dict:
    """
    The no-edit control for one architecture: fully-trained vs retrain oracle,
    both on the CLEAN probe. This is the ceiling the sweep is measured against.

    Returns a dict of `baseline_*` columns plus the cached residual laws that
    `sweep_point` needs (p1 = clean fully-trained law), so those are not
    refitted at every grid point either.
    """
    yhat1, yq1 = clean_forward(spec, probe, M_full)
    yhat0, yq0 = clean_forward(spec, probe, M_oracle)

    obs1 = audit.observables(yhat1, yq1)
    obs0 = audit.observables(yhat0, yq0)

    out = {}
    for name in OBSERVABLES:
        ci = diagnostics.bootstrap_auc_ci(obs1[name], obs0[name],
                                          n_boot=n_boot, seed=seed)
        out[f"baseline_auc_{name}"] = ci["auc"]
        out[f"baseline_auc_{name}_lo"] = ci["lo"]
        out[f"baseline_auc_{name}_hi"] = ci["hi"]
        out[f"baseline_auc_{name}_sym"] = audit.symmetrised_auc(ci["auc"])
        out[f"baseline_gapspread_{name}"] = diagnostics.gap_to_spread(
            obs1[name], obs0[name])

    out.update(diagnostics.ensemble_separation(M_full, M_oracle))
    out = {("weight_" + k if k in ("between", "within", "ratio") else k): v
           for k, v in out.items()}

    # residual laws of the two unedited arms; p1 feeds alpha = KL(p1 || p)
    out["_p1"] = audit.fit_residual_law(yhat1, yq1)
    out["_p2_clean"] = audit.fit_residual_law(yhat0, yq0)
    return out


@torch.no_grad()
def sweep_point(spec: MixtureSpec, probe: Probe, M_full: torch.Tensor,
                M_oracle: torch.Tensor, mode: str, param: float,
                gen: torch.Generator, baseline: dict, retain_x=None,
                n_boot: int = 300, boot_seed: int = 0) -> dict:
    """One grid point. Returns a flat record ready for a dataframe row."""
    S = M_full.shape[0]

    # H1: fully trained + in-context edit
    X1, yl1, yq1 = corrupt(probe, S, mode, param, gen, retain_x=retain_x)
    yhat1 = apply_frozen(M_full, X1, yl1, spec.N)

    # H0: retrain oracle on the clean probe
    yhat0, yq0 = clean_forward(spec, probe, M_oracle)

    obs1 = audit.observables(yhat1, yq1)
    obs0 = audit.observables(yhat0, yq0)

    rec = {"mode": mode, "param": float(param)}
    for name in OBSERVABLES:
        ci = diagnostics.bootstrap_auc_ci(obs1[name], obs0[name],
                                          n_boot=n_boot, seed=boot_seed)
        rec[f"auc_{name}"] = ci["auc"]
        rec[f"auc_{name}_lo"] = ci["lo"]
        rec[f"auc_{name}_hi"] = ci["hi"]
        rec[f"auc_{name}_sym"] = audit.symmetrised_auc(ci["auc"])
        # why the AUC is what it is: numerator = real separation,
        # denominator = ensemble variance a stochastic edit can inflate for free
        rec[f"gapspread_{name}"] = diagnostics.gap_to_spread(obs1[name],
                                                             obs0[name])
        rec[f"spread_h1_{name}"] = audit.spread(obs1[name])
        rec[f"spread_h0_{name}"] = audit.spread(obs0[name])

    # distributional criterion on the forget-population residual law.
    # p1 (clean fully-trained) is arch-constant and comes from the baseline.
    p = audit.fit_residual_law(yhat1, yq1)
    p2 = audit.fit_residual_law(yhat0, yq0)
    rec.update(audit.alpha_eps(baseline["_p1"], p2, p))
    rec["mmd2_to_oracle"] = audit.mmd2((yhat1 - yq1).reshape(-1),
                                       (yhat0 - yq0).reshape(-1))

    # arch-level control, carried on every row so a plot never has to join
    rec.update({k: v for k, v in baseline.items() if not k.startswith("_")})
    return rec


@torch.no_grad()
def run_sweep(spec: MixtureSpec, probe: Probe, ensembles: dict, grids: dict,
              seed: int = 0, device="cuda", retain_x=None,
              n_boot: int = 300) -> list[dict]:
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

        base = arch_baseline(spec, probe, M_full, M_oracle,
                             n_boot=n_boot, seed=seed)
        b_out = base["baseline_auc_output"]
        print(f"[{arch}] no-edit baseline: AUC(out)={b_out:.3f} "
              f"[{base['baseline_auc_output_lo']:.3f}, "
              f"{base['baseline_auc_output_hi']:.3f}]  "
              f"AUC(loss)={base['baseline_auc_loss']:.3f}  "
              f"gap/spread={base['baseline_gapspread_output']:.2f}  "
              f"weight ratio={base['weight_ratio']:.2f}", flush=True)
        if audit.symmetrised_auc(b_out) < 0.9:
            print(f"[{arch}] WARNING: dead baseline -- the swept AUCs below "
                  f"are not interpretable. Run scripts/diagnose.py.", flush=True)

        for mode, params in grids.items():
            for prm in params:
                rec = sweep_point(spec, probe, M_full, M_oracle, mode, prm,
                                  gen, base, retain_x=retain_x,
                                  n_boot=n_boot, boot_seed=seed)
                rec["arch"] = arch
                rows.append(rec)
                print(f"  {arch:7s} {mode:7s} p={prm:<8.4g} "
                      f"AUC(loss)={rec['auc_loss']:.3f} "
                      f"AUC(out)={rec['auc_output']:.3f} "
                      f"[{rec['auc_output_lo']:.3f},{rec['auc_output_hi']:.3f}] "
                      f"d={rec['gapspread_output']:.2f} "
                      f"eps={rec['eps']:.4f}", flush=True)
    return rows
