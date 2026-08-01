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

from . import audit, theory
from .corrupt import corrupt
from .data import MixtureSpec, Probe, assemble
from .models import apply_frozen


@torch.no_grad()
def sweep_point(spec: MixtureSpec, probe: Probe, M_full: torch.Tensor,
                M_oracle: torch.Tensor, mode: str, param: float,
                gen: torch.Generator, retain_x=None, n_boot: int = 200,
                boot_seed: int = 0, n_shared_reps: int = 16,
                mmd_max_n: int = 2000) -> dict:
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
    #
    # This MUST be averaged over several independent shared draws. With a single
    # draw every shadow sees the identical corruption, so the resulting AUC is a
    # function of that one draw and its sampling variance is enormous -- it comes
    # out as a jagged line that swings 0.4-0.9 between adjacent grid points and
    # is unreadable. `n_shared_reps` independent draws, averaged, fixes it.
    # `bern` belongs here and `flip` does not: the Bernoulli flip draws fresh
    # randomness per shadow, so it has a variance channel and the masking
    # control is meaningful for it. `flip` is deterministic given the probe.
    stochastic = mode in ("C1", "C2", "C3", "bern")
    shared_acc = {}
    if stochastic:
        for _ in range(n_shared_reps):
            Xs, yls, yqs = corrupt(probe, S, mode, param, gen, retain_x=retain_x,
                                   shared_noise=True)
            o1s = audit.observables(apply_frozen(M_full, Xs, yls, spec.N), yqs)
            o0s = audit.observables(apply_frozen(M_oracle, Xs, yls, spec.N), yqs)
            for nm in o1s:
                shared_acc.setdefault(nm, []).append(
                    audit.membership_auc(audit.membership_score(o1s[nm]),
                                         audit.membership_score(o0s[nm])))

    obs1 = audit.observables(yhat1, yq1)
    obs0 = audit.observables(yhat0, yq0)
    obs0m = audit.observables(yhat0m, yq1)
    raw1 = audit.observables_raw(yhat1, yq1)
    raw0m = audit.observables_raw(yhat0m, yq1)

    rec = {"mode": mode, "param": float(param)}
    for name in ("loss", "residual"):
        a = audit.membership_auc(audit.membership_score(obs1[name]),
                                 audit.membership_score(obs0[name]))
        rec[f"auc_{name}"] = a

        am, lo, hi = audit.membership_auc_ci(audit.membership_score(obs1[name]),
                                             audit.membership_score(obs0m[name]),
                                             n_boot=n_boot, seed=boot_seed)
        rec[f"auc_matched_{name}"] = am
        rec[f"auc_matched_{name}_lo"] = lo
        rec[f"auc_matched_{name}_hi"] = hi
        # correct aggregation order: symmetrise per probe point, then average
        rec[f"auc_matched_{name}_sym"] = audit.symmetrised_auc_per_probe(
            audit.membership_score(obs1[name]), audit.membership_score(obs0m[name]))

        a_sh = (sum(shared_acc[name]) / len(shared_acc[name])
                if stochastic else am)
        rec[f"auc_shared_{name}"] = a_sh
        # positive => part of the AUC drop is variance masking, not removal
        rec[f"masking_{name}"] = am - a_sh

        # Both spreads must come from the SAME context, or the masking
        # diagnostic is meaningless: obs1 is scored on the corrupted probe, so
        # obs0m (oracle, same corrupted probe) is the like-for-like companion.
        # This previously used obs0 (clean context), which made spread_h1 vs
        # spread_h0 an apples-to-oranges comparison that would show a spurious
        # gap growing with corruption strength.
        rec[f"spread_h1_{name}"] = audit.spread(obs1[name])
        rec[f"spread_h0_{name}"] = audit.spread(obs0m[name])
        rec[f"spread_h0_clean_{name}"] = audit.spread(obs0[name])

    # the un-aligned output observable, kept ONLY to quantify sign cancellation:
    # this is the quantity that pins to ~0.5 regardless of the true signal
    rec["auc_matched_output_unaligned"] = audit.membership_auc(
        audit.membership_score(raw1["output"]), audit.membership_score(raw0m["output"]))

    # Closed-form prediction, independent of everything measured above. Where
    # this tracks auc_matched_residual, the pipeline and the algebra agree;
    # a systematic gap means one of them is wrong. C3/flip/whiten have no
    # derived form (see theory.py), so they get NaN rather than a guess.
    rec["auc_theory_residual"] = (
        theory.predicted_auc(M_full, M_oracle, probe, mode, param)
        if mode in ("C1", "C2", "bern") else float("nan"))

    # Distributional criterion on the forget-population residual law.
    #
    # NOTE, deliberate and different from the AUC convention above: p2 uses the
    # CLEAN-context oracle (yhat0), not the matched-context one. These measure
    # different things and the asymmetry is intentional.
    #   - AUC asks "can an attacker distinguish?", so both hypotheses must see
    #     the same prompt or the attacker wins on edit-detection alone.
    #   - eps asks "how close is the unlearned model to the retrain ideal?".
    #     The retrained model is the target and does not need the edit -- in
    #     deployment you would query it with a normal prompt. So the honest
    #     comparison is edited-full vs un-edited-oracle.
    # If you disagree with that reading, swap yhat0 -> yhat0m here and re-run;
    # it is a one-line change, but state which convention you used.
    p = audit.fit_residual_law(yhat1, yq1)
    p2 = audit.fit_residual_law(yhat0, yq0)
    X_c, yl_c, yq_c = corrupt(probe, S, "none", 0.0, gen)
    p1 = audit.fit_residual_law(apply_frozen(M_full, X_c, yl_c, spec.N), yq_c)
    rec.update(audit.alpha_eps(p1, p2, p))
    rec["mmd2_to_oracle"] = audit.mmd2((yhat1 - yq1).reshape(-1),
                                       (yhat0 - yq0).reshape(-1),
                                       max_n=mmd_max_n, seed=boot_seed)
    return rec


@torch.no_grad()
def run_sweep(spec: MixtureSpec, probe: Probe, ensembles: dict, grids: dict,
              seed: int = 0, device="cuda", retain_x=None,
              n_boot: int = 200, n_shared_reps: int = 16,
              mmd_max_n: int = 2000) -> list[dict]:
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
                                  boot_seed=seed + 1000 * k,
                                  n_shared_reps=n_shared_reps,
                                  mmd_max_n=mmd_max_n)
                rec["arch"] = arch
                rows.append(rec)
                print(f"  {arch:7s} {mode:7s} p={prm:<8.4g} "
                      f"AUCm(loss)={rec['auc_matched_loss']:.3f}"
                      f"[{rec['auc_matched_loss_lo']:.3f},"
                      f"{rec['auc_matched_loss_hi']:.3f}] "
                      f"AUCm(res)={rec['auc_matched_residual']:.3f} "
                      f"mask={rec['masking_loss']:+.3f} "
                      f"eps={rec['eps']:.4f}", flush=True)
    return rows
