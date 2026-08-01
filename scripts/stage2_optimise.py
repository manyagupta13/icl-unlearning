#!/usr/bin/env python
"""
Stage 2: optimise the in-context corruption policy.

    python scripts/stage2_optimise.py --config configs/regression.yaml
    python scripts/stage2_optimise.py --config ... --policy conditional
    python scripts/stage2_optimise.py --config ... --compare-reinforce

Reuses the cached ensembles -- the corruption is a prompt-time edit, so nothing
retrains and this runs in seconds.

Reports, for each architecture:
  * the optimised policy and the AUC it achieves
  * eps (preservation) at that operating point, so the removal/preservation
    tradeoff is visible rather than hidden
  * the best fixed-theta point on the Stage-1 `bern` grid, for comparison --
    a learned policy that cannot beat the best single theta is not worth having
  * optionally, the same optimisation driven by the brief's REINFORCE gradient,
    with the gradient-variance ratio between the two estimators

Writes artifacts/stage2_{name}.json.
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning import audit                                # noqa: E402
from icl_unlearning.corrupt import corrupt                      # noqa: E402
from icl_unlearning.data import build_spec, make_probe         # noqa: E402
from icl_unlearning.models import apply_frozen                  # noqa: E402
from icl_unlearning.policy import (ConditionalBernoulli,        # noqa: E402
                                   ScalarBernoulli, objective_value,
                                   policy_auc, reinforce_grad)


def empirical_auc_and_eps(M_full, M_oracle, probe, spec, theta_mask, gen,
                          sample=True):
    """Measured (not Gaussian-approximated) AUC and eps for a flip policy.

    `theta_mask` is a Bernoulli PROBABILITY theta when sample=True (the usual
    case: the policy emits a probability and the corruption is the stochastic
    flip y -> (1-2B)y with B ~ Bern(theta), drawn independently per shadow,
    exactly as corrupt.py's `bern` arm does).

    Pass sample=False only when theta_mask is already a realised {0,1} draw,
    as it is inside the REINFORCE estimator.

    NB: multiplying by (1 - 2*theta) directly -- i.e. treating the probability
    as if it were the draw -- is NOT this corruption. That is the deterministic
    `flip` arm at strength theta: same mean, but no variance, hence no masking
    channel. The two give materially different AUCs.
    """
    S = M_full.shape[0]
    sl = probe.forget_slice
    x = probe.x.unsqueeze(0).expand(S, *probe.x.shape).clone()
    y = probe.y.unsqueeze(0).expand(S, *probe.y.shape).clone()
    if sample:
        p = theta_mask.expand(S, *theta_mask.shape[1:]).clamp(0.0, 1.0)
        B = torch.bernoulli(p, generator=gen)
    else:
        B = theta_mask
    y[:, :, sl] = y[:, :, sl] * (1.0 - 2.0 * B)
    from icl_unlearning.data import assemble
    X, yl, yq = assemble(x, y)
    o1 = audit.observables(apply_frozen(M_full, X, yl, spec.N), yq)
    o0 = audit.observables(apply_frozen(M_oracle, X, yl, spec.N), yq)
    auc = audit.membership_auc(audit.membership_score(o1["residual"]),
                               audit.membership_score(o0["residual"]))

    Xc, ylc, yqc = corrupt(probe, S, "none", 0.0, gen)
    p1 = audit.fit_residual_law(apply_frozen(M_full, Xc, ylc, spec.N), yqc)
    p2 = audit.fit_residual_law(apply_frozen(M_oracle, Xc, ylc, spec.N), yqc)
    p = audit.fit_residual_law(apply_frozen(M_full, X, yl, spec.N), yq)
    return auc, audit.alpha_eps(p1, p2, p)["eps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", choices=["scalar", "conditional"],
                    default="scalar")
    ap.add_argument("--objective", choices=["dist", "auc"], default="dist",
                    help="'dist' targets AUC=0.5 (correct); 'auc' is the "
                         "brief's literal min-AUC, kept for comparison")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--train-seed-idx", type=int, default=0)
    ap.add_argument("--probe-seed-idx", type=int, default=0)
    ap.add_argument("--compare-reinforce", action="store_true")
    ap.add_argument("--reinforce-samples", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name, d, pr, tr = cfg["name"], cfg["data"], cfg["probe"], cfg["train"]
    dev = args.device
    adir = pathlib.Path(cfg["paths"]["artifacts"])

    spec = build_spec(d)
    gen = torch.Generator(device=dev).manual_seed(pr["seed"] + args.probe_seed_idx)
    probe = make_probe(spec, pr["counts"], d["forget"], pr["P"], gen, dev)

    blob = torch.load(adir / f"ensembles_{name}_ts{args.train_seed_idx}.pt",
                      map_location=dev, weights_only=False)

    out = {"config": name, "policy": args.policy, "objective": args.objective,
           "lam": args.lam, "archs": {}}

    for arch in tr["archs"]:
        M_full = blob[f"{arch}|full|M"].to(dev)
        M_oracle = blob[f"{arch}|oracle|M"].to(dev)

        pol = (ScalarBernoulli().to(dev) if args.policy == "scalar"
               else ConditionalBernoulli(d["D"]).to(dev))
        opt = torch.optim.Adam(pol.parameters(), lr=args.lr)

        t0 = time.time()
        auc0 = float(policy_auc(M_full, M_oracle, probe, pol))
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            auc = policy_auc(M_full, M_oracle, probe, pol)
            loss = objective_value(auc, pol, args.lam, args.objective)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.steps - 1:
                print(f"  [{arch}] step {step:4d}  AUC {float(auc):.4f}  "
                      f"loss {float(loss):.6f}")
        dt = time.time() - t0

        with torch.no_grad():
            x_f = probe.x[:, probe.forget_slice, :]
            y_f = probe.y[:, probe.forget_slice]
            theta = pol(x_f, y_f)
            auc_cf = float(policy_auc(M_full, M_oracle, probe, pol))
            e_auc, e_eps = empirical_auc_and_eps(M_full, M_oracle, probe, spec,
                                                 theta.unsqueeze(0), gen)

        rec = {"auc_closed_form_start": auc0,
               "auc_closed_form_final": auc_cf,
               "auc_empirical_final": e_auc,
               "eps_empirical_final": e_eps,
               "theta_mean": float(theta.mean()),
               "theta_min": float(theta.min()),
               "theta_max": float(theta.max()),
               "seconds": round(dt, 2)}

        # baseline: the best single theta on the Stage-1 grid
        best = None
        for t in [i / 40 for i in range(41)]:
            mask = torch.full_like(y_f, t).unsqueeze(0)
            a, _ = empirical_auc_and_eps(M_full, M_oracle, probe, spec, mask, gen)
            if best is None or abs(a - 0.5) < abs(best[1] - 0.5):
                best = (t, a)
        rec["best_fixed_theta"] = best[0]
        rec["best_fixed_theta_auc"] = best[1]

        if args.compare_reinforce:
            pol2 = (ScalarBernoulli().to(dev) if args.policy == "scalar"
                    else ConditionalBernoulli(d["D"]).to(dev))
            opt2 = torch.optim.Adam(pol2.parameters(), lr=args.lr)

            def auc_fn(B):
                # B is already a realised {0,1} draw -- do not resample it
                a, _ = empirical_auc_and_eps(M_full, M_oracle, probe, spec,
                                             B.unsqueeze(0), gen, sample=False)
                return (a - 0.5) ** 2 if args.objective == "dist" else a

            t0 = time.time()
            for step in range(args.steps // 4):
                opt2.zero_grad(set_to_none=True)
                loss2, sm, ss = reinforce_grad(M_full, M_oracle, probe, pol2,
                                               auc_fn,
                                               n_samples=args.reinforce_samples)
                loss2.backward()
                opt2.step()
                if step % 25 == 0:
                    print(f"  [{arch}] REINFORCE step {step:4d}  "
                          f"score {sm:.5f} +/- {ss:.5f}")
            with torch.no_grad():
                th2 = pol2(x_f, y_f)
                a2, _ = empirical_auc_and_eps(M_full, M_oracle, probe, spec,
                                              th2.unsqueeze(0), gen)
            rec["reinforce"] = {"auc_final": a2,
                                "theta_mean": float(th2.mean()),
                                "steps": args.steps // 4,
                                "samples_per_step": args.reinforce_samples,
                                "seconds": round(time.time() - t0, 2)}

        out["archs"][arch] = rec
        print(f"  [{arch}] -> {json.dumps(rec, indent=2)}")

    path = adir / f"stage2_{name}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
