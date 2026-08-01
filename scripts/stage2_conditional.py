#!/usr/bin/env python
"""
Does a per-token corruption policy beat a single flip probability?

    python scripts/stage2_conditional.py --config configs/regression.yaml
    python scripts/stage2_conditional.py --config configs/mnist.yaml

Stage 2 as reported in the paper learns ONE flip probability theta for the whole
forget group. `policy.ConditionalBernoulli` learns one per token,
theta_i = sigmoid(MLP([x_i ; y_i])), but was never run. This script runs it and
asks the only two questions that make the answer worth anything:

  1. At MATCHED preservation cost, does it reach a lower |AUC - 1/2|?
     Comparing the two at their own separately-chosen operating points would be
     meaningless -- a policy that corrupts harder will always look better on AUC
     alone. So both are swept over a shared budget penalty lambda, tracing an
     (eps, AUC) frontier each, and the frontiers are compared at common eps.

  2. If it wins, is it winning for the reason the moment argument predicts?
     `diagnose.py` measures whether theta_i concentrates on the tokens with the
     largest lever on the hypothesis gap. A win with no measured targeting means
     the mechanism story is wrong even though the number improved.

A third curve is included as the floor: the best fixed theta on the Stage-1
`bern` grid, i.e. what you get with no learning at all. A learned scalar policy
that cannot beat the grid is not worth having, and a conditional policy that
cannot beat the learned scalar is not worth having either.

Writes artifacts/stage2_conditional_{name}.json.
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from icl_unlearning import diagnose                             # noqa: E402
from icl_unlearning.data import build_spec, make_probe          # noqa: E402
from icl_unlearning.policy import (ConditionalBernoulli,        # noqa: E402
                                   ScalarBernoulli, policy_auc)

# Reuse the measurement from stage2_optimise rather than reimplementing it. The
# comparison is only meaningful if both scripts measure AUC and eps the same
# way, and a second copy of this function is a second thing to keep in sync.
from stage2_optimise import empirical_auc_and_eps               # noqa: E402


def _fit(policy, probe, M_full, M_oracle, lam, steps, lr):
    """Train one policy at one lambda. Returns the fitted policy."""
    x_f = probe.x[:, probe.forget_slice, :]
    y_f = probe.y[:, probe.forget_slice]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        auc = policy_auc(M_full, M_oracle, probe, policy)
        loss = (auc - 0.5) ** 2 + lam * policy.budget(x_f, y_f)
        loss.backward()
        opt.step()
    return policy


def _evaluate(policy, probe, spec, M_full, M_oracle, lever, gen):
    """Closed-form AUC, measured AUC, eps, and the targeting diagnostics."""
    x_f = probe.x[:, probe.forget_slice, :]
    y_f = probe.y[:, probe.forget_slice]
    with torch.no_grad():
        theta = policy(x_f, y_f)
        auc_cf = float(policy_auc(M_full, M_oracle, probe, policy))
        auc_emp, eps = empirical_auc_and_eps(M_full, M_oracle, probe, spec,
                                             theta.unsqueeze(0), gen)
    rec = {"auc_closed_form": auc_cf, "auc_empirical": auc_emp, "eps": eps,
           "dist_from_chance": abs(auc_emp - 0.5)}
    rec.update(diagnose.describe_policy(theta, lever))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--restarts", type=int, default=3,
                    help="random restarts for the conditional MLP; the scalar "
                         "policy is one-dimensional and does not need them, but "
                         "an MLP that lands in a bad basin would otherwise be "
                         "reported as evidence against conditioning")
    ap.add_argument("--train-seed-idx", type=int, default=0)
    ap.add_argument("--probe-seed-idx", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-suffix", default="")
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

    out = {"config": name, "lambdas": args.lambdas, "steps": args.steps,
           "restarts": args.restarts, "hidden": args.hidden,
           "train_seed_idx": args.train_seed_idx,
           "probe_seed_idx": args.probe_seed_idx, "archs": {}}

    for arch in tr["archs"]:
        print(f"\n=== {arch} ===")
        M_full = blob[f"{arch}|full|M"].to(dev)
        M_oracle = blob[f"{arch}|oracle|M"].to(dev)
        lever = diagnose.token_lever(M_full, M_oracle, probe)

        x_f = probe.x[:, probe.forget_slice, :]
        y_f = probe.y[:, probe.forget_slice]

        rec = {"lever_gini": float(diagnose._gini(lever.abs())),
               "scalar": [], "conditional": [], "fixed_grid": []}

        # ---- floor: no learning at all, just a fixed theta on the Stage-1 grid
        t0 = time.time()
        for t in [i / 20 for i in range(21)]:
            mask = torch.full_like(y_f, t)
            a, e = empirical_auc_and_eps(M_full, M_oracle, probe, spec,
                                         mask.unsqueeze(0), gen)
            rec["fixed_grid"].append(
                {"theta": t, "auc_empirical": a, "eps": e,
                 "dist_from_chance": abs(a - 0.5)})
        print(f"  fixed-theta grid: {time.time() - t0:.1f}s")

        # ---- the two learned policies, on a shared budget penalty
        for lam in args.lambdas:
            pol = ScalarBernoulli().to(dev)
            _fit(pol, probe, M_full, M_oracle, lam, args.steps, args.lr)
            r = _evaluate(pol, probe, spec, M_full, M_oracle, lever, gen)
            r["lam"] = lam
            rec["scalar"].append(r)

            # restarts: keep the run that got closest to chance
            best = None
            for k in range(args.restarts):
                torch.manual_seed(1000 * args.train_seed_idx + 10 * k + 7)
                pol = ConditionalBernoulli(d["D"], hidden=args.hidden).to(dev)
                _fit(pol, probe, M_full, M_oracle, lam, args.steps, args.lr)
                r2 = _evaluate(pol, probe, spec, M_full, M_oracle, lever, gen)
                r2["lam"] = lam
                r2["restart"] = k
                if best is None or r2["dist_from_chance"] < best["dist_from_chance"]:
                    best = r2
            best["n_restarts"] = args.restarts
            rec["conditional"].append(best)

            print(f"  lam={lam:<6g} scalar AUC {rec['scalar'][-1]['auc_empirical']:.4f} "
                  f"eps {rec['scalar'][-1]['eps']:.3e}   |   "
                  f"cond AUC {best['auc_empirical']:.4f} eps {best['eps']:.3e} "
                  f"T={best['targeting_T']:.3f} R={best['polarisation_R']:.3f}")

        rec["verdict"] = _verdict(rec)
        out["archs"][arch] = rec
        print(f"  verdict: {json.dumps(rec['verdict'], indent=4)}")

    path = adir / f"stage2_conditional_{name}{args.out_suffix}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwritten -> {path}")


def _verdict(rec: dict) -> dict:
    """
    Reduce the frontiers to the comparison that actually answers the question.

    `eps_at_best` is the preservation cost each family pays at ITS OWN closest
    approach to chance. That is the honest headline: a policy is better if it
    gets as close to chance for less eps, or closer to chance for the same eps.

    `cond_beats_scalar_at_matched_eps` is the stricter test. For every scalar
    operating point, find the conditional point with eps no larger, and ask
    whether it lands closer to chance. Reported as the fraction of scalar points
    that are dominated. Anything below ~0.5 means the frontiers cross and no
    clean claim can be made either way.
    """
    def best(rows):
        return min(rows, key=lambda r: r["dist_from_chance"])

    b_s, b_c, b_g = (best(rec["scalar"]), best(rec["conditional"]),
                     best(rec["fixed_grid"]))

    dominated, total = 0, 0
    for s in rec["scalar"]:
        cands = [c for c in rec["conditional"] if c["eps"] <= s["eps"] * 1.01]
        if not cands:
            continue
        total += 1
        if min(c["dist_from_chance"] for c in cands) < s["dist_from_chance"]:
            dominated += 1

    return {
        "best_auc_fixed_grid": b_g["auc_empirical"],
        "best_auc_scalar": b_s["auc_empirical"],
        "best_auc_conditional": b_c["auc_empirical"],
        "eps_at_best_scalar": b_s["eps"],
        "eps_at_best_conditional": b_c["eps"],
        "eps_ratio_cond_over_scalar": (b_c["eps"] / b_s["eps"]
                                       if b_s["eps"] else None),
        "cond_beats_scalar_at_matched_eps": (dominated / total if total else None),
        "targeting_T_at_best_conditional": b_c["targeting_T"],
        "polarisation_R_at_best_conditional": b_c["polarisation_R"],
        "spearman_at_best_conditional": b_c["spearman_theta_abs_lever"],
    }


if __name__ == "__main__":
    main()
