#!/usr/bin/env python
"""
Control experiment: does the classic forget-vs-retain loss attack measure
MEMBERSHIP, or just group difficulty?

    python scripts/oracle_control_attack.py --config configs/regression.yaml

Why this exists
---------------
The standard loss-threshold membership attack takes ONE model, scores
forget-group examples and retain-group examples, and computes AUC over the
EXAMPLE axis. Run that here and it reports a large effect. But in this setup
the groups differ by construction in covariance spectrum:

    z1 PR=1.90   z2 PR=2.38   z3 PR=3.33   (z3 = forget, flattest = hardest)

so z3 examples have higher in-context error than z1/z2 examples REGARDLESS of
whether the model was trained on z3. The attack can therefore score far from
0.5 with zero membership information in play.

This script runs the identical attack against both ensembles:

    full    trained on z1,z2,z3   -> z3 IS a member
    oracle  trained on z1,z2 only -> z3 is NOT a member, never seen

If the oracle scores about the same as the full model, the attack is reading
spectral difficulty, not membership, and any AUC near 1 from that design is an
artefact. That is the control the shadow/retrain-oracle design in sweep.py
exists to provide.

Note this AUC is a different estimand from the one in run_auc_sweep.py:
  here    AUC over the EXAMPLE axis, one model  ("which examples look hard?")
  sweep   AUC over the MODEL axis, one example  ("which ensemble made this?")
Only the second is a membership test.
"""
import argparse
import pathlib
import sys

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning import audit                            # noqa: E402
from icl_unlearning.data import build_spec, make_sequences  # noqa: E402
from icl_unlearning.models import apply_frozen              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-eval", type=int, default=512,
                    help="in-context tasks per group, per shadow")
    ap.add_argument("--train-seed-idx", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name, d, tr = cfg["name"], cfg["data"], cfg["train"]
    dev = args.device
    forget = d["forget"]
    retain = [g for g in d["groups"] if g != forget]

    spec = build_spec(d)
    print(f"forget={forget} (PR={spec.pr(forget):.2f})   "
          f"retain={retain} (PR=" +
          ", ".join(f"{spec.pr(g):.2f}" for g in retain) + ")")

    adir = pathlib.Path(cfg["paths"]["artifacts"])
    blob = torch.load(adir / f"ensembles_{name}_ts{args.train_seed_idx}.pt",
                      map_location=dev, weights_only=False)

    print(f"\n{'arch':8s} {'trained on':16s} {'z3 a member?':13s} "
          f"{'AUC(forget vs retain)':>22s} {'loss(z3)':>10s} {'loss(retain)':>13s}")
    print("-" * 92)

    results = {}
    for arch in tr["archs"]:
        for hyp in ("full", "oracle"):
            M = blob[f"{arch}|{hyp}|M"].to(dev)
            S = M.shape[0]
            gen = torch.Generator(device=dev).manual_seed(4242)

            Xf, ylf, yqf = make_sequences(spec, [forget], S, args.n_eval,
                                          gen, dev)
            loss_f = (apply_frozen(M, Xf, ylf, spec.N) - yqf) ** 2   # [S, n]

            Xr, ylr, yqr = make_sequences(spec, retain, S, args.n_eval,
                                          gen, dev)
            loss_r = (apply_frozen(M, Xr, ylr, spec.N) - yqr) ** 2   # [S, n]

            # AUC over the EXAMPLE axis, one value per shadow. auc_per_probe
            # ranks along dim 0, so transpose to put examples there.
            aucs = audit.auc_per_probe(loss_f.T.contiguous(), loss_r.T.contiguous())
            mean, std = float(aucs.mean()), float(aucs.std())

            member = "YES" if hyp == "full" else "no (never seen)"
            trained = "z1,z2,z3" if hyp == "full" else "z1,z2 only"
            print(f"{arch:8s} {trained:16s} {member:13s} "
                  f"{mean:15.4f} +- {std:.4f} {float(loss_f.mean()):10.4f} "
                  f"{float(loss_r.mean()):13.4f}")
            results[(arch, hyp)] = mean

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    for arch in tr["archs"]:
        a_full, a_orc = results[(arch, "full")], results[(arch, "oracle")]
        gap = abs(a_full - a_orc)
        # how much of the full model's deviation from chance is NOT explained
        # by the oracle's deviation from chance?
        dev_full, dev_orc = abs(a_full - 0.5), abs(a_orc - 0.5)
        explained = 100.0 * min(dev_orc, dev_full) / max(dev_full, 1e-12)
        print(f"  {arch}: full={a_full:.4f}  oracle={a_orc:.4f}  |gap|={gap:.4f}")
        print(f"    {explained:.1f}% of the full model's deviation from 0.5 is "
              f"reproduced by a model that NEVER saw {forget}.")
        if gap < 0.05:
            print(f"    -> the attack is measuring GROUP DIFFICULTY, not "
                  f"membership. An AUC near 1 from this design is an artefact.")
        else:
            print(f"    -> a real membership component survives the control "
                  f"(gap {gap:.4f}); worth reporting alongside the oracle value.")
    print()


if __name__ == "__main__":
    main()
