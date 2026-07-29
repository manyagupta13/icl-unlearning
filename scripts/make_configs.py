#!/usr/bin/env python
"""
Generate config files for the three follow-up experiments.

    python scripts/make_configs.py --out configs/generated

Everything is driven by the parametric spectrum family in data.py, so the
groups are specified by the quantity that matters -- their participation
ratio -- instead of by hand-written eigenvalue lists that do not survive a
change of D.

Experiments
-----------
rotation   THE CONTROL. All groups share the SAME spectrum and differ only by
           a random rotation (basis: random). Lambda_train is then nearly
           identical for the full and oracle models, so the preconditioner
           mismatch that dominates the baseline AUC largely cancels. Whatever
           AUC survives at sigma^2=0 is much closer to genuine per-example
           memorisation. Run this before claiming the headline result means
           what it appears to mean.

pr         PR SWEEP. Retain groups fixed, forget group's PR swept across a
           range at fixed D. Turns "these three groups behaved differently"
           into a curve of AUC against spectral geometry.

nd         N/D SWEEP. The difficulty of in-context regression is set by the
           ratio of context length to dimension, not by D alone. The shipped
           config sits at N/D = 31/4 ~ 7.75, a comfortable regime. This sweeps
           down toward N/D ~ 2 where the read-out must shrink hard. PR is held
           at a FIXED FRACTION of D so "how anisotropic" means the same thing
           at every D -- otherwise D and spectral geometry are confounded.

Each generated config gets a distinct `name`, so artifacts never collide.
"""
import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning.data import spectrum_for_pr          # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "regression.yaml"


def load_base():
    return yaml.safe_load(open(BASE))


def make(name, D, N, prs, basis, base, forget="z3", counts=None):
    """One config. `prs` maps group name -> target participation ratio."""
    cfg = yaml.safe_load(yaml.safe_dump(base))       # deep copy
    cfg["name"] = name
    cfg["data"]["D"] = D
    cfg["data"]["N"] = N
    cfg["data"]["basis"] = basis
    cfg["data"]["groups"] = list(prs)
    cfg["data"]["forget"] = forget
    cfg["data"]["eigs"] = {g: [round(v, 10) for v in spectrum_for_pr(D, pr)]
                           for g, pr in prs.items()}
    cfg["data"]["_target_pr"] = {g: round(pr, 6) for g, pr in prs.items()}

    if counts is None:
        # split N across groups, remainder to the forget group so its slice
        # stays contiguous and non-empty
        k = len(prs)
        per = N // k
        counts = {g: per for g in prs}
        counts[forget] = N - per * (k - 1)
    assert sum(counts.values()) == N, (counts, N)
    cfg["probe"]["counts"] = counts
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/generated")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_base()
    D0, N0 = 4, 31
    written = []

    # ------------------------------------------------------------- rotation
    # Identical spectra, different random rotations. Two PR levels so the
    # control is checked in both an easy and a hard spectral regime.
    for tag, frac in (("mid", 0.55), ("flat", 0.85)):
        pr = 1.0 + frac * (D0 - 1.0)
        cfg = make(f"rot_{tag}", D0, N0,
                   {"z1": pr, "z2": pr, "z3": pr}, "random", base,
                   counts={"z1": 10, "z2": 10, "z3": 11})
        # matched identity-basis twin: isolates "rotation" from "same spectrum"
        cfg_id = make(f"rot_{tag}_identity", D0, N0,
                      {"z1": pr, "z2": pr, "z3": pr}, "identity", base,
                      counts={"z1": 10, "z2": 10, "z3": 11})
        for c in (cfg, cfg_id):
            p = out / f"{c['name']}.yaml"
            yaml.safe_dump(c, open(p, "w"), sort_keys=False)
            written.append(p)

    # ------------------------------------------------------------------- PR
    # Retain groups pinned; forget group's PR swept. D fixed so the only thing
    # moving is the forget group's spectral geometry.
    for pr3 in (1.45, 1.90, 2.35, 2.80, 3.25, 3.70):
        cfg = make(f"pr_{pr3:.2f}".replace(".", "p"), D0, N0,
                   {"z1": 1.90, "z2": 2.38, "z3": pr3}, "identity", base,
                   counts={"z1": 10, "z2": 10, "z3": 11})
        p = out / f"{cfg['name']}.yaml"
        yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
        written.append(p)

    # ------------------------------------------------------------------ N/D
    # PR held at a fixed FRACTION of D, so anisotropy is comparable across D.
    FRACS = {"z1": 0.30, "z2": 0.46, "z3": 0.78}     # matches D=4 PRs ~1.9/2.4/3.3
    for D, N in ((4, 31), (8, 31), (16, 31), (16, 63), (32, 63), (32, 127)):
        prs = {g: 1.0 + f * (D - 1.0) for g, f in FRACS.items()}
        cfg = make(f"nd_D{D}_N{N}", D, N, prs, "identity", base)
        p = out / f"{cfg['name']}.yaml"
        yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
        written.append(p)

    print(f"wrote {len(written)} configs to {out}/\n")
    for p in written:
        c = yaml.safe_load(open(p))
        d = c["data"]
        prs = c["data"]["_target_pr"]
        print(f"  {c['name']:22s} D={d['D']:3d} N={d['N']:4d} "
              f"N/D={d['N']/d['D']:5.2f} basis={d['basis']:8s} "
              f"PR=" + ",".join(f"{v:.2f}" for v in prs.values()))
    print("\nRun any of them with:")
    print("  python scripts/train_ensembles.py --config <cfg>")
    print("  python scripts/run_auc_sweep.py   --config <cfg>")
    print("  python scripts/plot_auc_vs_var.py --config <cfg>")
    print("\nStart with rot_mid / rot_mid_identity -- that pair is the control")
    print("that decides what the baseline AUC actually measures.")


if __name__ == "__main__":
    main()
