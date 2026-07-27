#!/usr/bin/env python
"""
Signal check. Run this after training and BEFORE sweeping.

    python scripts/diagnose.py --config configs/regression.yaml

The sweep asks "how far does an in-context edit push membership AUC toward
0.5?". That question is only meaningful if the UNEDITED AUC starts near 1.0 --
the fully-trained ensemble and the retrain oracle saw different training data,
so a membership attacker ought to separate them almost perfectly. If the
control already sits at 0.5, every curve in the sweep is noise around a dead
baseline and none of it means anything.

This script loads the cached ensembles and reports, per architecture:

    weight-space separation   ||E[M_full] - E[M_oracle]|| / within-ensemble spread
    baseline AUC + bootstrap CI    both observables, no edit applied
    gap / spread              the standardised effect size driving that AUC
    per-group query MSE       should track participation ratio

then prints a verdict. No training, no edits, no files written.
"""
import argparse
import pathlib
import sys

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning import audit, diagnostics                  # noqa: E402
from icl_unlearning.data import MixtureSpec, make_probe        # noqa: E402
from icl_unlearning.sweep import clean_forward                # noqa: E402
from icl_unlearning.train import per_group_mse                 # noqa: E402

OK_THRESHOLD = 0.9        # baseline AUC(output) above this and we are in business


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    task, d, pr = cfg["task"], cfg["data"], cfg["probe"]
    dev = args.device

    adir = pathlib.Path(cfg["paths"]["artifacts"])
    blob = torch.load(adir / f"ensembles_{task}.pt", map_location=dev,
                      weights_only=False)

    spec = MixtureSpec(names=d["groups"], eigs=d["eigs"], D=d["D"], N=d["N"],
                       basis=d["basis"], seed=d["seed"])
    gen = torch.Generator(device=dev).manual_seed(pr["seed"])
    probe = make_probe(spec, pr["counts"], d["forget"], pr["P"], task, gen, dev)

    print(f"device={dev}  task={task}  D={spec.D}  N={spec.N}  P={probe.P}")
    print(f"forget group = {d['forget']}")
    print("participation ratios: " +
          "  ".join(f"{g}={spec.pr(g):.2f}" for g in spec.names) +
          f"   (max possible = {spec.D})")

    verdicts = {}
    for arch in cfg["train"]["archs"]:
        M_full = blob[f"{arch}|full|M"].to(dev)
        M_oracle = blob[f"{arch}|oracle|M"].to(dev)
        S = M_full.shape[0]

        print(f"\n{'='*66}\n{arch}   ({S} shadows per hypothesis)\n{'='*66}")

        # ------------------------------------------------ weight space
        sep = diagnostics.ensemble_separation(M_full, M_oracle)
        print("weight space")
        print(f"  between-ensemble  ||E[M_full] - E[M_oracle]||_F = {sep['between']:.4g}")
        print(f"  within-ensemble   spread                        = {sep['within']:.4g}")
        print(f"  ratio                                           = {sep['ratio']:.3f}")

        # ------------------------------------------------ output space
        yhat1, yq1 = clean_forward(spec, probe, M_full)
        yhat0, yq0 = clean_forward(spec, probe, M_oracle)
        obs1 = audit.observables(yhat1, yq1)
        obs0 = audit.observables(yhat0, yq0)

        print(f"baseline membership audit (no edit, {args.n_boot} bootstrap reps)")
        cis = {}
        for name in ("loss", "output"):
            ci = diagnostics.bootstrap_auc_ci(obs1[name], obs0[name],
                                              n_boot=args.n_boot, seed=0)
            gs = diagnostics.gap_to_spread(obs1[name], obs0[name])
            cis[name] = ci
            print(f"  AUC({name:6s}) = {ci['auc']:.4f}  "
                  f"95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]  "
                  f"sym {audit.symmetrised_auc(ci['auc']):.4f}  "
                  f"gap/spread = {gs:.3f}")

        # ------------------------------------------------ per-group fit
        key = f"{arch}|per_group_mse"
        if key in blob:
            mse = {g: float(v) for g, v in zip(spec.names, blob[key])}
        else:
            g2 = torch.Generator(device=dev).manual_seed(999)
            mse = per_group_mse(spec, M_full, arch, task, S, g2, dev)
        print("per-group query MSE (fully-trained; expect it to track PR)")
        print("  " + "  ".join(f"{g}={v:.4f}" for g, v in mse.items()))

        # ------------------------------------------------ verdict
        base = audit.symmetrised_auc(cis["output"]["auc"])
        if base > OK_THRESHOLD:
            v = ("OK", f"baseline AUC(output) = {base:.3f} > {OK_THRESHOLD}; "
                       "the audit has signal and the sweep is interpretable.")
        elif sep["ratio"] < 1.0:
            v = ("DEAD BASELINE - SAME MODEL LEARNED",
                 f"baseline AUC(output) = {base:.3f} and the ensembles are not "
                 f"separated in weight space either (ratio {sep['ratio']:.3f} < 1): "
                 "the fully-trained and oracle models converged to the same "
                 "function, so there is no membership signal to find. Fix the "
                 "DATA MODEL -- raise D and widen the spectral contrast between "
                 "the forget group and the retain groups (see "
                 "configs/regression_strong.yaml).")
        else:
            v = ("UNDERTRAINED ENSEMBLES",
                 f"baseline AUC(output) = {base:.3f} but the ensembles DO differ "
                 f"in weight space (ratio {sep['ratio']:.3f} > 1): the difference "
                 "exists and the probe is not reading it out. Train longer, or "
                 "check that the probe query is drawn from the forget group and "
                 "that within-ensemble spread has not swamped the gap "
                 f"(gap/spread = {diagnostics.gap_to_spread(obs1['output'], obs0['output']):.3f}).")
        verdicts[arch] = v
        print(f"\n  VERDICT [{arch}]: {v[0]}\n    {v[1]}")

    print(f"\n{'='*66}\nsummary")
    for arch, (tag, _) in verdicts.items():
        print(f"  {arch:8s} {tag}")
    if all(t == "OK" for t, _ in verdicts.values()):
        print("\nall architectures have signal -> run scripts/run_auc_sweep.py")
    else:
        print("\nat least one architecture has no baseline signal. Fix that "
              "before reading any sweep output.")


if __name__ == "__main__":
    main()
