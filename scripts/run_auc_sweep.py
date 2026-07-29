#!/usr/bin/env python
"""
Run the corruption sweep against cached ensembles.

    python scripts/run_auc_sweep.py --config configs/regression.yaml

Loops over every (training-seed, probe-seed) combination:
    train.n_train_seeds  x  probe.n_probe_seeds
loading artifacts/ensembles_{name}_ts{ts}.pt for each training seed and
rebuilding the probe for each probe seed. Every row in the output CSV is
tagged with `train_seed_idx` and `probe_seed_idx` so plot_auc_vs_var.py can
show the across-seed spread rather than just the within-run bootstrap CI.

Writes artifacts/results_{name}.csv. No training happens here -- re-run freely
with new grids.
"""
import argparse
import csv
import json
import pathlib
import sys

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning.data import MixtureSpec, make_probe    # noqa: E402
from icl_unlearning.sweep import run_sweep                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name, d, pr, tr, sw = (cfg["name"], cfg["data"], cfg["probe"],
                           cfg["train"], cfg["sweep"])
    dev = args.device
    n_train_seeds = int(tr.get("n_train_seeds", 1))
    n_probe_seeds = int(pr.get("n_probe_seeds", 1))
    print(f"n_train_seeds={n_train_seeds}  n_probe_seeds={n_probe_seeds}  "
         f"-> {n_train_seeds * n_probe_seeds} combinations")

    spec = MixtureSpec(names=d["groups"], eigs=d["eigs"], D=d["D"], N=d["N"],
                       basis=d["basis"], seed=d["seed"])
    retain = [g for g in d["groups"] if g != d["forget"]]
    adir = pathlib.Path(cfg["paths"]["artifacts"])

    all_rows = []
    for ts in range(n_train_seeds):
        blob = torch.load(adir / f"ensembles_{name}_ts{ts}.pt", map_location=dev,
                          weights_only=False)
        ensembles = {}
        for arch in tr["archs"]:
            for hyp in ("full", "oracle"):
                ensembles[(arch, hyp)] = blob[f"{arch}|{hyp}|M"].to(dev)

        for ps in range(n_probe_seeds):
            probe_seed = pr["seed"] + ps
            print(f"  [ts={ts} ps={ps}] train_seed={blob.get('train_seed')} "
                 f"probe_seed={probe_seed}")
            gen = torch.Generator(device=dev).manual_seed(probe_seed)
            probe = make_probe(spec, pr["counts"], d["forget"], pr["P"], gen, dev)

            # empirical retain-group inputs for the whiten arm (never the true Lambda)
            retain_x = torch.cat([
                torch.randn(2048, spec.D, generator=gen, device=dev)
                @ spec.sqrt_cov(g, dev, torch.float32).T for g in retain])

            rows = run_sweep(spec, probe, ensembles, sw["grids"],
                             seed=sw["seed"] + 1_000_000 * ts + 1_000 * ps,
                             device=dev, retain_x=retain_x,
                             n_boot=sw.get("n_boot", 200),
                             n_shared_reps=sw.get("n_shared_reps", 16),
                             mmd_max_n=sw.get("mmd_max_n", 2000))
            for r in rows:
                r["train_seed_idx"] = ts
                r["probe_seed_idx"] = ps
            all_rows.extend(rows)

    # Make the artifact self-describing. A results CSV that cannot answer
    # "what settings produced this?" is not usable evidence six months later,
    # and it is the first thing anyone reviewing the numbers will ask.
    # Key scalars go in as constant columns so the CSV alone is sufficient;
    # the full config (including every eigenvalue) goes in a sidecar.
    meta = {
        "cfg_D": d["D"], "cfg_N": d["N"],
        "cfg_ND_ratio": round(d["N"] / d["D"], 6),
        "cfg_basis": d["basis"], "cfg_forget": d["forget"],
        "cfg_n_shadows": tr["n_shadows"], "cfg_steps": tr["steps"],
        "cfg_optim": tr["optim"], "cfg_lr": tr["lr"],
        "cfg_probe_P": pr["P"],
        "cfg_n_train_seeds": n_train_seeds,
        "cfg_n_probe_seeds": n_probe_seeds,
    }
    for g in d["groups"]:
        meta[f"cfg_PR_{g}"] = round(spec.pr(g), 6)
    for r in all_rows:
        r.update(meta)

    path = adir / f"results_{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    side = adir / f"results_{name}_config.json"
    with open(side, "w") as f:
        json.dump({"config": cfg, "participation_ratios":
                   {g: spec.pr(g) for g in d["groups"]}}, f, indent=2)

    print(f"\n{len(all_rows)} rows -> {path}")
    print(f"full config    -> {side}")
    print("  " + "  ".join(f"{k}={v}" for k, v in list(meta.items())[:6]))


if __name__ == "__main__":
    main()
