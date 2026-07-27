#!/usr/bin/env python
"""
Run the corruption sweep against cached ensembles.

    python scripts/run_auc_sweep.py --config configs/regression.yaml

Loads artifacts/ensembles_{name}.pt, sweeps every (arch, mode, strength) in the
config grid, and writes artifacts/results_{name}.csv.

No training happens here. Re-run freely with new grids.
"""
import argparse
import csv
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
    name, d, pr = cfg["name"], cfg["data"], cfg["probe"]
    dev = args.device

    adir = pathlib.Path(cfg["paths"]["artifacts"])
    blob = torch.load(adir / f"ensembles_{name}.pt", map_location=dev,
                      weights_only=False)

    spec = MixtureSpec(names=d["groups"], eigs=d["eigs"], D=d["D"], N=d["N"],
                       basis=d["basis"], seed=d["seed"])

    gen = torch.Generator(device=dev).manual_seed(pr["seed"])
    probe = make_probe(spec, pr["counts"], d["forget"], pr["P"], gen, dev)

    # empirical retain-group inputs for the whiten arm (never the true Lambda)
    retain = [g for g in d["groups"] if g != d["forget"]]
    retain_x = torch.cat([
        torch.randn(2048, spec.D, generator=gen, device=dev)
        @ spec.sqrt_cov(g, dev, torch.float32).T for g in retain])

    ensembles = {}
    for arch in cfg["train"]["archs"]:
        for hyp in ("full", "oracle"):
            ensembles[(arch, hyp)] = blob[f"{arch}|{hyp}|M"].to(dev)

    rows = run_sweep(spec, probe, ensembles, cfg["sweep"]["grids"],
                     seed=cfg["sweep"]["seed"], device=dev, retain_x=retain_x,
                     n_boot=cfg["sweep"].get("n_boot", 200))

    path = adir / f"results_{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
