#!/usr/bin/env python
"""
Figures from artifacts/results_{name}.csv.

    python scripts/make_figures.py --config configs/regression.yaml

Produces:
  figures/auc_sweep_{name}.pdf     AUC vs strength, arch x mode, both observables,
                                   with bootstrap bands and the masking control
  figures/frontier_{name}.pdf      (alpha, eps) against the Pareto frontier
  figures/staircase_{name}.pdf     training traces (ATTN-S staircase vs ATTN-M)

The AUC panels plot the MATCHED-CONTEXT AUC (`auc_matched_*`), not the legacy
clean-context one. See sweep.py's module docstring for why the clean-context
comparison is not a membership test. The legacy curve is drawn faintly so the
size of the confound stays visible.
"""
import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k not in ("mode", "arch"):
                r[k] = float(v)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    adir = pathlib.Path(cfg["paths"]["artifacts"])
    fdir = pathlib.Path(cfg["paths"]["figures"])
    fdir.mkdir(parents=True, exist_ok=True)

    rows = load(adir / f"results_{name}.csv")
    archs = sorted({r["arch"] for r in rows})
    modes = [m for m in ("C1", "C2", "C3", "flip", "whiten")
             if any(r["mode"] == m for r in rows)]

    # ---------------------------------------------------------- AUC sweeps
    fig, axes = plt.subplots(len(archs), len(modes),
                             figsize=(3.1 * len(modes), 2.8 * len(archs)),
                             squeeze=False, sharey=True)
    for i, arch in enumerate(archs):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            sel = sorted([r for r in rows if r["arch"] == arch and r["mode"] == mode],
                         key=lambda r: r["param"])
            xs = [r["param"] for r in sel]
            for obs, style, col in (("loss", "-o", "C0"), ("output", "-s", "C1")):
                ax.plot(xs, [r[f"auc_matched_{obs}"] for r in sel], style,
                        ms=3, color=col, label=f"AUC({obs})")
                ax.fill_between(xs,
                                [r[f"auc_matched_{obs}_lo"] for r in sel],
                                [r[f"auc_matched_{obs}_hi"] for r in sel],
                                color=col, alpha=0.20, lw=0)
                # legacy clean-context comparison, for reference only
                ax.plot(xs, [r[f"auc_{obs}"] for r in sel], ls="--", lw=0.7,
                        color=col, alpha=0.45)
            # shared-noise control: where it separates from the solid curve, the
            # drop is variance masking rather than removal
            if mode in ("C1", "C2", "C3"):
                ax.plot(xs, [r["auc_shared_loss"] for r in sel], ":", lw=1.1,
                        color="C0", alpha=0.9, label="AUC(loss), shared noise")
            ax.axhline(0.5, color="k", lw=0.8, ls=":")
            if mode == "flip":
                ax.axvline(0.5, color="r", lw=0.8, ls="--", alpha=0.6)
                ax.set_xlabel("strength t")
            elif mode == "whiten":
                ax.set_xlabel("interpolation")
            else:
                ax.set_xscale("symlog", linthresh=1e-3)
                ax.set_xlabel(r"$\sigma^2$")
            ax.set_ylim(0.4, 1.02)
            if j == 0:
                ax.set_ylabel(f"{arch}\nmembership AUC")
            if i == 0:
                ax.set_title(mode)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Membership AUC vs corruption strength — {name}", y=1.0)
    fig.tight_layout()
    fig.savefig(fdir / f"auc_sweep_{name}.pdf", bbox_inches="tight")

    # ------------------------------------------------------ Pareto frontier
    fig2, axes2 = plt.subplots(1, len(archs), figsize=(4.4 * len(archs), 3.8),
                               squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes2[0][i]
        by_mode = defaultdict(list)
        for r in rows:
            if r["arch"] == arch:
                by_mode[r["mode"]].append(r)
        base = next(iter(rows))["base"]
        al = torch.linspace(base, max(r["alpha"] for r in rows) * 1.05, 200)
        ax.plot(al, (al.sqrt() - base ** 0.5) ** 2, "k-", lw=1,
                label=r"frontier $\varepsilon_{\min}(\alpha)$")
        for mode, rs in by_mode.items():
            ax.scatter([r["alpha"] for r in rs], [r["eps"] for r in rs],
                       s=14, label=mode)
        ax.axvline(base, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"$\alpha$ = removal"); ax.set_ylabel(r"$\varepsilon$ = preservation")
        ax.set_title(arch); ax.legend(fontsize=7)
    fig2.suptitle(f"Removal–preservation plane — {name}")
    fig2.tight_layout()
    fig2.savefig(fdir / f"frontier_{name}.pdf", bbox_inches="tight")

    # ---------------------------------------------------------- staircases
    blob = torch.load(adir / f"ensembles_{name}.pt", map_location="cpu",
                      weights_only=False)
    fig3, ax3 = plt.subplots(figsize=(5, 3.4))
    for arch in archs:
        tr = blob[f"{arch}|full|trace"]
        k = max(1, len(tr) // 400)
        ax3.plot(range(0, len(tr), k), tr[::k], lw=1, label=f"{arch} (full)")
    ax3.set_xlabel("step"); ax3.set_ylabel("query MSE")
    ax3.set_title(f"Training dynamics — {name}")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(fdir / f"staircase_{name}.pdf", bbox_inches="tight")

    print(f"figures -> {fdir}/")


if __name__ == "__main__":
    main()
