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
import numpy as np
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

    # If run_auc_sweep.py was run across multiple (train_seed, probe_seed)
    # combinations, this diagnostic grid is not seed-aware (unlike
    # plot_auc_vs_var.py) and would plot several overlapping values per x,
    # producing a zigzag. Restrict to the first combo and say so; use
    # plot_auc_vs_var.py for the cross-seed view.
    if "train_seed_idx" in (rows[0] if rows else {}):
        ts0 = min(r["train_seed_idx"] for r in rows)
        ps0 = min(r["probe_seed_idx"] for r in rows)
        n_before = len(rows)
        rows = [r for r in rows
               if r["train_seed_idx"] == ts0 and r["probe_seed_idx"] == ps0]
        if n_before != len(rows):
            print(f"note: results contain multiple seed combinations; "
                 f"this diagnostic grid shows only train_seed_idx={ts0:.0f}, "
                 f"probe_seed_idx={ps0:.0f} ({len(rows)}/{n_before} rows). "
                 f"Use plot_auc_vs_var.py for the cross-seed aggregate.")

    archs = sorted({r["arch"] for r in rows})
    modes = [m for m in ("C1", "C2", "C3", "flip", "whiten")
             if any(r["mode"] == m for r in rows)]

    # ---------------------------------------------------------- AUC sweeps
    #
    # Layout rules, learned the hard way from the first version of this figure:
    #
    #  * symlog with linthresh=1e-3 renders the NEGATIVE branch of the axis even
    #    though every sigma^2 is >= 0. That is what produced the unreadable
    #    "-10^0 10^-1 10^-2 10^-3 0 10^-3 ..." tick pile-up. Clamp xlim to
    #    [0, max] and set the tick locations explicitly.
    #  * a hard ylim of (0.4, 1.02) left the data squashed into the bottom 15%
    #    of every panel. Autoscale to the data plus the null band instead.
    #  * five overlapping lines per panel (2 solid + 2 dashed + 1 dotted) is too
    #    many, and the dashed legacy curve was the visually dominant one despite
    #    being the quantity we explicitly do not want read. It moves to its own
    #    figure below.
    def style_x(ax, mode, xs):
        if mode == "flip":
            ax.axvline(0.5, color="r", lw=0.8, ls="--", alpha=0.6)
            ax.set_xlabel("flip strength $t$")
        elif mode == "whiten":
            ax.set_xlabel("interpolation")
        else:
            lt = 1e-3
            ax.set_xscale("symlog", linthresh=lt, linscale=0.35)
            ax.set_xlim(0, max(xs) * 1.3)
            decades = [0.0] + [10.0 ** k for k in range(-3, 3)
                               if 10.0 ** k <= max(xs)]
            ax.set_xticks(decades)
            ax.set_xticklabels(["0"] + [rf"$10^{{{int(round(np.log10(d)))}}}$"
                                        for d in decades[1:]])
            ax.set_xlabel(r"noise variance $\sigma^2$")
        ax.tick_params(axis="x", labelsize=8)

    OBS = (("loss", "-o", "C0", r"loss $\ell=(\hat y-y)^2$"),
           ("residual", "-s", "C1", r"residual $\mathrm{sign}(y)(\hat y-y)$"))

    fig, axes = plt.subplots(len(archs), len(modes),
                             figsize=(3.3 * len(modes), 3.0 * len(archs)),
                             squeeze=False, sharey=True)
    lo_all, hi_all = [], []
    for i, arch in enumerate(archs):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            sel = sorted([r for r in rows if r["arch"] == arch and r["mode"] == mode],
                         key=lambda r: r["param"])
            xs = [r["param"] for r in sel]
            for obs, style, col, lab in OBS:
                ax.plot(xs, [r[f"auc_matched_{obs}"] for r in sel], style,
                        ms=3.5, lw=1.4, color=col, label=lab, zorder=3)
                lo = [r[f"auc_matched_{obs}_lo"] for r in sel]
                hi = [r[f"auc_matched_{obs}_hi"] for r in sel]
                ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)
                lo_all += lo
                hi_all += hi
            if mode in ("C1", "C2", "C3"):
                ax.plot(xs, [r["auc_shared_loss"] for r in sel], ":", lw=1.3,
                        color="C4", zorder=2,
                        label="loss, shared noise (masking control)")
            ax.axhline(0.5, color="k", lw=0.9, ls=":", zorder=0)
            style_x(ax, mode, xs)
            if j == 0:
                ax.set_ylabel(f"{arch}\nmembership AUC")
            if i == 0:
                ax.set_title(mode)
    # one shared y-range, sized to the data and always containing 0.5
    pad = 0.02
    ylo = min(min(lo_all), 0.5) - pad
    yhi = max(max(hi_all), 0.5) + pad
    for row in axes:
        for ax in row:
            ax.set_ylim(ylo, yhi)
    axes[0][0].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle(f"Membership AUC vs corruption strength — {name}   "
                 f"(matched context, {int(cfg['train']['n_shadows'])} shadows, "
                 f"95% bootstrap CI)", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fdir / f"auc_sweep_{name}.pdf", bbox_inches="tight")

    # ------------------------------------------- confound / artefact diagnostics
    # Everything we do NOT want overlaid on the main figure, kept so the size of
    # each artefact stays auditable.
    figd, axd = plt.subplots(len(archs), len(modes),
                             figsize=(3.3 * len(modes), 3.0 * len(archs)),
                             squeeze=False, sharey=True)
    for i, arch in enumerate(archs):
        for j, mode in enumerate(modes):
            ax = axd[i][j]
            sel = sorted([r for r in rows if r["arch"] == arch and r["mode"] == mode],
                         key=lambda r: r["param"])
            xs = [r["param"] for r in sel]
            ax.plot(xs, [r["auc_matched_loss"] for r in sel], "-o", ms=3,
                    color="C0", label="matched context (correct)")
            ax.plot(xs, [r["auc_loss"] for r in sel], "--", lw=1.2, color="C3",
                    label="clean-context H0 (confounded)")
            ax.plot(xs, [r["auc_matched_output_unaligned"] for r in sel], "-.",
                    lw=1.1, color="C7", label=r"un-aligned $\hat y$ (cancels)")
            ax.axhline(0.5, color="k", lw=0.9, ls=":")
            style_x(ax, mode, xs)
            ax.set_ylim(0.3, 1.02)
            if j == 0:
                ax.set_ylabel(f"{arch}\nmembership AUC")
            if i == 0:
                ax.set_title(mode)
    axd[0][0].legend(fontsize=7, loc="best", framealpha=0.9)
    figd.suptitle(f"Artefact diagnostics — {name}  (gap between C0 and C3 is the "
                  f"context confound; C7 is sign cancellation)", y=0.99, fontsize=10)
    figd.tight_layout(rect=(0, 0, 1, 0.96))
    figd.savefig(fdir / f"diagnostics_{name}.pdf", bbox_inches="tight")

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
    # Illustrative only -- shows training-seed index 0, not an aggregate.
    blob = torch.load(adir / f"ensembles_{name}_ts0.pt", map_location="cpu",
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
