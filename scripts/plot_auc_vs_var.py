#!/usr/bin/env python
"""
The plot from the spec: AUC vs Var(epsilon), one clean image per corruption family.

    python scripts/plot_auc_vs_var.py --config configs/regression.yaml

Writes, for each of C1 / C2 / C3:
    figures/auc_{mode}.pdf   AUC on the output observable vs Var(eps)
    figures/eps_{mode}.pdf   preservation eps = KL(p_oracle || p_unlearned) vs Var(eps)
and .png alongside each.

    C1:  (x_f, y_f) -> (x_f,     y_f + e)     label noise
    C2:  (x_f, y_f) -> (x_f + e, y_f)         input noise
    C3:  (x_f, y_f) -> (x_f + e1, y_f + e2)   both

One figure, one panel, two lines (ATTN-S / ATTN-M). Nothing else on it.

If results_{name}.csv has more than one (train_seed_idx, probe_seed_idx)
combination -- i.e. run_auc_sweep.py was run with train.n_train_seeds > 1 or
probe.n_probe_seeds > 1 -- the band switches from the within-run bootstrap CI
to the min-max range ACROSS seed combinations, and the line becomes the
across-seed median. This is deliberate: per NOTES.md section 2, the within-run
CI covers shadow/probe sampling noise but says nothing about whether a
different training seed gives a different ensemble mean. If the two kinds of
uncertainty disagree, the across-seed one is the one to trust.
"""
import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

MODES = ("C1", "C2", "C3")
TITLES = {
    "C1": r"C1: $(x_f,\, y_f + \epsilon)$   label noise",
    "C2": r"C2: $(x_f + \epsilon,\, y_f)$   input noise",
    "C3": r"C3: $(x_f + \epsilon_1,\, y_f + \epsilon_2)$   both",
}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k not in ("mode", "arch"):
                r[k] = float(v)
    return rows


def logx(ax, xs):
    """Log axis in Var(eps) that still shows the sigma^2 = 0 control point."""
    ax.set_xscale("symlog", linthresh=1e-3, linscale=0.35)
    ax.set_xlim(0, max(xs) * 1.3)
    dec = [0.0] + [10.0 ** k for k in range(-3, 3) if 10.0 ** k <= max(xs)]
    ax.set_xticks(dec)
    ax.set_xticklabels(["0"] + [rf"$10^{{{int(round(np.log10(d)))}}}$" for d in dec[1:]])
    ax.set_xlabel(r"$\mathrm{Var}(\epsilon)$", fontsize=12)


def n_seed_combos(rows):
    combos = {(r.get("train_seed_idx", 0.0), r.get("probe_seed_idx", 0.0))
             for r in rows}
    return len(combos)


def collapse_by_param(rows, arch, mode, field):
    """
    All values of `field` for (arch, mode), grouped by param, across every
    seed combination present. Returns (xs_sorted, median, lo, hi) where lo/hi
    are the min/max across seed combos (not a bootstrap CI).
    """
    by_param = defaultdict(list)
    for r in rows:
        if r["arch"] == arch and r["mode"] == mode:
            by_param[r["param"]].append(r[field])
    xs = sorted(by_param)
    med = [float(np.median(by_param[x])) for x in xs]
    lo = [min(by_param[x]) for x in xs]
    hi = [max(by_param[x]) for x in xs]
    return xs, med, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--observable", default="residual",
                    choices=["residual", "loss"],
                    help="'residual' is the sign-aligned output observable "
                         "sign(y)*(yhat-y); raw yhat cancels to 0.5, see NOTES.md")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    adir = pathlib.Path(cfg["paths"]["artifacts"])
    fdir = pathlib.Path(cfg["paths"]["figures"])
    fdir.mkdir(parents=True, exist_ok=True)

    rows = load(adir / f"results_{name}.csv")
    archs = sorted({r["arch"] for r in rows})
    obs = args.observable
    S = int(cfg["train"]["n_shadows"])
    n_combos = n_seed_combos(rows)
    multi_seed = n_combos > 1
    band_label = (f"min-max over {n_combos} (train,probe) seed pairs"
                 if multi_seed else "95% bootstrap CI")
    print(f"seed combinations in results: {n_combos} "
         f"({'cross-seed' if multi_seed else 'single-seed'} mode)")

    for mode in MODES:
        if not any(r["mode"] == mode for r in rows):
            continue

        # ---------------------------------------------------- AUC vs Var(eps)
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        for arch, mk, col in zip(archs, ("-o", "-s"), ("C0", "C1")):
            if multi_seed:
                xs, ys, lo, hi = collapse_by_param(
                    rows, arch, mode, f"auc_matched_{obs}")
            else:
                sel = sorted([r for r in rows if r["arch"] == arch and r["mode"] == mode],
                             key=lambda r: r["param"])
                xs = [r["param"] for r in sel]
                ys = [r[f"auc_matched_{obs}"] for r in sel]
                lo = [r[f"auc_matched_{obs}_lo"] for r in sel]
                hi = [r[f"auc_matched_{obs}_hi"] for r in sel]
            ax.plot(xs, ys, mk, ms=4, lw=1.6, color=col, label=arch, zorder=3)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)

        ax.axhline(0.5, color="k", lw=1.0, ls=":", zorder=0)
        ax.text(0.985, 0.5, " chance", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=8, color="0.35")
        logx(ax, xs)
        ax.set_ylabel("membership AUC", fontsize=12)
        ax.set_title(TITLES[mode], fontsize=12)
        ax.legend(fontsize=10, frameon=False)
        ax.margins(y=0.12)
        fig.text(0.99, 0.01, f"{S} shadows, {band_label}", ha="right",
                 va="bottom", fontsize=7, color="0.45")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fdir / f"auc_{mode}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # ---------------------------------------------------- eps vs Var(eps)
        fig2, ax2 = plt.subplots(figsize=(5.4, 4.0))
        for arch, mk, col in zip(archs, ("-o", "-s"), ("C0", "C1")):
            if multi_seed:
                xs, med, lo, hi = collapse_by_param(rows, arch, mode, "eps")
                ax2.plot(xs, med, mk, ms=4, lw=1.6, color=col, label=arch)
                ax2.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
            else:
                sel = sorted([r for r in rows if r["arch"] == arch and r["mode"] == mode],
                             key=lambda r: r["param"])
                xs = [r["param"] for r in sel]
                ax2.plot(xs, [r["eps"] for r in sel], mk, ms=4, lw=1.6,
                        color=col, label=arch)
        logx(ax2, xs)
        ax2.set_ylabel(r"$\varepsilon$  (preservation, lower is better)", fontsize=12)
        ax2.set_title(TITLES[mode], fontsize=12)
        ax2.legend(fontsize=10, frameon=False)
        ax2.margins(y=0.12)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(fdir / f"eps_{mode}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig2)

        print(f"  {mode}: figures/auc_{mode}.pdf  figures/eps_{mode}.pdf")

    print(f"\nwritten to {fdir}/")


if __name__ == "__main__":
    main()
