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
from matplotlib.lines import Line2D

MODES = ("C1", "C2", "C3", "flip", "bern", "whiten")

# Two kinds of x-axis. The stochastic families sweep a noise variance (log
# scale, spanning decades); the deterministic edits sweep a strength in [0, ~1]
# (linear). Mixing them on one axis type makes both unreadable.
AXIS_KIND = {"C1": "var", "C2": "var", "C3": "var",
             "flip": "strength", "bern": "prob", "whiten": "interp"}

# The variance x-axis is a single scalar knob sigma^2 = the config `param`. It
# is the variance of each scalar noise COMPONENT, in every family. Spelled out
# in the titles because "Var(eps)" alone is ambiguous for C2 (eps is a
# D-vector) and for C3 (there is no single eps -- there are two independent
# ones sharing sigma^2).
TITLES = {
    "C1": "C1  label noise\n"
          r"$y_f + \epsilon,\ \ \epsilon\sim\mathcal{N}(0,\sigma^2)$",
    "C2": "C2  input noise\n"
          r"$x_f + \epsilon,\ \ \epsilon\sim\mathcal{N}(0,\sigma^2 I_D)$",
    "C3": "C3  both (independent draws, shared $\\sigma^2$)\n"
          r"$x_f + \epsilon_1,\ y_f + \epsilon_2,\ \ \epsilon_1\perp\epsilon_2$",
    # t=1 is exact sign inversion: y_f -> -y_f, i.e. 5 becomes -5. That point
    # IS parameter-free ICUL; the rest of the axis is a generalisation of it.
    "flip": "ICUL / label flip  (deterministic)\n"
            r"$y_f \to (1-2t)\,y_f$   ($t{=}1$ is exact flip, $5\to-5$)",
    "bern": "Bernoulli label flip  (stochastic)\n"
            r"$y_f \to (1-2B)\,y_f,\ \ B\sim\mathrm{Bern}(\theta)$",
    "whiten": "whiten  input-space\n"
              r"$x_f \to$ interpolate toward retain covariance",
}

# Perturbation magnitude per forget token. Equal x does NOT mean equal
# perturbation across panels, so this travels with the figure.
FOOTER_ENERGY = {
    "C1": r"perturbs 1 component/token:  $\mathbb{E}\|\delta\|^2=\sigma^2$",
    "C2": r"perturbs $D$ components/token:  $\mathbb{E}\|\delta\|^2=D\sigma^2$",
    "C3": r"perturbs $D{+}1$ components/token:  "
          r"$\mathbb{E}\|\delta\|^2=(D{+}1)\sigma^2$",
    # deterministic: delta_y = -2t y_f exactly, so the "energy" is data-scaled
    "flip": r"deterministic, label only:  $\delta y=-2t\,y_f$,  "
            r"$\mathbb{E}\|\delta\|^2=4t^2\,\mathbb{E}[y_f^2]$",
    "bern": r"stochastic, label only:  $\mathbb{E}[\delta y]=-2\theta y_f$,  "
            r"$\mathrm{Var}=4\theta(1-\theta)y_f^2$ (max at $\theta{=}0.5$)",
    "whiten": r"deterministic, inputs only; magnitude set by the "
              r"$\Lambda_f\!\to\!\Lambda_r$ gap",
}

# Landmarks worth a vertical line, per mode.
VLINES = {
    "flip": [(0.5, "r", r"$t{=}0.5$: $\tilde y_f{=}0$, zero-mean dead zone"),
             (1.0, "g", r"$t{=}1$: ICUL (exact flip)")],
    "bern": [(0.5, "r", r"$\theta{=}0.5$: maximum variance $4\theta(1-\theta)$"),
             (1.0, "g", r"$\theta{=}1$: deterministic flip, variance $\to 0$")],
}


def load(path):
    """
    Read the results CSV, coercing numeric columns to float and leaving the
    rest as strings. Do NOT hardcode which columns are non-numeric: the sweep
    now writes self-describing `cfg_*` columns, some of which are strings
    (basis, optim, forget), and a blanket float() cast crashes on them.
    """
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass                      # leave as string
    return rows


def logx(ax, xs, mode="C1"):
    """
    Per-mode x-axis. `var` modes get a symlog decade axis in sigma^2 that still
    shows the sigma^2 = 0 control point; `strength`/`interp` modes get a plain
    linear axis, since t only ever runs over [0, ~1.25].
    """
    kind = AXIS_KIND.get(mode, "var")
    if kind == "var":
        ax.set_xscale("symlog", linthresh=1e-3, linscale=0.35)
        ax.set_xlim(0, max(xs) * 1.3)
        dec = [0.0] + [10.0 ** k for k in range(-3, 3) if 10.0 ** k <= max(xs)]
        ax.set_xticks(dec)
        ax.set_xticklabels(["0"] +
                           [rf"$10^{{{int(round(np.log10(d)))}}}$" for d in dec[1:]])
        ax.set_xlabel(r"$\mathrm{Var}(\epsilon) = \sigma^2$   "
                      r"(per noise component)", fontsize=11)
    elif kind == "strength":
        ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
        ax.set_xlabel(r"flip strength $t$   ($y_f \to (1-2t)\,y_f$)", fontsize=11)
    elif kind == "prob":
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(r"flip probability $\theta$   "
                      r"(variance $4\theta(1-\theta)$, max at $0.5$)", fontsize=11)
    else:
        ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02)
        ax.set_xlabel("interpolation toward retain covariance", fontsize=11)

    for xv, col, _lab in VLINES.get(mode, []):
        if min(xs) <= xv <= max(xs):
            ax.axvline(xv, color=col, lw=1.0, ls="--", alpha=0.55, zorder=0)


def collapse_theory(rows, arch, mode):
    """
    Median closed-form AUC per param, skipping NaN (modes with no derived
    form). Returns ([], []) if nothing usable, so the caller can just not draw.
    """
    by = defaultdict(list)
    for r in rows:
        if r["arch"] == arch and r["mode"] == mode:
            v = r.get("auc_theory_residual", float("nan"))
            if v == v:                       # NaN check without importing math
                by[r["param"]].append(v)
    xs = sorted(by)
    return xs, [float(np.median(by[x])) for x in xs]


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
    has_theory = "auc_theory_residual" in (rows[0] if rows else {})
    band_label = (f"min-max over {n_combos} (train,probe) seed pairs"
                 if multi_seed else "95% bootstrap CI")
    print(f"seed combinations in results: {n_combos} "
         f"({'cross-seed' if multi_seed else 'single-seed'} mode)")

    for mode in MODES:
        if not any(r["mode"] == mode for r in rows):
            continue

        # ---------------------------------------------------- AUC vs Var(eps)
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
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

            # Closed-form prediction (C1/C2 only; NaN elsewhere by design).
            # Drawn thin and underneath so measurement stays the foreground.
            if has_theory:
                tx, ty = collapse_theory(rows, arch, mode)
                if tx:
                    ax.plot(tx, ty, "-", lw=1.0, color="k", alpha=0.55,
                            zorder=2,
                            label="closed form" if arch == archs[0] else None)

        ax.axhline(0.5, color="k", lw=1.0, ls=":", zorder=0)
        ax.text(0.985, 0.5, " chance", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=8, color="0.35")
        logx(ax, xs, mode)
        ax.set_ylabel("membership AUC", fontsize=12)
        ax.set_title(TITLES[mode], fontsize=10.5)
        # name the landmark lines (t=0.5 dead zone, t=1 ICUL) in the legend
        handles, labels = ax.get_legend_handles_labels()
        for xv, col, lab in VLINES.get(mode, []):
            if min(xs) <= xv <= max(xs):
                handles.append(Line2D([], [], color=col, ls="--", lw=1.0, alpha=0.7))
                labels.append(lab)
        ax.legend(handles, labels, fontsize=8, frameon=False)
        ax.margins(y=0.12)
        # tight_layout ignores fig.text, so reserve the bottom strip explicitly
        # or the footer lands on top of the x-label.
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        fig.text(0.99, 0.005, f"{S} shadows, {band_label}\n"
                 + FOOTER_ENERGY[mode],
                 ha="right", va="bottom", fontsize=6.5, color="0.45")
        for ext in ("pdf", "png"):
            fig.savefig(fdir / f"auc_{mode}_{name}.{ext}", dpi=180,
                        bbox_inches="tight")
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
        logx(ax2, xs, mode)
        ax2.set_ylabel(r"$\varepsilon$  (preservation, lower is better)", fontsize=12)
        ax2.set_title(TITLES[mode], fontsize=10.5)
        ax2.legend(fontsize=10, frameon=False)
        ax2.margins(y=0.12)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(fdir / f"eps_{mode}_{name}.{ext}", dpi=180,
                         bbox_inches="tight")
        plt.close(fig2)

        print(f"  {mode}: figures/auc_{mode}_{name}.pdf  "
              f"figures/eps_{mode}_{name}.pdf")

    print(f"\nwritten to {fdir}/")


if __name__ == "__main__":
    main()
