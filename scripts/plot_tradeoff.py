#!/usr/bin/env python
"""
The removal-preservation tradeoff: does chance-level AUC cost you the model?

    python scripts/plot_tradeoff.py --config configs/regression.yaml

AUC -> 0.5 on its own is not a result. You can always reach chance by adding
enough noise to destroy the context entirely. The claim worth making is whether
there exists a sigma* at which the attacker is defeated (AUC ~ 0.5) while the
model is still close to the retrain oracle (eps small).

Writes, for each of C1 / C2 / C3:
  figures/tradeoff_{mode}.pdf   twin-axis: AUC and eps vs Var(eps), with the
                                eps-minimising and AUC-chance points marked
  figures/frontier_{mode}.pdf   the (alpha, eps) plane against the theoretical
                                frontier eps_min(alpha) = (sqrt(a)-sqrt(b))^2
and prints a table of the operating points.

Reading the tradeoff panel
--------------------------
Two vertical markers per architecture:
  sigma_eps   where eps is minimised          -- best preservation
  sigma_auc   where |AUC - 0.5| is minimised  -- best removal
If these coincide, the corruption family has a clean operating point. If
sigma_auc >> sigma_eps (AUC only reaches chance long after preservation has
blown up), the family CANNOT remove without destroying, and that gap is the
headline number.
"""
import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

MODES = ("C1", "C2", "C3", "flip", "whiten")
TITLES = {"C1": "C1  label noise", "C2": "C2  input noise", "C3": "C3  both",
          "flip": "ICUL / label flip", "whiten": "whiten"}
AXIS_KIND = {"C1": "var", "C2": "var", "C3": "var",
             "flip": "strength", "whiten": "interp"}

# t=1 is exact sign inversion (5 -> -5): parameter-free ICUL. Reported as a
# named operating point so it can be compared against the tuned noise families.
ICUL_T = 1.0


def load(path):
    """Numeric columns -> float; `cfg_*` string columns stay strings."""
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def logx(ax, xs, mode="C1"):
    kind = AXIS_KIND.get(mode, "var")
    if kind == "var":
        ax.set_xscale("symlog", linthresh=1e-3, linscale=0.35)
        ax.set_xlim(0, max(xs) * 1.3)
        dec = [0.0] + [10.0 ** k for k in range(-3, 3) if 10.0 ** k <= max(xs)]
        ax.set_xticks(dec)
        ax.set_xticklabels(["0"] +
                           [rf"$10^{{{int(round(np.log10(d)))}}}$" for d in dec[1:]])
        ax.set_xlabel(r"$\mathrm{Var}(\epsilon) = \sigma^2$", fontsize=11)
    elif kind == "strength":
        ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
        ax.set_xlabel(r"flip strength $t$   ($t{=}1$ is ICUL)", fontsize=11)
        if min(xs) <= ICUL_T <= max(xs):
            ax.axvline(ICUL_T, color="g", lw=1.1, ls="--", alpha=0.6, zorder=0)
    else:
        ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02)
        ax.set_xlabel("interpolation toward retain covariance", fontsize=11)


def agg(rows, arch, mode, field):
    """Median over seed combinations, grouped by param."""
    by = defaultdict(list)
    for r in rows:
        if r["arch"] == arch and r["mode"] == mode:
            by[r["param"]].append(r[field])
    xs = sorted(by)
    return xs, np.array([float(np.median(by[x])) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--observable", default="residual", choices=["residual", "loss"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    adir = pathlib.Path(cfg["paths"]["artifacts"])
    fdir = pathlib.Path(cfg["paths"]["figures"])
    fdir.mkdir(parents=True, exist_ok=True)

    rows = load(adir / f"results_{name}.csv")
    archs = sorted({r["arch"] for r in rows})
    obs = args.observable

    print(f"\n{'mode':6s} {'arch':8s} {'sigma*_eps':>11s} {'eps_min':>10s} "
          f"{'AUC there':>10s} | {'sigma*_AUC':>11s} {'AUC there':>10s} "
          f"{'eps there':>10s} | {'eps ratio':>9s}")
    print("-" * 100)

    summary = []
    for mode in MODES:
        if not any(r["mode"] == mode for r in rows):
            continue

        # ------------------------------------------------ tradeoff twin axis
        fig, ax = plt.subplots(figsize=(5.8, 4.3))
        ax2 = ax.twinx()

        for arch, mk, col in zip(archs, ("o", "s"), ("C0", "C1")):
            xs, auc = agg(rows, arch, mode, f"auc_matched_{obs}")
            _, eps = agg(rows, arch, mode, "eps")
            xs = np.array(xs)

            ax.plot(xs, auc, "-" + mk, ms=4, lw=1.6, color=col,
                    label=f"{arch}  AUC", zorder=3)
            ax2.plot(xs, eps, "--", lw=1.3, color=col, alpha=0.75,
                     label=f"{arch}  $\\varepsilon$", zorder=2)

            # operating points
            i_eps = int(np.argmin(eps))
            i_auc = int(np.argmin(np.abs(auc - 0.5)))
            ax.axvline(xs[i_eps], color=col, ls=":", lw=1.0, alpha=0.55)
            ax.axvline(xs[i_auc], color=col, ls="-.", lw=1.0, alpha=0.55)

            ratio = eps[i_auc] / max(eps[i_eps], 1e-12)
            summary.append((mode, arch, xs[i_eps], eps[i_eps], auc[i_eps],
                            xs[i_auc], auc[i_auc], eps[i_auc], ratio))
            print(f"{mode:6s} {arch:8s} {xs[i_eps]:11.4g} {eps[i_eps]:10.5f} "
                  f"{auc[i_eps]:10.4f} | {xs[i_auc]:11.4g} {auc[i_auc]:10.4f} "
                  f"{eps[i_auc]:10.5f} | {ratio:9.1f}x")

        ax.axhline(0.5, color="k", lw=0.9, ls=":", zorder=0)
        ax2.set_yscale("log")
        logx(ax, xs, mode)
        ax.set_ylabel("membership AUC  (solid)", fontsize=11)
        ax2.set_ylabel(r"$\varepsilon$ preservation, log scale  (dashed)", fontsize=11)
        ax.set_title(f"{TITLES[mode]} — removal vs preservation", fontsize=11)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center left")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.text(0.99, 0.005,
                 "dotted line: $\\varepsilon$ minimum   "
                 "dash-dot: AUC closest to chance", ha="right", va="bottom",
                 fontsize=6.5, color="0.45")
        for ext in ("pdf", "png"):
            fig.savefig(fdir / f"tradeoff_{mode}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------ alpha-eps frontier
        fig2, axf = plt.subplots(figsize=(5.2, 4.2))
        for arch, mk, col in zip(archs, ("o", "s"), ("C0", "C1")):
            xs_, alpha = agg(rows, arch, mode, "alpha")
            _, eps = agg(rows, arch, mode, "eps")
            _, base = agg(rows, arch, mode, "base")
            b = float(np.median(base))
            axf.scatter(alpha, eps, s=18, color=col, label=arch, zorder=3)
            a_grid = np.linspace(b, max(alpha.max(), b) * 1.05, 200)
            axf.plot(a_grid, (np.sqrt(a_grid) - np.sqrt(b)) ** 2, "-",
                     color=col, lw=1.0, alpha=0.6,
                     label=f"{arch} frontier" if arch == archs[0] else None)
        axf.set_xscale("log"); axf.set_yscale("log")
        axf.set_xlabel(r"$\alpha$  removal (larger better)", fontsize=11)
        axf.set_ylabel(r"$\varepsilon$  preservation (smaller better)", fontsize=11)
        axf.set_title(f"{TITLES[mode]} — removal/preservation plane", fontsize=11)
        axf.legend(fontsize=8, frameon=False)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(fdir / f"frontier_{mode}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig2)

    print("\n'eps ratio' = how much preservation you pay to reach chance-level AUC.")
    print("Ratio ~1 means a clean operating point exists. Ratio >> 1 means this")
    print("corruption family cannot defeat the attacker without wrecking the model,")
    print("which is the headline result if it holds.")

    # ---------------------------------------------------- ICUL head-to-head
    # Parameter-free ICUL is the single point t=1 on the flip arm. The noise
    # families get to TUNE sigma^2; ICUL does not get to tune anything. So the
    # fair comparison is ICUL against each family's best achievable point, and
    # the honest question is whether tuning buys you anything over just
    # flipping the sign.
    if any(r["mode"] == "flip" for r in rows):
        print("\n" + "=" * 100)
        print("ICUL (t=1, exact sign flip) vs the tunable noise families")
        print("=" * 100)
        print(f"{'arch':8s} {'method':22s} {'AUC':>9s} {'|AUC-0.5|':>10s} "
              f"{'eps':>10s}   {'verdict':s}")
        print("-" * 100)
        for arch in archs:
            xs_f, auc_f = agg(rows, arch, "flip", f"auc_matched_{obs}")
            _, eps_f = agg(rows, arch, "flip", "eps")
            xs_f = np.array(xs_f)
            j = int(np.argmin(np.abs(xs_f - ICUL_T)))
            icul_dev, icul_eps = abs(auc_f[j] - 0.5), eps_f[j]
            print(f"{arch:8s} {'ICUL (t=1)':22s} {auc_f[j]:9.4f} "
                  f"{icul_dev:10.4f} {icul_eps:10.5f}   (no tuning)")

            for mode in ("C1", "C2", "C3"):
                if not any(r["mode"] == mode for r in rows):
                    continue
                xs_c, auc_c = agg(rows, arch, mode, f"auc_matched_{obs}")
                _, eps_c = agg(rows, arch, mode, "eps")
                # best point = closest to chance among those no worse than
                # ICUL on preservation; falls back to closest-to-chance overall
                ok = np.where(np.array(eps_c) <= max(icul_eps, 1e-12))[0]
                pool = ok if len(ok) else np.arange(len(auc_c))
                k = pool[int(np.argmin(np.abs(np.array(auc_c)[pool] - 0.5)))]
                dev = abs(auc_c[k] - 0.5)
                better = "beats ICUL" if dev < icul_dev else "worse than ICUL"
                note = "" if len(ok) else "  (no point matches ICUL's eps)"
                print(f"{'':8s} {mode + f' best @ s2={xs_c[k]:g}':22s} "
                      f"{auc_c[k]:9.4f} {dev:10.4f} {eps_c[k]:10.5f}   "
                      f"{better}{note}")
            print()
        print("If no tuned noise setting beats ICUL at equal-or-better eps, then")
        print("the parameter-free sign flip is the strongest edit in this family,")
        print("and the sigma^2 sweeps are a (useful) negative result.")

    print(f"\nwritten to {fdir}/")


if __name__ == "__main__":
    main()
