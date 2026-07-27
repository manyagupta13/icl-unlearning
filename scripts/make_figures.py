#!/usr/bin/env python
"""
Figures from artifacts/results_{task}.csv.

    python scripts/make_figures.py --config configs/regression.yaml

ONE FIGURE PER CORRUPTION MODE. The old 2x5 grid of thumbnails made every
curve unreadable and, worse, hid the thing that mattered: the no-edit baseline.
Each mode now gets a full page, one panel per architecture.

    figures/auc_C1_{task}.pdf        additive label noise
    figures/auc_C2_{task}.pdf        additive input noise
    figures/auc_C3_{task}.pdf        both
    figures/auc_flip_{task}.pdf      tunable sign strength
    figures/auc_whiten_{task}.pdf    covariance interpolation
    figures/frontier_{task}.pdf      (alpha, eps) against the Pareto frontier
    figures/dynamics_{task}.pdf      training traces, full vs oracle

X-axis: sigma^2 = 0 is NOT a point on a log or symlog axis. symlog spent half
the panel rendering a linear neighbourhood of zero that held exactly one point.
The sigma^2 = 0 value is drawn as a labelled horizontal reference line and the
positive grid gets a clean log axis. `flip` and `whiten` are linear in their
parameter and are plotted as such.

Every panel carries three references:
    grey band   no-edit baseline AUC (fully-trained vs oracle, control) + CI
    dashed      the sigma^2 = 0 value from this mode's own grid
    dotted 0.5  the target -- an edit succeeds by reaching it FROM the baseline
If the grey band sits near 0.5 the audit had no signal and nothing in the
figure is interpretable; run scripts/diagnose.py.
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

MODES = ("C1", "C2", "C3", "flip", "whiten")
LOG_MODES = ("C1", "C2", "C3")          # parameter is sigma^2 -> log axis
OBS = (("output", "tab:blue", "o"), ("loss", "tab:orange", "s"))

XLABEL = {"C1": r"label-noise $\sigma^2$",
          "C2": r"input-noise $\sigma^2$",
          "C3": r"joint noise $\sigma^2$",
          "flip": r"sign strength $t$",
          "whiten": "interpolation toward retain covariance"}

TITLE = {"C1": "C1 — additive label noise on forget tokens",
         "C2": "C2 — additive input noise on forget tokens",
         "C3": "C3 — additive label + input noise",
         "flip": "flip — tunable sign strength  $y_f \\to (1-2t)\\,y_f$",
         "whiten": "whiten — forget inputs mapped toward retain covariance"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k not in ("mode", "arch"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float("nan")
    return rows


def get(row, key, default=None):
    """Tolerate result CSVs written before the CI/baseline columns existed."""
    v = row.get(key, default)
    return default if v is None else v


def has(rows, key):
    return key in rows[0]


# ------------------------------------------------------------------ one panel

def draw_panel(ax, sel, mode, rows_have_ci, rows_have_base):
    """sel: rows for one (arch, mode), sorted by param."""
    log = mode in LOG_MODES
    pos = [r for r in sel if r["param"] > 0] if log else sel
    zero = next((r for r in sel if r["param"] == 0), None) if log else None

    # ---- no-edit baseline: the control every curve is read against
    if rows_have_base and sel:
        b = sel[0]
        lo = get(b, "baseline_auc_output_lo")
        hi = get(b, "baseline_auc_output_hi")
        if lo is not None and hi is not None:
            ax.axhspan(lo, hi, color="0.75", alpha=0.35, zorder=0)
        ax.axhline(b["baseline_auc_output"], color="0.35", lw=1.2, ls="-",
                   zorder=1, label="no-edit baseline (output)")
        ax.axhline(b["baseline_auc_loss"], color="0.60", lw=1.0, ls="-",
                   zorder=1, label="no-edit baseline (loss)")

    # ---- swept curves
    for name, colour, marker in OBS:
        xs = [r["param"] for r in pos]
        ys = [r[f"auc_{name}"] for r in pos]
        if rows_have_ci:
            lo = [r[f"auc_{name}_lo"] for r in pos]
            hi = [r[f"auc_{name}_hi"] for r in pos]
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.20, lw=0, zorder=2)
        ax.plot(xs, ys, marker=marker, color=colour, ms=4, lw=1.4, zorder=3,
                label=f"AUC({name})")

        # sigma^2 = 0 belongs on no log axis. Draw it as a reference level.
        if zero is not None:
            ax.axhline(zero[f"auc_{name}"], color=colour, lw=1.0, ls="--",
                       alpha=0.8, zorder=2,
                       label=rf"$\sigma^2=0$, AUC({name})")

    # ---- target
    ax.axhline(0.5, color="k", lw=0.9, ls=":", zorder=4, label="target 0.5")

    # ---- axes
    if log:
        ax.set_xscale("log")
    elif mode == "flip":
        # t = 0.5 kills the label column entirely: the zero-mean dead zone.
        ax.axvline(0.5, color="crimson", lw=1.0, ls="--", alpha=0.7, zorder=1)
        dead = next((r for r in sel if abs(r["param"] - 0.5) < 1e-9), None)
        if dead is not None:
            ax.plot([0.5], [dead["auc_output"]], marker="*", ms=15,
                    color="crimson", zorder=6, label="dead zone $t=0.5$")
            ax.annotate("dead zone\n$(1-2t)=0$", xy=(0.5, dead["auc_output"]),
                        xytext=(8, 14), textcoords="offset points",
                        fontsize=7.5, color="crimson")
    ax.set_xlabel(XLABEL[mode])
    ax.margins(x=0.03)


def figure_for_mode(rows, mode, archs, task, fdir, rows_have_ci, rows_have_base):
    fig, axes = plt.subplots(1, len(archs), figsize=(5.4 * len(archs), 4.2),
                             squeeze=False, sharey=True)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        sel = sorted([r for r in rows if r["arch"] == arch and r["mode"] == mode],
                     key=lambda r: r["param"])
        if not sel:
            ax.set_visible(False)
            continue
        draw_panel(ax, sel, mode, rows_have_ci, rows_have_base)
        ax.set_title(arch, fontsize=11)
        if i == 0:
            ax.set_ylabel("membership AUC")
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=2)

    lo = min(0.45, min(r["auc_output"] for r in rows if r["mode"] == mode) - 0.03)
    axes[0][0].set_ylim(lo, 1.02)

    fig.suptitle(f"{TITLE[mode]}  —  {task}", y=0.99, fontsize=12)
    if rows_have_ci:
        fig.text(0.5, -0.02,
                 "bands = 95% percentile bootstrap CI over shadow models; "
                 "grey = no-edit control (fully-trained vs retrain oracle)",
                 ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout()
    out = fdir / f"auc_{mode}_{task}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- other figures

def figure_frontier(rows, archs, task, fdir):
    fig, axes = plt.subplots(1, len(archs), figsize=(4.8 * len(archs), 4.0),
                             squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        by_mode = defaultdict(list)
        for r in rows:
            if r["arch"] == arch and r["mode"] in MODES:
                by_mode[r["mode"]].append(r)
        base = next(r["base"] for r in rows if r["arch"] == arch)
        top = max(r["alpha"] for r in rows if r["arch"] == arch) * 1.05
        al = torch.linspace(base, max(top, base * 1.01), 200)
        ax.plot(al, (al.sqrt() - base ** 0.5) ** 2, "k-", lw=1.2,
                label=r"frontier $\varepsilon_{\min}(\alpha)$")
        for mode in MODES:
            if mode in by_mode:
                rs = by_mode[mode]
                ax.scatter([r["alpha"] for r in rs], [r["eps"] for r in rs],
                           s=20, label=mode, alpha=0.85)
        ax.axvline(base, color="gray", ls=":", lw=0.9)
        ax.annotate(r"$\alpha_{\min}$ = intrinsic gap", xy=(base, 0),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=7.5, color="gray")
        ax.set_xlabel(r"$\alpha$ = removal  (larger better)")
        if i == 0:
            ax.set_ylabel(r"$\varepsilon$ = preservation  (smaller better)")
        ax.set_title(arch, fontsize=11)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Removal–preservation plane — {task}", fontsize=12)
    fig.tight_layout()
    out = fdir / f"frontier_{task}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_dynamics(blob, archs, task, fdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colours = dict(zip(archs, ["tab:blue", "tab:orange", "tab:green"]))
    for arch in archs:
        for hyp, ls in (("full", "-"), ("oracle", "--")):
            key = f"{arch}|{hyp}|trace"
            if key not in blob:
                continue
            tr = blob[key]
            k = max(1, len(tr) // 600)
            ax.plot(range(0, len(tr), k), tr[::k], ls, lw=1.1,
                    color=colours.get(arch), alpha=0.9 if hyp == "full" else 0.55,
                    label=f"{arch} ({hyp})")
    ax.set_yscale("log")
    ax.set_xlabel("SGD step")
    ax.set_ylabel("query MSE")
    ax.set_title(f"Training dynamics — {task}\n"
                 "(ATTN-S staircase vs ATTN-M; flat tail = converged, "
                 "still-falling tail = undertrained)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fdir / f"dynamics_{task}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    task = cfg["task"]
    adir = pathlib.Path(cfg["paths"]["artifacts"])
    fdir = pathlib.Path(cfg["paths"]["figures"])
    fdir.mkdir(parents=True, exist_ok=True)

    rows = load(adir / f"results_{task}.csv")
    archs = sorted({r["arch"] for r in rows})
    rows_have_ci = has(rows, "auc_output_lo")
    rows_have_base = has(rows, "baseline_auc_output")

    if not rows_have_base:
        print("note: results CSV predates the baseline columns — re-run "
              "scripts/run_auc_sweep.py to get the no-edit reference lines.")

    written = []
    for mode in MODES:
        if any(r["mode"] == mode for r in rows):
            written.append(figure_for_mode(rows, mode, archs, task, fdir,
                                           rows_have_ci, rows_have_base))

    written.append(figure_frontier(rows, archs, task, fdir))

    blob = torch.load(adir / f"ensembles_{task}.pt", map_location="cpu",
                      weights_only=False)
    written.append(figure_dynamics(blob, archs, task, fdir))

    # loud, because a dead baseline invalidates every figure above
    if rows_have_base:
        print()
        for arch in archs:
            b = next(r for r in rows if r["arch"] == arch)
            v = b["baseline_auc_output"]
            flag = "OK" if max(v, 1 - v) > 0.9 else "DEAD BASELINE — see diagnose.py"
            print(f"  {arch:8s} no-edit baseline AUC(output) = {v:.3f}   {flag}")

    print("\nfigures:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
