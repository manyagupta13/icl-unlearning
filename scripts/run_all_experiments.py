#!/usr/bin/env python
"""
Batch runner for the three follow-up experiments (NOTES.md section 4g/6).

    python scripts/make_configs.py --out configs/generated
    python scripts/run_all_experiments.py --plan            # cost estimate, no work
    python scripts/run_all_experiments.py                   # run everything
    python scripts/run_all_experiments.py --tier rotation   # one experiment
    python scripts/run_all_experiments.py --resume          # skip finished configs

Designed for a Kaggle/Colab session that can drop at any moment:

  * RESUMABLE. Every stage checks for its own artifact first. A dropped
    session costs you the config that was in flight, not the batch.
  * ORDERED BY WHAT DECIDES THE OTHERS. rot_mid_identity runs first because
    it is a NULL (see PREDICTIONS.md P8: the three groups are the same
    distribution, so full and oracle train on indistinguishable data and the
    AUC must sit at chance). If that fails, nothing else in the batch means
    anything, so by default the runner STOPS.
  * LOUD ABOUT HEALTH WARNINGS. `check_ensemble_health` prints rather than
    raises when a shadow's ||M|| is an outlier (NOTES 4c). Buried in 16
    configs of scrolling output that is invisible, so it is re-surfaced in
    the final summary and written to artifacts/BATCH_STATUS.json.
  * COSTED BEFORE IT RUNS. --plan prints per-config sweep-tensor memory and a
    relative cost model, so you find out that nd_D32_N127 is ~26x the D=4
    config before you queue it, not four hours in.

Per-config logs land in artifacts/logs/{name}.{stage}.log.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "generated"

# Order matters. Within `rotation`, the identity twins come FIRST: they are
# nulls, and a null that does not sit at chance is a pipeline bug that would
# silently contaminate every other number in the batch.
TIERS: dict[str, list[str]] = {
    "rotation": ["rot_mid_identity", "rot_mid",
                 "rot_flat_identity", "rot_flat"],
    "pr":       ["pr_1p45", "pr_1p90", "pr_2p35",
                 "pr_2p80", "pr_3p25", "pr_3p70"],
    "nd":       ["nd_D4_N31", "nd_D8_N31", "nd_D16_N31",
                 "nd_D16_N63", "nd_D32_N63", "nd_D32_N127"],
}

NULL_CONFIGS = {"rot_mid_identity", "rot_flat_identity"}


# --------------------------------------------------------------------- costing

def cost_model(cfg: dict) -> dict:
    """
    Relative cost and peak sweep-tensor size. Not a wall-clock prediction --
    it is a ratio against the shipped D=4/N=31 config, which is the only thing
    that transfers across GPUs.
    """
    d, tr, pr, sw = cfg["data"], cfg["train"], cfg["probe"], cfg["sweep"]
    S, P, N, D = tr["n_shadows"], pr["P"], d["N"], d["D"]
    n_grid = sum(len(v) for v in sw["grids"].values())
    n_combos = (int(tr.get("n_train_seeds", 1))
                * int(pr.get("n_probe_seeds", 1)))
    n_archs = len(tr["archs"])

    # [S, P, N+1, D+1] float32, the tensor that dominates sweep memory
    bytes_per_copy = S * P * (N + 1) * (D + 1) * 4
    # forward-pass work per sweep point ~ S*P*N*D, and each point does
    # ~(2 + 2*n_shared_reps) frozen passes plus n_boot argsorts
    per_point = S * P * (N + 1) * (D + 1)
    sweep_units = per_point * n_grid * n_combos * n_archs
    # training: [S, B, N+1, D+1] per step
    train_units = (S * tr["batch_per_shadow"] * (N + 1) * (D + 1)
                   * tr["steps"] * n_archs * 2
                   * int(tr.get("n_train_seeds", 1)))
    return {
        "D": D, "N": N, "S": S, "P": P,
        "n_grid": n_grid, "n_combos": n_combos,
        "sweep_tensor_MB": bytes_per_copy / 1e6,
        "sweep_units": sweep_units,
        "train_units": train_units,
    }


def print_plan(names: list[str]) -> None:
    rows = []
    for n in names:
        cfg = yaml.safe_load(open(CFG_DIR / f"{n}.yaml"))
        c = cost_model(cfg)
        c["name"] = n
        rows.append(c)
    # normalise to the D=4/N=31 config when it is in the batch, so the "rel"
    # column means the same thing whichever tier you asked for
    base = next((r for r in rows if r["name"] == "nd_D4_N31"), rows[0])
    ref_s, ref_t = base["sweep_units"], base["train_units"]

    print(f"\n{'config':>18} {'D':>4} {'N':>5} {'grid':>5} {'combos':>7} "
          f"{'sweep MB/copy':>14} {'train rel':>10} {'sweep rel':>10}")
    print("-" * 84)
    tot_s = tot_t = 0.0
    for r in rows:
        tot_s += r["sweep_units"] / ref_s
        tot_t += r["train_units"] / ref_t
        print(f"{r['name']:>18} {r['D']:4d} {r['N']:5d} {r['n_grid']:5d} "
              f"{r['n_combos']:7d} {r['sweep_tensor_MB']:14.1f} "
              f"{r['train_units']/ref_t:10.2f} {r['sweep_units']/ref_s:10.2f}")
    print("-" * 84)
    print(f"{'TOTAL':>18} {'':4} {'':5} {'':5} {'':7} {'':14} "
          f"{tot_t:10.2f} {tot_s:10.2f}")
    print("\n'rel' = multiples of one nd_D4_N31 run. Time the first config, then")
    print("multiply. Sweep dominates: it is 80 grid points x 2 archs x 9 seed")
    print("combos = 1440 sweep_point() calls per config, each doing ~34 frozen")
    print("forward passes plus 400 bootstrap argsorts.")
    biggest = max(rows, key=lambda r: r["sweep_tensor_MB"])
    print(f"\nPeak sweep tensor: {biggest['sweep_tensor_MB']:.0f} MB/copy at "
          f"{biggest['name']}; sweep_point holds 2-3 copies live.")
    if biggest["sweep_tensor_MB"] > 400:
        print("  -> If this OOMs, lower probe.P before lowering train.n_shadows.")
        print("     S enters the AUC standard error directly; P only through")
        print("     correlated probe points. See PREDICTIONS.md P5.")


# ----------------------------------------------------------------- run helpers

def run(cmd: list[str], log: pathlib.Path) -> tuple[int, str]:
    """Run, tee to a log file, return (returncode, captured stdout+stderr)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    buf = []
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
            buf.append(line)
        p.wait()
    return p.returncode, "".join(buf)


def null_check(cfg: dict) -> tuple[bool, str]:
    """
    PREDICTIONS.md P8, via scripts/check_null.py.

    Delegated rather than inlined, because the obvious inline version is wrong.
    An earlier revision here required every per-row bootstrap CI at param=0 to
    cover 0.5. That failed on correct data for three reasons: the six modes at
    param=0 are the SAME measurement (identity edit) counted six times; 108
    simultaneous 95% intervals all covering has probability 0.95^108 = 0.4%
    under a perfect null; and the 9 (train, probe) combos reuse 3 ensembles so
    they are not independent. See check_null.py's docstring.
    """
    name = cfg["name"]
    path = ROOT / cfg["paths"]["artifacts"] / f"results_{name}.csv"
    if not path.exists():
        return False, "results CSV missing"
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_null                                   # noqa: PLC0415
    return check_null.check(path, verbose=True)


def stage_done(cfg: dict, stage: str) -> bool:
    adir = ROOT / cfg["paths"]["artifacts"]
    name = cfg["name"]
    if stage == "train":
        n = int(cfg["train"].get("n_train_seeds", 1))
        return all((adir / f"ensembles_{name}_ts{i}.pt").exists()
                   for i in range(n))
    if stage == "sweep":
        return (adir / f"results_{name}.csv").exists()
    return False


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS) + ["all"], default="all")
    ap.add_argument("--only", nargs="*", help="explicit config names")
    ap.add_argument("--plan", action="store_true",
                    help="print the cost table and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages whose artifact already exists")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--ignore-null-failure", action="store_true",
                    help="continue even if a null config is not at chance. "
                         "You almost certainly do not want this.")
    args = ap.parse_args()

    if args.only:
        names = args.only
    elif args.tier == "all":
        names = [n for t in ("rotation", "pr", "nd") for n in TIERS[t]]
    else:
        names = TIERS[args.tier]

    missing = [n for n in names if not (CFG_DIR / f"{n}.yaml").exists()]
    if missing:
        sys.exit(f"missing configs: {missing}\n"
                 f"run: python scripts/make_configs.py --out configs/generated")

    if args.plan:
        print_plan(names)
        return

    status: dict[str, dict] = {}
    status_path = ROOT / "artifacts" / "BATCH_STATUS.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    logdir = ROOT / "artifacts" / "logs"
    dev = ["--device", args.device] if args.device else []
    t_batch = time.time()

    print_plan(names)
    print("\n" + "=" * 84)

    for i, name in enumerate(names, 1):
        cfg_path = CFG_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        st = status.setdefault(name, {"stages": {}, "warnings": []})
        print(f"\n[{i}/{len(names)}] {name}   D={cfg['data']['D']} "
              f"N={cfg['data']['N']} basis={cfg['data']['basis']}"
              + ("   <-- NULL CONTROL" if name in NULL_CONFIGS else ""))
        print("=" * 84)

        for stage, script in (("train", "train_ensembles.py"),
                              ("sweep", "run_auc_sweep.py")):
            if args.resume and stage_done(cfg, stage):
                print(f"  [{stage}] artifact present, skipping (--resume)")
                st["stages"][stage] = "skipped"
                continue
            t0 = time.time()
            rc, out = run([sys.executable, f"scripts/{script}",
                           "--config", str(cfg_path)] + dev,
                          logdir / f"{name}.{stage}.log")
            dt = time.time() - t0
            st["stages"][stage] = {"rc": rc, "seconds": round(dt, 1)}
            if "WARNING" in out and "ensemble median" in out:
                for line in out.splitlines():
                    if "WARNING" in line:
                        st["warnings"].append(line.strip())
            if rc != 0:
                st["stages"][stage]["error"] = "nonzero exit"
                print(f"\n  !! {name}/{stage} FAILED (rc={rc}). "
                      f"See artifacts/logs/{name}.{stage}.log")
                json.dump(status, open(status_path, "w"), indent=2)
                break
            print(f"  [{stage}] ok in {dt:.0f}s")
        else:
            # null gate -- PREDICTIONS.md P8
            if name in NULL_CONFIGS:
                ok, msg = null_check(cfg)
                st["null_check"] = {"pass": ok, "detail": msg}
                banner = "PASS" if ok else "FAIL"
                print(f"\n  NULL CHECK [{banner}]: {msg}")
                if not ok and not args.ignore_null_failure:
                    print("\n  Stopping. A null config that is not at chance "
                          "means 'full' and 'oracle' are not independent.\n"
                          "  Look at stable_offset(arch, hyp) in "
                          "train_ensembles.py and the torch.manual_seed(seed)\n"
                          "  inside train_ensemble. Re-run with "
                          "--ignore-null-failure only if you have a reason.")
                    json.dump(status, open(status_path, "w"), indent=2)
                    return

            if not args.no_plots:
                for script in ("plot_auc_vs_var.py", "plot_tradeoff.py",
                               "make_figures.py"):
                    rc, _ = run([sys.executable, f"scripts/{script}",
                                 "--config", str(cfg_path)],
                                logdir / f"{name}.{script}.log")
                    st["stages"][script] = rc

        json.dump(status, open(status_path, "w"), indent=2)

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 84)
    print(f"BATCH DONE in {(time.time()-t_batch)/60:.1f} min")
    print("=" * 84)
    warned = {k: v["warnings"] for k, v in status.items() if v.get("warnings")}
    if warned:
        print("\n!! ENSEMBLE-HEALTH WARNINGS -- do not interpret these configs "
              "before investigating (NOTES.md 4c):")
        for k, v in warned.items():
            print(f"  {k}:")
            for line in v[:3]:
                print(f"    {line}")
    else:
        print("\nno ensemble-health warnings")

    for k, v in status.items():
        if "null_check" in v:
            print(f"null check {k}: "
                  f"{'PASS' if v['null_check']['pass'] else 'FAIL'} "
                  f"-- {v['null_check']['detail']}")
    print(f"\nstatus -> {status_path}")
    print("logs   -> artifacts/logs/")
    print("\nNext: score PREDICTIONS.md section 8 against the results CSVs.")


if __name__ == "__main__":
    main()
