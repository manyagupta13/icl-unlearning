#!/usr/bin/env python
"""
Pre-flight verification for the 16-config batch, WITHOUT torch.

torch could not be installed in this sandbox (the 526 MB wheel download is
killed by the environment), so the training/sweep numerics cannot be executed
here. Everything that does NOT need torch is checked:

  1. every source + script file compiles
  2. every generated config is internally consistent (eigs length == D,
     probe counts sum to N, forget group present, PR matches the target)
  3. run_all_experiments.py's cost model and --plan run
  4. null_check() reads columns that run_auc_sweep.py actually writes
  5. plot_auc_vs_var.py and plot_tradeoff.py run end-to-end on a synthetic
     results CSV carrying the REAL column set, including the string cfg_*
     columns that used to crash the blanket float() cast
  6. the figure filenames no longer collide across configs
"""
import csv
import itertools
import json
import pathlib
import py_compile
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "generated"
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# `configs/generated/` is gitignored (reproducible from make_configs.py), so a
# fresh clone has none of the 16. Say that plainly instead of emitting a wall
# of downstream failures and a FileNotFoundError traceback.
if not list(CFG.glob("*.yaml")):
    sys.exit(f"no configs in {CFG.relative_to(ROOT)}/ -- this check runs AFTER "
             f"config generation:\n\n"
             f"    python scripts/make_configs.py --out configs/generated\n"
             f"    python tests/verify_batch_preflight.py\n")


# ------------------------------------------------------------------ 1 compile
print("\n1. compile everything")
for p in sorted(itertools.chain(ROOT.glob("src/**/*.py"),
                                ROOT.glob("scripts/*.py"),
                                ROOT.glob("tests/*.py"))):
    try:
        py_compile.compile(str(p), doraise=True)
        ok = True
        err = ""
    except py_compile.PyCompileError as e:
        ok, err = False, str(e).splitlines()[-1]
    check(str(p.relative_to(ROOT)), ok, err)


# ------------------------------------------------------- 2 config consistency
print("\n2. generated configs are internally consistent")


def pr_of(eigs):
    e = np.array(eigs, dtype=np.float64)
    e = e / e.sum()
    return float(e.sum() ** 2 / (e ** 2).sum())


names = sorted(p.stem for p in CFG.glob("*.yaml"))
check("16 configs present", len(names) == 16, f"{len(names)} found")
for n in names:
    c = yaml.safe_load(open(CFG / f"{n}.yaml"))
    d, pcfg = c["data"], c["probe"]
    D, N = d["D"], d["N"]
    probs = []
    if c["name"] != n:
        probs.append(f"name mismatch {c['name']!r}")
    for g in d["groups"]:
        if len(d["eigs"][g]) != D:
            probs.append(f"{g}: {len(d['eigs'][g])} eigs != D={D}")
        tgt = d.get("_target_pr", {}).get(g)
        if tgt is not None and abs(pr_of(d["eigs"][g]) - tgt) > 1e-6:
            probs.append(f"{g}: PR {pr_of(d['eigs'][g]):.6f} != target {tgt}")
        if not (0 < min(d["eigs"][g])):
            probs.append(f"{g}: non-positive eigenvalue")
    if sum(pcfg["counts"].values()) != N:
        probs.append(f"counts sum {sum(pcfg['counts'].values())} != N={N}")
    if d["forget"] not in d["groups"]:
        probs.append("forget group not in groups")
    if pcfg["counts"].get(d["forget"], 0) < 1:
        probs.append("forget group has no probe tokens")
    if d["basis"] not in ("identity", "random"):
        probs.append(f"bad basis {d['basis']}")
    check(n, not probs, "; ".join(probs))


# --------------------------------------------------------------- 3 --plan run
print("\n3. run_all_experiments.py --plan")
r = subprocess.run([sys.executable, "scripts/run_all_experiments.py", "--plan"],
                   cwd=ROOT, capture_output=True, text=True)
check("--plan exits 0", r.returncode == 0, r.stderr.strip()[-200:])
check("--plan costs nd_D32_N127 at ~26x",
      "26.40" in r.stdout, "")
r2 = subprocess.run([sys.executable, "scripts/run_all_experiments.py",
                     "--plan", "--tier", "rotation"], cwd=ROOT,
                    capture_output=True, text=True)
check("--tier rotation works", r2.returncode == 0 and "rot_mid" in r2.stdout)


# ---------------------------------------------- 4/5 synthetic results CSV run
print("\n4+5. plot scripts on a synthetic results CSV (real column set)")

GRIDS = {
    "none": [0.0],
    "C1": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "C2": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "C3": [0.0, 0.01, 1.0],
    "flip": [0.0, 0.5, 1.0],
    "whiten": [0.0, 0.5, 1.0],
}


def synth_rows(cfg, rng, null=False):
    """Rows with exactly the keys sweep.py + run_auc_sweep.py emit."""
    d, tr, pcfg = cfg["data"], cfg["train"], cfg["probe"]
    meta = {
        "cfg_D": d["D"], "cfg_N": d["N"],
        "cfg_ND_ratio": round(d["N"] / d["D"], 6),
        "cfg_basis": d["basis"], "cfg_forget": d["forget"],   # <- strings
        "cfg_n_shadows": tr["n_shadows"], "cfg_steps": tr["steps"],
        "cfg_optim": tr["optim"], "cfg_lr": tr["lr"],          # <- string
        "cfg_probe_P": pcfg["P"],
        "cfg_n_train_seeds": 3, "cfg_n_probe_seeds": 3,
    }
    for g in d["groups"]:
        meta[f"cfg_PR_{g}"] = 2.0
    rows = []
    for ts, ps in itertools.product(range(3), range(3)):
        for arch in tr["archs"]:
            for mode, params in GRIDS.items():
                for prm in params:
                    a = 0.5 if null else 0.5 - 0.12 / (1 + 40 * prm)
                    a += rng.normal(0, 0.004)
                    half = 0.02
                    rec = {"mode": mode, "param": float(prm), "arch": arch,
                           "train_seed_idx": ts, "probe_seed_idx": ps}
                    for nm in ("loss", "residual"):
                        rec[f"auc_{nm}"] = a + 0.01
                        rec[f"auc_matched_{nm}"] = a
                        rec[f"auc_matched_{nm}_lo"] = a - half
                        rec[f"auc_matched_{nm}_hi"] = a + half
                        rec[f"auc_matched_{nm}_sym"] = abs(a - 0.5) + 0.5
                        rec[f"auc_shared_{nm}"] = a + 0.03
                        rec[f"masking_{nm}"] = 0.03
                        rec[f"spread_h1_{nm}"] = 0.1
                        rec[f"spread_h0_{nm}"] = 0.1
                        rec[f"spread_h0_clean_{nm}"] = 0.1
                    rec["auc_matched_output_unaligned"] = 0.5
                    rec["auc_theory_residual"] = (a if mode in ("C1", "C2")
                                                  else float("nan"))
                    rec["alpha"] = 0.01 + prm
                    rec["eps"] = 0.001 + 0.01 * prm
                    rec["base"] = 0.005
                    rec["eps_min"] = 0.0005
                    rec["gap"] = 0.0005
                    rec["mmd2_to_oracle"] = 0.001
                    rec.update(meta)
                    rows.append(rec)
    return rows


rng = np.random.default_rng(0)
tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "artifacts").mkdir()
(tmp / "figures").mkdir()

produced = {}
seen: set[str] = set()      # figures dir is SHARED, so diff it between configs
for n, is_null in (("rot_mid_identity", True), ("nd_D32_N127", False)):
    cfg = yaml.safe_load(open(CFG / f"{n}.yaml"))
    cfg["paths"] = {"artifacts": str(tmp / "artifacts"),
                    "figures": str(tmp / "figures")}
    cpath = tmp / f"{n}.yaml"
    yaml.safe_dump(cfg, open(cpath, "w"), sort_keys=False)
    rows = synth_rows(cfg, rng, null=is_null)
    out = tmp / "artifacts" / f"results_{n}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for script in ("plot_auc_vs_var.py", "plot_tradeoff.py"):
        r = subprocess.run([sys.executable, f"scripts/{script}",
                            "--config", str(cpath)],
                           cwd=ROOT, capture_output=True, text=True)
        check(f"{script} on {n}", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
    now = {p.name for p in (tmp / "figures").glob("*.pdf")}
    produced[n] = now - seen        # files THIS config wrote
    seen = now

# 6. filename collision check
print("\n6. figure filenames are config-scoped")
a, b = produced["rot_mid_identity"], produced["nd_D32_N127"]
overlap = a & b
check("no filename collision between two configs", not overlap,
      f"overlap={sorted(overlap)[:4]}")
check("names carry the config name",
      all("rot_mid_identity" in f or "nd_D32_N127" in f for f in a | b),
      f"e.g. {sorted(a)[:2]}")

# 4. null_check wiring
print("\n4b. null_check() reads real columns")
sys.path.insert(0, str(ROOT / "scripts"))
import run_all_experiments as R                                   # noqa: E402

cfg_null = yaml.safe_load(open(tmp / "rot_mid_identity.yaml"))
cfg_null["paths"]["artifacts"] = str(tmp / "artifacts")
ok, msg = R.null_check(cfg_null)
check("null config passes the null gate", ok, msg)

cfg_sig = yaml.safe_load(open(tmp / "nd_D32_N127.yaml"))
cfg_sig["paths"]["artifacts"] = str(tmp / "artifacts")
ok2, msg2 = R.null_check(cfg_sig)
check("config WITH signal is flagged by the null gate", not ok2, msg2[:90])

print("\n" + "=" * 70)
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("all pre-flight checks passed")
print("NOT checked here (needs torch/GPU): training, the sweep numerics,")
print("theory.py's closed form, and actual memory use at D=32.")
