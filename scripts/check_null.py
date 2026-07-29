#!/usr/bin/env python
"""
Statistically correct null test for the rot_*_identity configs.

    python scripts/check_null.py --config configs/generated/rot_mid_identity.yaml

Runs on an EXISTING results CSV. No retraining, no GPU.

WHY THIS FILE EXISTS
--------------------
The first version of this check (inline in run_all_experiments.py) required
every per-row bootstrap CI to cover 0.5. That was wrong three times over, and
the three errors compound:

  1. DUPLICATE ROWS. At param = 0 every corruption mode is the identity edit,
     so the `none`, `C1`, `C2`, `C3`, `flip` and `whiten` rows at param = 0 are
     the SAME measurement recorded six times. The old check treated them as six
     independent tests.

  2. MULTIPLE COMPARISONS. 9 seed combos x 2 archs x 6 (duplicated) modes = 108
     tests, each a 95% interval, all required to cover. Under a PERFECT null
     that succeeds with probability 0.95^108 = 0.004. The check was designed to
     fail regardless of the data.

  3. CORRELATED REPLICATES. The 9 (train, probe) combos reuse 3 trained
     ensembles across 3 probes. They are not 9 independent draws, so a t-test
     with df = 8 overstates significance. The replicated unit is the TRAINING
     SEED.

WHAT THIS DOES INSTEAD
----------------------
  - keeps only mode == "none" (deduplicates)
  - averages over probe seeds within each training seed  -> one value per
    (arch, train seed), which is the correctly clustered unit
  - two-sided t-test of those cluster means against 0.5, df = n_train_seeds - 1
  - ALSO applies a hard absolute bound, because with 3 training seeds the
    t-test has almost no power: a gross failure (say AUC = 0.35) could pass a
    df=2 test if the three seeds happen to agree with each other. NOTES.md
    section 5 calibrates the noise floor at 0.01-0.02, so |mean - 0.5| > 0.02
    is treated as a hard fail on its own.

A null config must satisfy BOTH. Report both numbers either way -- "it passed"
is much less informative than "it passed at 0.502 +/- 0.001".
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import defaultdict

import yaml

# NOTES.md section 5: at S=100 two genuinely identical distributions measured
# 0.488. At S=512 the noise floor is roughly half that. 0.02 is deliberately
# loose -- this bound exists to catch gross failures, not subtle ones.
ABS_TOL = 0.02
ALPHA = 0.01          # for the clustered t-test; deliberately strict, because
                      # we do not want seed noise tripping the batch gate

# two-sided critical values, df = 1..8, at alpha = 0.01
_T_CRIT = {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604,
           5: 4.032, 6: 3.707, 7: 3.499, 8: 3.355}


def load_none_rows(path, observable="residual"):
    """-> {(arch, train_seed_idx): {probe_seed_idx: auc}}, deduplicated."""
    by = defaultdict(dict)
    field = f"auc_matched_{observable}"
    with open(path) as fh:
        for r in csv.DictReader(fh):
            # param = 0 makes every mode the identity edit; keep one copy
            if r["mode"] != "none":
                continue
            if float(r["param"]) != 0.0:
                continue
            ts = int(float(r.get("train_seed_idx", 0)))
            ps = int(float(r.get("probe_seed_idx", 0)))
            by[(r["arch"], ts)][ps] = float(r[field])
    return by


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v)


def check(path, observable="residual", verbose=True):
    by = load_none_rows(path, observable)
    if not by:
        return False, f"no mode='none' rows in {path}"

    archs = sorted({a for (a, _) in by})
    verdicts, details = [], []

    for arch in archs:
        seeds = sorted(ts for (a, ts) in by if a == arch)
        # cluster: average over probe seeds within each training seed
        cluster = [sum(by[(arch, ts)].values()) / len(by[(arch, ts)])
                   for ts in seeds]
        flat = [v for ts in seeds for v in by[(arch, ts)].values()]

        m, sd = mean_sd(cluster)
        n = len(cluster)
        sem = sd / math.sqrt(n) if n > 1 else float("nan")
        t = (m - 0.5) / sem if sem and sem == sem and sem > 0 else 0.0
        df = n - 1
        tcrit = _T_CRIT.get(df, 2.576)

        abs_ok = abs(m - 0.5) <= ABS_TOL
        t_ok = abs(t) <= tcrit
        ok = abs_ok and t_ok
        verdicts.append(ok)

        if verbose:
            print(f"\n  {arch}   observable = {observable}, mode = none, param = 0")
            print(f"    {len(flat)} rows -> {n} training-seed clusters "
                  f"(averaged over {len(flat)//n} probe seeds each)")
            print(f"    per-seed means : "
                  + "  ".join(f"{v:.4f}" for v in cluster))
            print(f"    mean           : {m:.4f}   (deviation {m-0.5:+.4f})")
            print(f"    across-seed sd : {sd:.4f}   sem {sem:.4f}")
            print(f"    t (df={df})       : {t:+.2f}   |t| crit at "
                  f"alpha={ALPHA}: {tcrit:.2f}   -> "
                  f"{'ok' if t_ok else 'REJECT'}")
            print(f"    |dev| <= {ABS_TOL}  : {abs(m-0.5):.4f}  -> "
                  f"{'ok' if abs_ok else 'REJECT'}")
            print(f"    VERDICT        : {'PASS' if ok else 'FAIL'}")

        details.append(f"{arch}: {m:.4f}{m-0.5:+.4f} "
                       f"t={t:+.2f}/{tcrit:.2f} "
                       f"{'ok' if ok else 'FAIL'}")

    return all(verdicts), "; ".join(details)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--observable", default="residual",
                    choices=["residual", "loss"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / cfg["paths"]["artifacts"] / f"results_{cfg['name']}.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run run_auc_sweep.py first")

    print("=" * 74)
    print(f"NULL CHECK  {cfg['name']}")
    print("=" * 74)
    print("""
rot_*_identity gives all groups the same spectrum AND the same basis, so they
are one distribution: 'full' and 'oracle' train on indistinguishable data and
the membership AUC must sit at chance. See PREDICTIONS.md P8.""")

    ok, detail = check(path, args.observable)

    print("\n" + "=" * 74)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 74)
    if ok:
        print("""
The pipeline separates 'full' and 'oracle' correctly. Proceed with the batch.

Note the residual deviation reported above. If it is a consistent few
thousandths on the same side across archs, that is worth a sentence in the
write-up (and more training seeds would settle it), but it is far below the
0.01-0.02 noise floor NOTES.md section 5 measured and is not a blocker.""")
    else:
        print("""
Investigate before running anything else. Places to look, in order:
  - stable_offset(arch, hyp) in train_ensembles.py: do 'full' and 'oracle'
    actually get different seeds?
  - torch.manual_seed(seed) inside train_ensemble: does the init differ?
  - make_probe: is the forget slice the one the config names?
  - is the deviation the SAME SIGN for both archs and all seeds? A consistent
    sign points at a systematic coupling; alternating signs point at noise.""")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
