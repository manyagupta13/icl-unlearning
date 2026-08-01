#!/usr/bin/env python
"""
Go/no-go probe for the MNIST extension. CPU only, ~30 seconds, no GPU needed.

    python scripts/mnist_pr_probe.py
    python scripts/mnist_pr_probe.py --dims 4 8 16 32 --center per-class

THE QUESTION THIS ANSWERS
-------------------------
This repo's thesis is that spectral geometry -- the participation ratio of a
group's input covariance -- governs how unlearnable that group is. Porting to
MNIST only makes sense if MNIST's ten digit classes actually SPREAD OUT in PR.
If every digit lands at PR ~ 0.8 D, there is no spectral geometry to study and
the port reduces to "we swapped the input distribution", which is not a result.

So before building anything, measure:

  1. per-digit PR of the class-conditional covariance, at each PCA dimension D
  2. the spread (max/min) -- compare against the synthetic config, which uses
     PR = 1.90 / 2.38 / 3.34 at D = 4, i.e. a 1.76x span
  3. tr(Lambda^-1) trace-normalised, which is what sets the C2 noise channel
     (PREDICTIONS.md section 2) and therefore the predicted spread of transition
     locations across digits

THE CENTERING FORK
------------------
data.py's MixtureSpec says groups "share mean 0 and mixture weight; they differ
ONLY in covariance spectrum". MNIST digits differ mostly in their MEANS. So:

  --center per-class   subtract each digit's own mean. Faithful to the current
                       design and to theory.py, but digit identity is now
                       reduced to covariance SHAPE, and the classes may become
                       nearly indistinguishable.
  --center pooled      subtract one global mean. Digits stay digits, but the
                       group means now differ, which is a channel section 0's
                       algebra does not cover at all.

Both are computed by default so the cost of the choice is visible.

MNIST SOURCE
------------
Tries torchvision, then keras, then sklearn's OpenML fetch. On Kaggle any of
these works; torchvision is usually already present.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


# ------------------------------------------------------------------- loading

def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """-> (X [n, 784] float64 in [0,1], y [n] int)"""
    errs = []
    try:
        from torchvision import datasets            # noqa: PLC0415
        ds = datasets.MNIST(root="./data", train=True, download=True)
        X = ds.data.numpy().reshape(len(ds), -1).astype(np.float64) / 255.0
        return X, ds.targets.numpy().astype(int)
    except Exception as e:                          # noqa: BLE001
        errs.append(f"torchvision: {e}")
    try:
        from tensorflow.keras.datasets import mnist  # noqa: PLC0415
        (Xtr, ytr), _ = mnist.load_data()
        return Xtr.reshape(len(Xtr), -1).astype(np.float64) / 255.0, ytr.astype(int)
    except Exception as e:                          # noqa: BLE001
        errs.append(f"keras: {e}")
    try:
        from sklearn.datasets import fetch_openml    # noqa: PLC0415
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        return d.data.astype(np.float64) / 255.0, d.target.astype(int)
    except Exception as e:                          # noqa: BLE001
        errs.append(f"sklearn: {e}")
    sys.exit("could not load MNIST:\n  " + "\n  ".join(errs))


# ---------------------------------------------------------------- spectral

def participation_ratio(eigs: np.ndarray) -> float:
    """PR = tr(L)^2 / tr(L^2). Scale-invariant, so normalisation is irrelevant."""
    e = np.clip(eigs, 0.0, None)
    return float(e.sum() ** 2 / (e ** 2).sum())


def pca_basis(X: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """Global PCA fit on all classes pooled -> (mean [784], components [784, D])."""
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD on a subsample is plenty for the basis and much faster
    idx = np.random.default_rng(0).choice(len(Xc), size=min(20000, len(Xc)),
                                          replace=False)
    _, _, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
    return mu, Vt[:D].T


def class_stats(F: np.ndarray, y: np.ndarray, center: str) -> dict[int, dict]:
    """Per-class covariance spectrum in the shared D-dim feature space."""
    out = {}
    pooled_mu = F.mean(0)
    for c in range(10):
        Fc = F[y == c]
        class_mu = Fc.mean(0)
        mu = class_mu if center == "per-class" else pooled_mu
        Z = Fc - mu
        C = (Z.T @ Z) / (len(Z) - 1)
        eigs = np.linalg.eigvalsh(C)[::-1]
        eigs = np.clip(eigs, 1e-12, None)
        norm = eigs / eigs.sum()                      # trace-normalised
        out[c] = {
            "n": len(Fc),
            "pr": participation_ratio(eigs),
            "tr_inv": float((1.0 / norm).sum()),      # sets the C2 channel
            # ALWAYS the class mean vs the pooled mean, whichever centering is
            # in use: this is the size of the mean channel that section 0's
            # algebra does not model, so it must not silently read 0.
            "mu_norm": float(np.linalg.norm(class_mu - pooled_mu)),
            "eigs": norm,
        }
    return out


# -------------------------------------------------------------------- report

def report(F_all, y, dims, center):
    print("\n" + "=" * 88)
    print(f"CENTERING = {center}")
    print("=" * 88)
    verdicts = {}
    for D in dims:
        F = F_all[:, :D]
        st = class_stats(F, y, center)
        prs = np.array([st[c]["pr"] for c in range(10)])
        tri = np.array([st[c]["tr_inv"] for c in range(10)])
        print(f"\n  D = {D}   (PR can range over (1, {D}])")
        print(f"    {'digit':>6} {'n':>7} {'PR':>8} {'PR/D':>7} "
              f"{'tr(Lam^-1)':>12} {'||mu-mu_pool||':>15}")
        for c in range(10):
            s = st[c]
            print(f"    {c:>6} {s['n']:>7} {s['pr']:>8.3f} {s['pr']/D:>7.3f} "
                  f"{s['tr_inv']:>12.1f} {s['mu_norm']:>15.3f}")
        span_pr = prs.max() / prs.min()
        span_b = tri.max() / tri.min()
        print(f"    {'-'*66}")
        print(f"    PR span (max/min)          : {span_pr:6.2f}x    "
              f"[synthetic config at D=4: 1.76x]")
        print(f"    tr(Lam^-1) span            : {span_b:6.1f}x    "
              f"= predicted span of C2 transition locations")
        print(f"    flattest / sharpest digit  : {int(prs.argmax())} / "
              f"{int(prs.argmin())}")
        verdicts[D] = (span_pr, span_b, int(prs.argmax()), int(prs.argmin()))
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--center", choices=["per-class", "pooled", "both"],
                    default="both")
    args = ap.parse_args()

    print("loading MNIST ...")
    X, y = load_mnist()
    print(f"  {X.shape[0]} images, {X.shape[1]} pixels")

    Dmax = max(args.dims)
    mu, V = pca_basis(X, Dmax)
    F_all = (X - mu) @ V
    kept = F_all.var(0).sum() / (X - mu).var(0).sum()
    print(f"  PCA to {Dmax} dims, {kept:.1%} of total variance retained")

    modes = (["per-class", "pooled"] if args.center == "both" else [args.center])
    allv = {m: report(F_all, y, args.dims, m) for m in modes}

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    for m in modes:
        for D, (span_pr, span_b, flat, sharp) in allv[m].items():
            ok = span_pr >= 1.5
            print(f"  {m:>10}  D={D:<3} PR span {span_pr:5.2f}x  "
                  f"{'USABLE' if ok else 'TOO FLAT'}   "
                  f"(forget candidate: digit {flat}, the flattest)")
    print("""
How to read this:

  PR span >= ~1.5x   the digits are spectrally distinguishable; the
                     spectral-geometry story has room to transfer, and the
                     flattest digit is the natural analogue of z3.

  PR span  < ~1.5x   every digit has effectively the same covariance shape.
                     Any AUC you then measure is NOT about spectral geometry;
                     it is about whatever else differs between the classes
                     (mostly the means). Say so, or pick a different axis.

  Compare the two centerings. If 'pooled' shows a much larger span than
  'per-class', the separation you are seeing is mean-driven, and section 0's
  algebra does not cover it -- the moments in theory.py assume the corruption
  is the only thing shifting the mean.

Next step is gated on this: see MNIST_DESIGN.md.
""")


if __name__ == "__main__":
    main()
