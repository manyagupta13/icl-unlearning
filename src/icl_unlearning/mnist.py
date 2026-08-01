"""
MNIST feature banks for MnistSpec.

Loads MNIST once, projects to D dimensions with a PCA fitted on the pooled data
(so every class lives in the same feature space), then builds one bank of
feature vectors per requested digit.

Two normalisations, both deliberate:

  centering   "class"  subtracts each digit's own mean. Keeps the MixtureSpec
                       contract that groups share mean 0 and differ only in
                       covariance, so spectral geometry remains the sole axis
                       of variation -- but digit identity is then reduced to
                       covariance shape.
              "pooled" subtracts one global mean. Digits stay digits, but the
                       group means now differ, a channel theory.py's closed
                       forms do not model.

  scale       every bank is trace-normalised so tr(Cov) = 1, matching
              MixtureSpec. Without this, E[y^2] would differ per digit and the
              corruption arms would be confounded with class energy.

Run scripts/mnist_pr_probe.py first: if the digits' participation ratios do not
spread, this port measures the input distribution rather than spectral geometry.
"""
from __future__ import annotations

import sys

import torch

_CACHE: dict = {}


def _load_raw():
    """-> (X [n, 784] float32 in [0,1], y [n] int64). Cached per process."""
    if "raw" in _CACHE:
        return _CACHE["raw"]
    errs = []
    try:
        from torchvision import datasets                       # noqa: PLC0415
        ds = datasets.MNIST(root="./data", train=True, download=True)
        X = ds.data.reshape(len(ds), -1).float() / 255.0
        y = ds.targets.long()
        _CACHE["raw"] = (X, y)
        return X, y
    except Exception as e:                                     # noqa: BLE001
        errs.append(f"torchvision: {e}")
    try:
        from tensorflow.keras.datasets import mnist            # noqa: PLC0415
        (Xtr, ytr), _ = mnist.load_data()
        X = torch.tensor(Xtr.reshape(len(Xtr), -1), dtype=torch.float32) / 255.0
        y = torch.tensor(ytr, dtype=torch.long)
        _CACHE["raw"] = (X, y)
        return X, y
    except Exception as e:                                     # noqa: BLE001
        errs.append(f"keras: {e}")
    try:
        from sklearn.datasets import fetch_openml              # noqa: PLC0415
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = torch.tensor(d.data, dtype=torch.float32) / 255.0
        y = torch.tensor(d.target.astype(int), dtype=torch.long)
        _CACHE["raw"] = (X, y)
        return X, y
    except Exception as e:                                     # noqa: BLE001
        errs.append(f"sklearn: {e}")
    sys.exit("could not load MNIST:\n  " + "\n  ".join(errs))


def load_feature_banks(digits: list[int], names: list[str], D: int,
                       center: str = "class", seed: int = 0
                       ) -> dict[str, torch.Tensor]:
    """-> {group name: [n_g, D] float32}, each trace-normalised to tr(Cov)=1."""
    key = (tuple(digits), tuple(names), D, center, seed)
    if key in _CACHE:
        return _CACHE[key]

    X, y = _load_raw()
    g = torch.Generator().manual_seed(seed)

    # PCA on the pooled data so all classes share one feature space
    pooled_mu = X.mean(0)
    Xc = X - pooled_mu
    idx = torch.randperm(len(Xc), generator=g)[:20000]
    _, _, Vt = torch.linalg.svd(Xc[idx], full_matrices=False)
    V = Vt[:D].T                                                # [784, D]
    F = Xc @ V                                                  # [n, D]
    pooled_F_mu = F.mean(0)

    banks = {}
    for name, dig in zip(names, digits):
        Fd = F[y == dig]
        mu = Fd.mean(0) if center == "class" else pooled_F_mu
        Z = Fd - mu
        cov = (Z.T @ Z) / (len(Z) - 1)
        tr = torch.diagonal(cov).sum().clamp_min(1e-12)
        banks[name] = (Z / tr.sqrt()).contiguous()              # tr(Cov) = 1
    _CACHE[key] = banks
    return banks
