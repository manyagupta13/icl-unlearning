#!/usr/bin/env python
"""
Train every shadow ensemble for one task and cache them.

    python scripts/train_ensembles.py --config configs/regression.yaml

Produces artifacts/ensembles_{task}.pt containing, for each architecture:
    ("ATTN-S", "full")    trained on all groups
    ("ATTN-S", "oracle")  retrained on retain groups only
    ...

This is the only expensive step. The sweep reuses these.
"""
import argparse
import pathlib
import sys

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning.data import MixtureSpec               # noqa: E402
from icl_unlearning.train import per_group_mse, train_ensemble   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    task, d, tr = cfg["task"], cfg["data"], cfg["train"]
    dev = args.device
    print(f"device={dev}  task={task}")

    spec = MixtureSpec(names=d["groups"], eigs=d["eigs"], D=d["D"], N=d["N"],
                       basis=d["basis"], seed=d["seed"])
    for g in spec.names:
        print(f"  {g}: PR = {spec.pr(g):.2f}")

    retain = [g for g in d["groups"] if g != d["forget"]]
    out = {}
    for arch in tr["archs"]:
        for hyp, groups in [("full", d["groups"]), ("oracle", retain)]:
            print(f"[{arch} | {hyp}] groups={groups}")
            M, trace = train_ensemble(
                spec, groups, arch, task,
                S=tr["n_shadows"], batch_per_shadow=tr["batch_per_shadow"],
                steps=tr["steps"], lr=tr["lr"], momentum=tr["momentum"],
                optim=tr["optim"], grad_clip=tr["grad_clip"],
                init_scale=tr["init_scale"],
                seed=tr["seed"] + hash((arch, hyp)) % 10_000,
                device=dev)
            out[f"{arch}|{hyp}|M"] = M.cpu()
            out[f"{arch}|{hyp}|trace"] = trace.cpu()

        # sanity: per-group MSE should track participation ratio
        gen = torch.Generator(device=dev).manual_seed(999)
        mse = per_group_mse(spec, out[f"{arch}|full|M"].to(dev), arch, task,
                            tr["n_shadows"], gen, dev)
        print(f"  per-group MSE: " + "  ".join(f"{g}={v:.4f}" for g, v in mse.items()))
        out[f"{arch}|per_group_mse"] = torch.tensor([mse[g] for g in spec.names])

    adir = pathlib.Path(cfg["paths"]["artifacts"])
    adir.mkdir(parents=True, exist_ok=True)
    path = adir / f"ensembles_{task}.pt"
    torch.save({"cfg": cfg, **out}, path)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
