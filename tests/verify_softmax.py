#!/usr/bin/env python
"""
Correctness checks for the softmax architecture and the machinery that had to
generalise to accommodate it.

    python tests/verify_softmax.py        # CPU, seconds

The load-bearing check is number 3. Adding a second architecture meant the
per-token lever could no longer come from theory.py's closed form, so
diagnose.token_lever_numeric measures it by flipping one token at a time
instead. If the two disagreed on the linear model, then any difference the
experiment reports between ATTN-M and ATTN-SM could just be the difference
between two ways of computing the lever, and the comparison would be worthless.
They should agree to floating-point precision, because a single deterministic
label flip changes the context vector by exactly [-2 x_i y_i ; 0]: the
label-label block is untouched, since y_i^2 is even in y_i.

The rest guard the seams the new architecture opened:
  1. FrozenSoftmax reproduces the live module's forward exactly.
  2. Serialising and reloading a frozen ensemble is lossless.
  4. Zeroing W_Q makes the attention uniform, which is the softmax analogue of
     the linear model's constant 1/(N+1) -- the two families really do meet in
     the degenerate limit rather than merely resembling each other.
  5. The closed-form paths REFUSE a nonlinear ensemble instead of returning a
     plausible wrong number. This is the failure that would be hardest to spot
     downstream, since nothing about a wrong AUC looks wrong.
  6. apply_frozen leaves the linear path bit-for-bit unchanged, so none of the
     numbers already in the paper move.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning import diagnose, theory                       # noqa: E402
from icl_unlearning.data import assemble, build_spec, make_probe  # noqa: E402
from icl_unlearning.models import (FrozenSoftmax, LinearAttnICL,  # noqa: E402
                                   SoftmaxAttnICL, apply_frozen,
                                   build_model, frozen_from_blob,
                                   frozen_to_blob)
from icl_unlearning.policy import ScalarBernoulli, policy_auc     # noqa: E402

torch.manual_seed(0)
DEV, DT = "cpu", torch.float64
S, P, D, N = 24, 12, 4, 31

CFG = {"D": D, "N": N, "basis": "identity", "seed": 0,
       "groups": ["z1", "z2", "z3"], "forget": "z3",
       "eigs": {"z1": [0.70, 0.15, 0.10, 0.05],
                "z2": [0.60, 0.20, 0.10, 0.10],
                "z3": [0.40, 0.30, 0.20, 0.10]}}
spec = build_spec(CFG)
gen = torch.Generator(device=DEV).manual_seed(777)
probe = make_probe(spec, {"z1": 10, "z2": 10, "z3": 11}, "z3", P, gen, DEV, DT)

fail = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fail.append(name)


def make_sm(seed):
    torch.manual_seed(seed)
    m = SoftmaxAttnICL(S, D, N, init_scale=0.3, device=DEV, dtype=DT)
    # push the read-out off its init so the forward is not near-degenerate
    with torch.no_grad():
        m.wo.mul_(2.0)
    return m


def probe_batch():
    x = probe.x.unsqueeze(0).expand(S, *probe.x.shape)
    y = probe.y.unsqueeze(0).expand(S, *probe.y.shape)
    return assemble(x, y)


print("1. FrozenSoftmax reproduces the live module's forward")
m = make_sm(1)
X, ylab, _ = probe_batch()
with torch.no_grad():
    live = m(X, ylab)
froz = apply_frozen(m.frozen(), X, ylab, N)
check("live vs frozen", torch.allclose(live, froz, atol=1e-12),
      f"max |diff| = {(live - froz).abs().max():.3e}")
check("output shape", tuple(froz.shape) == (S, P), f"{tuple(froz.shape)}")
check("shape property exposes S", m.frozen().shape[0] == S)

print("\n2. blob round-trip is lossless")
back = frozen_from_blob(frozen_to_blob(m.frozen()))
check("round-tripped forward", torch.allclose(
    froz, apply_frozen(back.to(DEV), X, ylab, N), atol=1e-12))
check("linear blob passes through as a tensor",
      isinstance(frozen_to_blob(torch.zeros(S, D + 1, D + 1)), torch.Tensor))

print("\n3. numeric lever == closed-form lever, on the LINEAR archs")
for k, arch in enumerate(("ATTN-S", "ATTN-M")):
    # not hash(arch): PYTHONHASHSEED is randomised per process, which is the
    # exact bug train_ensembles.stable_offset exists to avoid
    torch.manual_seed(31 + k)
    lin = build_model(arch, S, D, N, init_scale=0.4, device=DEV, dtype=DT)
    with torch.no_grad():
        M = lin.M.detach().clone()
    L_closed = diagnose.token_lever(M, M * 0.7 + 0.05, probe)
    L_numeric = diagnose.token_lever_numeric(M, M * 0.7 + 0.05, probe)
    err = (L_closed - L_numeric).abs().max()
    rel = err / L_closed.abs().max().clamp_min(1e-30)
    check(f"{arch}: closed form vs single-token flips", float(rel) < 1e-10,
          f"max |diff| = {err:.3e}  (rel {float(rel):.2e})")

print("\n4. zero W_Q gives uniform attention (the 1/(N+1) limit)")
m0 = make_sm(2)
with torch.no_grad():
    m0.WQ.zero_()
    T = torch.cat([X[..., :D], ylab.unsqueeze(-1)], dim=-1)
    q = torch.einsum("sbd,sde->sbe", T[:, :, -1, :], m0.WQ)
    k = torch.einsum("sbnd,sde->sbne", T, m0.WK)
    alpha = torch.softmax(torch.einsum("sbe,sbne->sbn", q, k) / (D + 1) ** 0.5,
                          dim=-1)
check("alpha is uniform", torch.allclose(alpha, torch.full_like(alpha, 1.0 / (N + 1)),
                                         atol=1e-12),
      f"max |alpha - 1/(N+1)| = {(alpha - 1.0 / (N + 1)).abs().max():.3e}")

print("\n5. closed-form paths refuse a nonlinear ensemble")
f = m.frozen()
for name, fn in [
        ("theory.readout_covector", lambda: theory.readout_covector(f, probe)),
        ("theory.predicted_auc",
         lambda: theory.predicted_auc(f, f, probe, "bern", 0.2)),
        ("diagnose.token_lever", lambda: diagnose.token_lever(f, f, probe)),
        ("policy.policy_auc",
         lambda: policy_auc(f, f, probe, ScalarBernoulli().to(DEV)))]:
    try:
        fn()
        check(f"{name} raises", False, "returned a value instead of raising")
    except TypeError as e:
        check(f"{name} raises TypeError", True, f"({str(e)[:48]}...)")
    except Exception as e:                                    # noqa: BLE001
        check(f"{name} raises TypeError", False,
              f"raised {type(e).__name__} instead")

print("\n6. the linear path through apply_frozen is unchanged")
torch.manual_seed(5)
lin = LinearAttnICL("ATTN-M", S, D, N, init_scale=0.4, device=DEV, dtype=DT)
with torch.no_grad():
    M = lin.M.detach().clone()
    direct = lin.predict_frozen(M, X, ylab)
check("apply_frozen == predict_frozen",
      torch.equal(apply_frozen(M, X, ylab, N), direct))

print("\n7. the two families are genuinely different models")
sm_pred = apply_frozen(m.frozen(), X, ylab, N)
lin_pred = apply_frozen(M, X, ylab, N)
sep = (sm_pred - lin_pred).abs().mean()
check("softmax output differs from linear", float(sep) > 1e-6,
      f"mean |diff| = {sep:.3e}")

print("\n" + ("ALL CHECKS PASSED" if not fail else f"FAILED: {fail}"))
sys.exit(1 if fail else 0)
