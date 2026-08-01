#!/usr/bin/env python
"""
Known-answer checks for diagnose.py, in the style of tests/verify_bern.py.

The targeting and polarisation numbers are the evidence for WHY a conditional
policy beats a scalar one, so they need to be right before any claim rests on
them. Six checks, all with answers derivable by hand:

  1. A uniform theta must give T = 1 and R = 1 EXACTLY. This is the whole design
     of the two statistics -- the scalar policy is the null, so anything other
     than 1 there is a bug in the metric, not a finding.
  2. A worked three-token case with T and R computed by hand.
  3. R <= 1 always, by Jensen. Checked over random draws.
  4. Gini is 0 for equal levers and (n-1)/n for a single spike.
  5. Spearman is +1 when theta tracks |L| and -1 when it anti-tracks.
  6. The motivating claim itself: a greedy policy that spends its whole budget
     on the largest-|L| tokens must score T > 1 at equal average budget. If this
     failed, T would not be measuring targeting at all.

Run:  python tests/verify_diagnose.py
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning.diagnose import _gini, describe_policy   # noqa: E402

torch.manual_seed(0)
P, n_f = 64, 11
L = torch.randn(P, n_f)

fail = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {name:<44} {got:+.10f}  (want {want:+.6f})")
    if not ok:
        fail.append(name)


print("1. uniform theta is the null: T and R must be exactly 1")
for t in (0.05, 0.3, 0.5, 0.9):
    d = describe_policy(torch.full((P, n_f), t), L)
    check(f"theta={t}  T", d["targeting_T"], 1.0, 1e-9)
    check(f"theta={t}  R", d["polarisation_R"], 1.0, 1e-9)

print("\n2. worked case: L=[1,2,7], theta=[0,0,0.6]")
Lh = torch.tensor([[1.0, 2.0, 7.0]])
th = torch.tensor([[0.0, 0.0, 0.6]])
d = describe_policy(th, Lh, n_buckets=3)
# thetabar = 0.2, sum L = 10, so a scalar at the same budget shifts by 0.2*10 = 2
# while this policy shifts by 0.6*7 = 4.2
check("T", d["targeting_T"], 4.2 / 2.0, 1e-6)
# mean theta(1-theta) = (0 + 0 + 0.24)/3 = 0.08 ; thetabar(1-thetabar) = 0.16
check("R", d["polarisation_R"], 0.08 / 0.16, 1e-6)
# buckets are ordered by |L| ascending, so the flipped token lands in the last
check("theta in top-|L| bucket", d["theta_by_abs_lever_bucket"][-1], 0.6, 1e-6)
check("theta in bottom-|L| bucket", d["theta_by_abs_lever_bucket"][0], 0.0, 1e-6)

print("\n3. Jensen: R <= 1 for any theta")
worst = max(describe_policy(torch.rand(P, n_f), L)["polarisation_R"]
            for _ in range(200))
if worst <= 1.0 + 1e-9:
    print(f"  ok   max R over 200 random policies = {worst:.6f}  (<= 1)")
else:
    print(f"  FAIL max R over 200 random policies = {worst:.6f}  (> 1)")
    fail.append("Jensen")

print("\n4. Gini")
check("equal levers", float(_gini(torch.ones(1, n_f))), 0.0, 1e-9)
spike = torch.zeros(1, n_f)
spike[0, -1] = 1.0
check("single spike", float(_gini(spike)), (n_f - 1) / n_f, 1e-9)

print("\n5. Spearman against |L|")
d = describe_policy(L.abs(), L)
check("theta = |L|", d["spearman_theta_abs_lever"], 1.0, 1e-9)
d = describe_policy(-L.abs(), L)
check("theta = -|L|", d["spearman_theta_abs_lever"], -1.0, 1e-9)

print("\n6. greedy targeting must register as T > 1")
budget = 0.2
k = int(round(budget * n_f))
th = torch.zeros(P, n_f)
th.scatter_(1, L.abs().argsort(dim=1, descending=True)[:, :k], 1.0)
d = describe_policy(th, L)
print(f"   greedy T = {d['targeting_T']:.3f}   R = {d['polarisation_R']:.3f}")
if not d["targeting_T"] > 1.0:
    fail.append("greedy T > 1")
    print("  FAIL greedy policy did not register as targeting")
else:
    print("  ok  greedy policy registers as targeting")
if not d["polarisation_R"] < 1e-9:
    fail.append("greedy R ~ 0")
    print("  FAIL a fully polarised policy should give R = 0")
else:
    print("  ok  fully polarised policy gives R = 0")

print("\n" + ("ALL CHECKS PASSED" if not fail else f"FAILED: {fail}"))
sys.exit(1 if fail else 0)
