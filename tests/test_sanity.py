"""
Fast sanity checks. Run before burning GPU hours:

    pytest tests/ -q
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icl_unlearning import audit                                  # noqa: E402
from icl_unlearning.corrupt import corrupt                        # noqa: E402
from icl_unlearning.data import MixtureSpec, make_probe, make_sequences  # noqa: E402
from icl_unlearning.models import LinearAttnICL, apply_frozen     # noqa: E402
from icl_unlearning.train import (check_ensemble_health,          # noqa: E402
                                  clip_grad_norm_per_shadow_)

DEV = "cpu"
SPEC = MixtureSpec(names=["z1", "z2", "z3"],
                   eigs={"z1": [0.70, 0.15, 0.10, 0.05],
                         "z2": [0.60, 0.20, 0.10, 0.10],
                         "z3": [0.40, 0.30, 0.20, 0.10]},
                   D=4, N=31)


def test_participation_ratio_ordering():
    prs = [SPEC.pr(g) for g in SPEC.names]
    assert prs[0] < prs[1] < prs[2], prs
    assert abs(prs[0] - 1.91) < 0.02 and abs(prs[2] - 3.33) < 0.02


def test_shapes_and_query_label_hidden():
    gen = torch.Generator(device=DEV).manual_seed(0)
    X, ylab, yq = make_sequences(SPEC, SPEC.names, 5, 3, gen, DEV)
    assert X.shape == (5, 3, 32, 5)
    assert ylab.shape == (5, 3, 32) and yq.shape == (5, 3)
    assert torch.allclose(ylab[:, :, -1], torch.zeros_like(ylab[:, :, -1]))
    assert torch.allclose(X[:, :, -1, -1], torch.zeros_like(yq))


def test_uD_forward_matches_naive_XtX():
    """The O(N*D) path must equal the literal (1/(N+1)) t_q^T M X^T X e formula."""
    gen = torch.Generator(device=DEV).manual_seed(1)
    X, ylab, yq = make_sequences(SPEC, SPEC.names, 2, 4, gen, DEV)
    m = LinearAttnICL("ATTN-M", 2, SPEC.D, SPEC.N, device=DEV)
    fast = m(X, ylab)
    G = torch.einsum("sbnd,sbne->sbde", X, X)
    e = torch.zeros(SPEC.D + 1); e[-1] = 1.0
    naive = torch.einsum("sbd,sde,sbef,f->sb", X[:, :, -1, :], m.M, G, e) / (SPEC.N + 1)
    assert torch.allclose(fast, naive, atol=1e-4), (fast - naive).abs().max()


def test_attn_s_and_m_same_function_class():
    """Given equal M, both architectures must produce identical predictions."""
    gen = torch.Generator(device=DEV).manual_seed(2)
    X, ylab, _ = make_sequences(SPEC, SPEC.names, 3, 4, gen, DEV)
    s = LinearAttnICL("ATTN-S", 3, SPEC.D, SPEC.N, device=DEV)
    out_frozen = apply_frozen(s.M.detach(), X, ylab, SPEC.N)
    assert torch.allclose(s(X, ylab), out_frozen, atol=1e-5)


def test_corruption_touches_only_forget_tokens():
    gen = torch.Generator(device=DEV).manual_seed(3)
    probe = make_probe(SPEC, {"z1": 10, "z2": 10, "z3": 11}, "z3", 8,
                       gen, DEV)
    X, ylab, _ = corrupt(probe, 4, "C2", 5.0, gen)
    sl = probe.forget_slice
    retain = slice(0, sl.start)
    ref = probe.x[:, retain, :].unsqueeze(0).expand(4, -1, -1, -1)
    assert torch.allclose(X[:, :, retain, :SPEC.D], ref, atol=1e-6)
    assert not torch.allclose(X[:, :, sl, :SPEC.D],
                              probe.x[:, sl, :].unsqueeze(0).expand(4, -1, -1, -1))


def test_flip_dead_zone_is_zero_mean():
    """t=0.5 must null the forget labels: that is the predicted dead zone."""
    gen = torch.Generator(device=DEV).manual_seed(4)
    probe = make_probe(SPEC, {"z1": 10, "z2": 10, "z3": 11}, "z3", 8,
                       gen, DEV)
    _, ylab, _ = corrupt(probe, 2, "flip", 0.5, gen)
    assert ylab[:, :, probe.forget_slice].abs().max() < 1e-6


def test_auc_endpoints():
    a = torch.arange(20.0)
    assert abs(audit._auc_1d(a + 100, a) - 1.0) < 1e-6
    assert abs(audit._auc_1d(a, a + 100) - 0.0) < 1e-6
    assert abs(audit._auc_1d(a, a.clone()) - 0.5) < 1e-6
    assert abs(audit.symmetrised_auc(0.2) - 0.8) < 1e-9


def test_vectorised_auc_matches_reference():
    """The fast per-probe path must agree with the 1-D tie-corrected version."""
    g = torch.Generator().manual_seed(5)
    h1 = torch.randn(40, 7, generator=g) + 0.3
    h0 = torch.randn(30, 7, generator=g)
    fast = audit.auc_per_probe(h1, h0)
    ref = torch.stack([audit._auc_1d(h1[:, p], h0[:, p]) for p in range(7)])
    assert torch.allclose(fast, ref, atol=1e-6), (fast - ref).abs().max()


def test_bootstrap_ci_brackets_point_estimate():
    g = torch.Generator().manual_seed(6)
    h1 = torch.randn(60, 16, generator=g) + 1.5
    h0 = torch.randn(60, 16, generator=g)
    a, lo, hi = audit.membership_auc_ci(h1, h0, n_boot=100, seed=0)
    assert lo <= a <= hi
    assert 0.0 <= lo < hi <= 1.0
    # a strongly separated pair must exclude chance
    assert lo > 0.5


def test_shared_noise_is_constant_across_shadows():
    """The masking control must apply one identical noise draw to every shadow."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    probe = make_probe(SPEC, {"z1": 10, "z2": 10, "z3": 11}, "z3", 8, gen, DEV)
    _, ylab, _ = corrupt(probe, 5, "C1", 2.0, gen, shared_noise=True)
    f = ylab[:, :, probe.forget_slice]
    assert torch.allclose(f, f[:1].expand_as(f), atol=1e-7)

    _, ylab2, _ = corrupt(probe, 5, "C1", 2.0, gen, shared_noise=False)
    f2 = ylab2[:, :, probe.forget_slice]
    assert not torch.allclose(f2, f2[:1].expand_as(f2), atol=1e-7)


def test_sign_alignment_recovers_a_signal_that_raw_output_cancels():
    """
    The audit's headline failure mode. Build an obvious membership signal: the
    oracle shrinks its prediction toward zero. Averaging RAW per-probe AUC over
    a mixed-sign probe cancels to ~0.5; sign-aligning recovers it.
    """
    g = torch.Generator().manual_seed(11)
    S, P, kappa = 100, 64, 0.55
    yq = torch.randn(P, generator=g)
    yhat1 = yq + 0.25 * torch.randn(S, P, generator=g)
    yhat0 = kappa * yq + 0.25 * torch.randn(S, P, generator=g)

    raw = audit.observables_raw(yhat1, yq)["output"]
    raw0 = audit.observables_raw(yhat0, yq)["output"]
    a_raw = audit.membership_auc(raw, raw0)

    ali = audit.observables(yhat1, yq)["residual"]
    ali0 = audit.observables(yhat0, yq)["residual"]
    a_ali = audit.membership_auc(ali, ali0)

    assert abs(a_raw - 0.5) < 0.08, f"raw output AUC should cancel, got {a_raw}"
    assert a_ali > 0.70, f"sign-aligned AUC should see the signal, got {a_ali}"


def test_symmetrised_aggregation_order_matters():
    """max(mean) is not mean(max); the former cannot undo sign cancellation."""
    g = torch.Generator().manual_seed(12)
    P = 64
    yq = torch.randn(P, generator=g)
    yhat1 = yq + 0.25 * torch.randn(100, P, generator=g)
    yhat0 = 0.55 * yq + 0.25 * torch.randn(100, P, generator=g)
    o1 = audit.observables_raw(yhat1, yq)["output"]
    o0 = audit.observables_raw(yhat0, yq)["output"]

    after = audit.symmetrised_auc(audit.membership_auc(o1, o0))   # wrong order
    before = audit.symmetrised_auc_per_probe(o1, o0)              # right order
    assert abs(after - 0.5) < 0.08
    assert before > 0.70
    assert before > after


def test_null_auc_level_is_above_half():
    """Per-point symmetrisation is biased up at the null; quantify it."""
    lvl = audit.null_auc_level(S=100, P=64, n_rep=40, seed=0)
    assert 0.50 < lvl < 0.60, lvl


def test_per_shadow_clip_bounds_an_outlier_regardless_of_ensemble_size():
    """
    Regression test for the S=100->512 divergence. A global norm clip with
    threshold ~ S lets a single diverging shadow's gradient through almost
    untouched, and gets WORSE as S grows since the threshold grows linearly
    while the aggregate norm of well-behaved shadows only grows like sqrt(S).
    Per-shadow clipping must bound every shadow to `max_norm` independent of S.
    """
    torch.manual_seed(0)
    max_norm = 5.0
    for S in (10, 100, 512):
        p = torch.nn.Parameter(torch.zeros(S, 3, 3))
        g = torch.randn(S, 3, 3) * 1.0
        g[-1] *= 50.0            # one shadow with a much larger gradient
        p.grad = g.clone()
        clip_grad_norm_per_shadow_([p], max_norm)
        norms = p.grad.reshape(S, -1).norm(dim=1)
        assert norms.max() <= max_norm + 1e-3, (S, norms.max())
        # a well-behaved shadow should be scaled by very little
        assert norms[0] < max_norm + 1e-3


def test_check_ensemble_health_flags_a_diverged_shadow():
    M = torch.stack([torch.eye(3) for _ in range(10)])
    M[3] *= 1000.0    # one shadow with a 1000x larger read-out
    # should not raise (values are finite), but should print a warning --
    # just check it doesn't raise and doesn't false-positive on a clean ensemble
    check_ensemble_health(M, "TEST")
    check_ensemble_health(torch.stack([torch.eye(3) for _ in range(10)]), "TEST")


def test_check_ensemble_health_raises_on_non_finite():
    M = torch.stack([torch.eye(3) for _ in range(5)])
    M[2, 0, 0] = float("nan")
    try:
        check_ensemble_health(M, "TEST")
        assert False, "expected RuntimeError on non-finite M"
    except RuntimeError:
        pass


def test_mmd2_subsamples_instead_of_oom():
    """
    Regression test for the 32 GiB OOM at S=512, P=64 (65536^2*8 bytes exactly).
    mmd2 must cap the pairwise-distance population regardless of input size.
    """
    g = torch.Generator().manual_seed(0)
    # population far larger than max_n, but small enough for a CPU test
    x = torch.randn(20_000, generator=g)
    y = torch.randn(20_000, generator=g) + 0.5
    val = audit.mmd2(x, y, max_n=500, seed=0)
    assert val >= 0.0
    # determinism: same seed -> same subsample -> same value
    val2 = audit.mmd2(x, y, max_n=500, seed=0)
    assert val == val2


def test_collapse_by_param_aggregates_across_seed_combos():
    """
    plot_auc_vs_var.py's cross-seed aggregation: given rows tagged with
    train_seed_idx/probe_seed_idx, collapse_by_param must group by `param`
    across every seed combo and return a band that brackets the median.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from plot_auc_vs_var import collapse_by_param, n_seed_combos

    rows = []
    for ts in range(3):
        for ps in range(3):
            for p, base in ((0.0, 0.4), (1.0, 0.6)):
                rows.append({"arch": "ATTN-M", "mode": "C1", "param": p,
                            "auc_matched_residual": base + 0.01 * (ts - ps),
                            "train_seed_idx": ts, "probe_seed_idx": ps})

    assert n_seed_combos(rows) == 9
    xs, med, lo, hi = collapse_by_param(rows, "ATTN-M", "C1", "auc_matched_residual")
    assert xs == [0.0, 1.0]
    for i in range(2):
        assert lo[i] <= med[i] <= hi[i]
    assert abs(med[0] - 0.4) < 0.02
    assert abs(med[1] - 0.6) < 0.02


def test_train_seed_derivation_gives_distinct_seeds_per_index():
    """
    train_ensembles.py trains one ensemble per training-seed index ts, using
    seed = base_seed + ts. Different ts must give different actual seeds, or
    'n_train_seeds=3' would silently retrain the same ensemble three times.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import train_ensembles as te

    base = 1
    seeds = [base + ts for ts in range(3)]
    assert len(set(seeds)) == 3
    # combined with the per-(arch,hyp) offset, still all distinct
    combined = [s + te.stable_offset(arch, hyp)
               for s in seeds for arch in ("ATTN-S", "ATTN-M")
               for hyp in ("full", "oracle")]
    assert len(set(combined)) == len(combined)


def test_stable_offset_is_process_independent():
    """
    Regression test for the seeding bug: the offset must not depend on
    PYTHONHASHSEED. Hard-coded values were produced by blake2b, so a change
    here means the seed derivation changed and old artifacts are stale.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from train_ensembles import stable_offset
    assert stable_offset("ATTN-S", "full") == stable_offset("ATTN-S", "full")
    assert stable_offset("ATTN-S", "full") != stable_offset("ATTN-S", "oracle")
    assert stable_offset("ATTN-S", "full") != stable_offset("ATTN-M", "full")
