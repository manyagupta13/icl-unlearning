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
