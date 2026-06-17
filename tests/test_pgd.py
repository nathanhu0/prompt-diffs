"""Spec audit for the vendored Geisler PGD optimizer (optimize/pgd_geisler.py):
the projection math + the anneal / LR-coupling / discretization that the
faithfulness rests on. CPU-only, deterministic. Run:

    PYTHONPATH=. uv run python tests/test_pgd.py

We test the methods directly on a bare instance (GeislerPGD.__new__ + the few
attributes each method reads) so no model / tokenizer weights are needed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from optimize.pgd_geisler import GeislerPGD, tsallis_q2


def _proj_instance():
    inst = GeislerPGD.__new__(GeislerPGD)
    inst.eps = torch.finfo(torch.float32).eps
    inst.disallowed_tokens = None
    inst.simplex_proj_method = "sort"
    inst.tsallis_q2_proj_iter = 1
    inst.tsallis_exclude_already_zero = True
    return inst


# --------------------------------------------------------------------------
# Simplex projection (Duchi/Blondel sort-based).
# --------------------------------------------------------------------------
def ref_simplex_row(v):
    """Slow, obviously-correct Duchi (2008) projection of one vector onto the
    probability simplex — the oracle for _simplex_sort."""
    u = sorted(v.tolist(), reverse=True)
    cssum, theta = 0.0, 0.0
    for j, uj in enumerate(u, 1):
        cssum += uj
        if uj - (cssum - 1.0) / j > 0:
            theta = (cssum - 1.0) / j
    return torch.tensor([max(vi - theta, 0.0) for vi in v.tolist()])


def test_simplex_sort_matches_reference():
    torch.manual_seed(1)
    x = torch.rand(10, 40) * 2 + 0.5      # rows with sum > 1 so projection bites
    p = GeislerPGD._simplex_sort(x)
    for i in range(x.shape[0]):
        ref = ref_simplex_row(x[i])
        assert torch.allclose(p[i], ref, atol=1e-5), \
            f"row {i} mismatch vs Duchi reference (max {(p[i]-ref).abs().max():.2e})"
    print("OK _simplex_sort matches slow Duchi reference")


def test_simplex_projection_invariants():
    inst = _proj_instance()
    torch.manual_seed(0)
    p = inst.simplex_projection(torch.randn(7, 50) * 3)
    s = p.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), f"rows must sum to 1: {s}"
    assert torch.all(p >= -1e-6), "rows must be non-negative"
    print("OK simplex_projection: rows >= 0 and sum to 1")


# --------------------------------------------------------------------------
# Tsallis-q2 entropy + the ceiling projection.
# --------------------------------------------------------------------------
def test_tsallis_q2_endpoints():
    V = 100
    onehot = torch.zeros(1, V); onehot[0, 3] = 1.0
    uniform = torch.full((1, V), 1.0 / V)
    assert abs(float(tsallis_q2(onehot))) < 1e-6, "Gini(one-hot) must be 0"
    assert abs(float(tsallis_q2(uniform)) - (1 - 1.0 / V)) < 1e-6, "Gini(uniform) = 1 - 1/V"
    print("OK tsallis_q2 endpoints (one-hot=0, uniform=1-1/V)")


def test_entropy_ceiling_converges_and_monotonic():
    """The canonical projection is iter=1 (a per-step nudge), so ONE call does
    NOT hit the ceiling — but repeated application (mimicking per-optimization-
    step use) must drive Gini to <= ceiling, and a larger entropy_factor must
    give a lower ceiling / sharper result. This is the faithful behavior; the
    old impl's overshoot+20-iter hard-enforcement was MORE aggressive than
    canonical."""
    inst = _proj_instance()
    torch.manual_seed(3)
    finals = {}
    for ef in (0.2, 0.5, 0.8):
        p = inst.simplex_projection(
            torch.full((3, 200), 1.0 / 200) + 1e-3 * torch.randn(3, 200))
        for _ in range(200):
            p = inst.tsallis_q2_projection(p, ef)
        s = p.sum(-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4), "stays on simplex"
        assert torch.all(p >= -1e-6), "stays non-negative"
        gmax = float(tsallis_q2(p).max())
        ceil = (1 - ef) * (200 - 1) / 200
        assert gmax <= ceil + 1e-2, f"ef={ef}: converged Gini {gmax:.3f} exceeds ceiling {ceil:.3f}"
        finals[ef] = gmax
    assert finals[0.2] > finals[0.5] > finals[0.8], \
        f"larger entropy_factor must give a sharper (lower-Gini) result: {finals}"
    print(f"OK entropy ceiling converges <= target and is monotonic in ef ({finals})")


def test_entropy_one_call_is_a_soft_nudge():
    """A single iter=1 call only NUDGES toward the ceiling (does not enforce it),
    but moves in the right direction (Gini decreases)."""
    inst = _proj_instance()
    torch.manual_seed(4)
    p0 = inst.simplex_projection(
        torch.full((3, 200), 1.0 / 200) + 1e-3 * torch.randn(3, 200))
    p1 = inst.tsallis_q2_projection(p0.clone(), 0.8)
    assert float(tsallis_q2(p1).max()) < float(tsallis_q2(p0).max()), \
        "one projection call must reduce Gini (nudge toward the ceiling)"
    print("OK entropy projection: one iter=1 call is a soft nudge (not enforcement)")


# --------------------------------------------------------------------------
# Entropy-factor anneal (linear ramp init 0 -> end over duration, held after).
# --------------------------------------------------------------------------
def test_anneal_entropy_factor_ramp():
    inst = GeislerPGD.__new__(GeislerPGD)
    inst._entropy_factor_max = 0.4
    inst.anneal_duration = 100
    inst.anneal_entropy_factor(0);   assert abs(inst.entropy_factor - 0.0) < 1e-9
    inst.anneal_entropy_factor(50);  assert abs(inst.entropy_factor - 0.2) < 1e-9
    inst.anneal_entropy_factor(100); assert abs(inst.entropy_factor - 0.4) < 1e-9
    inst.anneal_entropy_factor(500); assert abs(inst.entropy_factor - 0.4) < 1e-9, "held after duration"
    print("OK anneal_entropy_factor ramps 0 -> 0.4 over duration then holds")


# --------------------------------------------------------------------------
# Dynamic entropy factor: LR coupling (base_lr = max(base, eta_min)).
# --------------------------------------------------------------------------
def test_dynamic_entropy_factor_lr_coupling():
    inst = GeislerPGD.__new__(GeislerPGD)
    inst.entropy_factor = 0.4
    inst.entropy_factor_scale_by_relaxation_gap = 0.0   # isolate the LR coupling
    inst.entropy_factor_alternate_scheduler = True

    param = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([param], lr=0.11)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=2),
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3, eta_min=0.325)],
        milestones=[2])
    for _ in range(4):
        opt.step(); sched.step()
    last_lr = sched.get_last_lr()[0]
    base_lr = max(0.11, 0.325)                          # eta_min > base -> floor is 0.325
    ef = inst.dynamic_entropy_factor(None, sched)
    expected = 0.4 * last_lr / base_lr
    assert abs(float(ef) - expected) < 1e-6, f"got {float(ef)}, expected {expected}"
    print(f"OK dynamic_entropy_factor couples to LR (ef=0.4*{last_lr:.3f}/{base_lr:.3f})")


# --------------------------------------------------------------------------
# Discretize: argmax + decode/re-encode round-trip.
# --------------------------------------------------------------------------
class RoundTripTok:
    """Space-separated single chars so decode/encode round-trip exactly."""
    def __init__(self, V=40):
        self.vocab_size = V
        self.chars = [chr(ord("a") + i) for i in range(V)]
    def decode(self, ids, **kwargs):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(self.chars[i] for i in ids)
    def __call__(self, s, add_special_tokens=False, **kwargs):
        toks = [self.chars.index(c) for c in s.split()]
        return type("E", (), {"input_ids": toks})()


def test_discretize_roundtrip():
    inst = GeislerPGD.__new__(GeislerPGD)
    inst.L = 5
    inst.tokenizer = RoundTripTok(40)
    V = 40
    target = torch.tensor([3, 9, 17, 24, 31])
    S = torch.zeros(1, 5, V)
    S[0, torch.arange(5), target] = 5.0                 # argmax = target
    ids = inst.discretize(S)
    assert ids.shape == (1, 5)
    assert torch.equal(ids[0], target), f"discretize must recover the argmax ids, got {ids[0]}"
    print("OK discretize returns argmax ids under a round-trip-stable tokenizer")


# --------------------------------------------------------------------------
# Per-row gradient clip (token_norm).
# --------------------------------------------------------------------------
def test_clip_grad_token_norm():
    inst = GeislerPGD.__new__(GeislerPGD)
    inst.grad_clip_value = 2.0
    inst.eps = torch.finfo(torch.float32).eps
    g = torch.tensor([[3.0, 4.0],          # norm 5 -> clipped to 2
                      [0.3, 0.4]])          # norm 0.5 -> untouched
    inst.clip_grad_(g)
    assert abs(float(g[0].norm()) - 2.0) < 1e-4, "over-norm row clipped to grad_clip_value"
    assert abs(float(g[1].norm()) - 0.5) < 1e-4, "under-norm row untouched"
    print("OK clip_grad_ token_norm clips per-row to grad_clip_value")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} PGD SPEC TESTS PASSED")
