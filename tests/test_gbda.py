"""Spec + loop tests for GBDA (optimize/gbda.py), a clean-room reimplementation
of Guo et al. 2021 (arXiv:2104.13733). CPU-only, deterministic. Run:

    PYTHONPATH=. uv run python tests/test_gbda.py

The pure helpers (gumbel-softmax sample, log_perplexity fluency, log_coeffs
init, hard-gumbel argmax) are checked against obviously-correct references; the
recovery loop is validated on the synthetic MSE objective from
tests/test_optimizer_loops.py (recover a KNOWN target token sequence).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F

from optimize.gbda import (gumbel_like, gumbel_softmax_coeffs, log_perplexity,
                           init_log_coeffs, run_gbda)
from tests.test_optimizer_loops import _setup, _overlap


# --------------------------------------------------------------------------
# Gumbel-softmax sample: rows are valid simplices; sharpens as tau -> 0.
# --------------------------------------------------------------------------
def test_gumbel_softmax_coeffs_is_simplex():
    torch.manual_seed(0)
    logits = torch.randn(7, 50) * 3
    noise = gumbel_like(logits)
    c = gumbel_softmax_coeffs(logits, tau=1.0, noise=noise)
    s = c.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), f"rows must sum to 1: {s}"
    assert torch.all(c >= 0), "rows must be non-negative"
    # lower tau -> sharper (lower entropy) for the same noise draw
    c_sharp = gumbel_softmax_coeffs(logits, tau=0.1, noise=noise)
    ent = -(c * c.clamp_min(1e-12).log()).sum(-1).mean()
    ent_sharp = -(c_sharp * c_sharp.clamp_min(1e-12).log()).sum(-1).mean()
    assert ent_sharp < ent, "smaller tau must sharpen the sample (lower entropy)"
    print("OK gumbel_softmax_coeffs: rows on the simplex; tau sharpens")


def test_hard_gumbel_is_argmax_of_perturbed_logits():
    """The end-of-run hard extraction uses argmax(logits + g); confirm that
    matches F.gumbel_softmax(hard=True).argmax (softmax is monotone, so the
    one-hot lands on argmax(logits + g))."""
    torch.manual_seed(1)
    logits = torch.randn(5, 30) * 2
    noise = gumbel_like(logits)
    ours = (logits + noise).argmax(-1)
    ref = torch.softmax((logits + noise) / 1.0, dim=-1).argmax(-1)
    assert torch.equal(ours, ref), "hard-gumbel argmax must equal softmax argmax"
    print("OK hard-gumbel sample == argmax(logits + gumbel noise)")


# --------------------------------------------------------------------------
# log_perplexity: soft causal CE matches an explicit reference.
# --------------------------------------------------------------------------
def test_log_perplexity_matches_manual():
    torch.manual_seed(2)
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V)
    coeffs = torch.softmax(torch.randn(B, T, V), dim=-1)
    got = log_perplexity(logits, coeffs)

    # Obvious reference: -mean over (b, t<T-1) of sum_v coeffs[b,t+1,v]*logp[b,t,v]
    logp = F.log_softmax(logits, dim=-1)
    acc, n = 0.0, 0
    for b in range(B):
        for t in range(T - 1):
            acc += float((coeffs[b, t + 1] * logp[b, t]).sum())
            n += 1
    ref = -acc / n
    assert abs(float(got) - ref) < 1e-5, f"log_perplexity {float(got):.6f} != ref {ref:.6f}"
    print("OK log_perplexity matches the manual soft causal-CE reference")


def test_log_perplexity_lower_when_coeffs_follow_logits():
    """A fluency sanity check: if the soft sequence follows what the LM predicts,
    perplexity is lower than for a sequence the LM finds unlikely."""
    torch.manual_seed(3)
    B, T, V = 1, 8, 20
    logits = torch.randn(B, T, V) * 4
    pred_next = logits[:, :-1].argmax(-1)                 # what the LM expects next
    fluent = F.one_hot(
        torch.cat([torch.zeros(B, 1, dtype=torch.long), pred_next], 1), V).float()
    rand = torch.softmax(torch.randn(B, T, V), dim=-1)
    assert float(log_perplexity(logits, fluent)) < float(log_perplexity(logits, rand)), \
        "an LM-consistent sequence must have lower log-perplexity than a random one"
    print("OK log_perplexity is lower for an LM-consistent sequence")


# --------------------------------------------------------------------------
# init_log_coeffs: argmax = init ids; mass concentrates with initial_coeff.
# --------------------------------------------------------------------------
def test_init_log_coeffs_anchors_at_init_ids():
    V = 100
    init_ids = [3, 50, 7, 99, 0]
    lc = init_log_coeffs(init_ids, V, initial_coeff=15.0, device="cpu")
    assert lc.shape == (len(init_ids), V)
    assert torch.equal(lc.argmax(-1), torch.tensor(init_ids)), "argmax must be the init ids"
    p = torch.softmax(lc, dim=-1)
    expected = torch.e ** 15 / (torch.e ** 15 + (V - 1))
    assert torch.allclose(p[torch.arange(len(init_ids)), torch.tensor(init_ids)],
                          torch.full((len(init_ids),), float(expected)), atol=1e-3), \
        "softmax mass at the init token must match e^c/(e^c+V-1)"
    # initial_coeff=0 -> uniform
    lc0 = init_log_coeffs(init_ids, V, initial_coeff=0.0, device="cpu")
    assert torch.allclose(torch.softmax(lc0, -1), torch.full((len(init_ids), V), 1.0 / V)), \
        "initial_coeff=0 must give a uniform init"
    print("OK init_log_coeffs anchors at init ids; mass set by initial_coeff")


# --------------------------------------------------------------------------
# Loop: recover a KNOWN target on the synthetic MSE objective.
# --------------------------------------------------------------------------
def test_gbda_recovers_known_target():
    """run_gbda on loss(z)=MSE(z, target_embeds) must drive the loss down and
    recover the target tokens. Toy-friendly settings (validating LOOP wiring,
    not canonical constants — those are the unit tests + source review):
    lam_perp=0 (no ref-LM forward; the synthetic FakeModel has none),
    initial_coeff=0 (uniform init — recovery has no clean input to warm-start at,
    and a uniform start is well-conditioned for the toy), allow_non_ascii (toy
    chars run past ASCII)."""
    E, tok, target_ids, obj, model = _setup()
    base = obj.hard_loss(tok.decode(torch.zeros(len(target_ids), dtype=torch.long)), "train")
    cfg = {"num_iters": 400, "lr": 0.3, "gumbel_tau": 1.0, "initial_coeff": 0.0,
           "lam_perp": 0.0, "gumbel_samples_per_step": 8, "mini_batch_size": 8,
           "train_batch_size": 16, "select_n": 16, "eval_chunk": 16, "eval_every": 10,
           "print_every": 100, "final_gumbel_samples": 30, "allow_non_ascii": True}
    res = run_gbda(obj, model, tok, E, cfg=cfg, seed=0)
    ov = _overlap(res["best_text"], target_ids, tok)
    print(f"  GBDA: select={res['best_select_score']:.4f} (base {base:.4f}) "
          f"recovered {ov}/{len(target_ids)} tokens")
    assert res["best_select_score"] < 0.5 * base, "GBDA must reduce the loss substantially"
    assert ov >= len(target_ids) - 1, f"GBDA must recover >= L-1 target tokens, got {ov}"
    print("OK run_gbda recovers the known target")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} GBDA TESTS PASSED")
