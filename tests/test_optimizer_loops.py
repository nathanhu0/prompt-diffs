"""Deterministic CPU integration test for the GCG and PGD optimization loops —
the clean-room "Milestone 0" (recover a KNOWN target). A synthetic objective
loss(z) = ||z - E[target_ids]||^2 with a SQUARE (invertible) embedding matrix
makes the optimum unique at the target token sequence, so both loops must drive
the loss down and recover the target tokens. Validates gradient -> step ->
project -> discretize -> converge without a GPU or real model.

    PYTHONPATH=. uv run python tests/test_optimizer_loops.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from optimize.gcg import run_gcg
from optimize.pgd import run_pgd


class CharTokenizer:
    """Space-separated single-token chars so decode/encode round-trip exactly
    (required by GCG's filter_retokenizable and slot-init check)."""
    def __init__(self, V):
        self.vocab_size = V
        self.chars = [chr(ord("a") + i) for i in range(V)]
        self.bos_token_id = None
        self.eos_token_id = V - 1
        self.pad_token_id = V - 1
        self.unk_token_id = None

    def decode(self, ids, **kwargs):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(self.chars[i] for i in ids)

    def encode(self, s, add_special_tokens=False):
        return [self.chars.index(c) for c in s.split()]

    def __call__(self, s, add_special_tokens=False, return_tensors=None):
        toks = self.encode(s)
        return type("Enc", (), {"input_ids": torch.tensor([toks])})()


class FakeModel:
    device = torch.device("cpu")


class RecoverObjective:
    """loss(z) = MSE(z, target_embeds); minimized iff the slot embeds equal the
    target token embeds. hard_loss scores a text by encoding -> embeds -> MSE."""
    def __init__(self, E, target_ids, tokenizer):
        self.E = E
        self.target_ids = target_ids
        self.target = E[target_ids]                 # (L, d)
        self.slot_sizes = [len(target_ids)]
        n = 32
        self.examples_by_split = {s: [None] * n for s in ("train", "val", "test")}
        self.xy_by_split = {s: [("", "")] * n for s in ("train", "val", "test")}
        self.tok = tokenizer

    def loss(self, z, split="train", indices=None, backward=False, mini_batch_size=None):
        if callable(z):                     # match NLLObjective: z may be a thunk
            z = z()
        val = ((z - self.target) ** 2).mean()
        if backward:
            val.backward()
            return float(val.detach())
        return val

    def hard_loss(self, text, split, mini_batch_size=None):
        L = self.target.shape[0]
        ids = (self.tok.encode(text) + [0] * L)[:L]
        emb = self.E[torch.tensor(ids)]
        return float(((emb - self.target) ** 2).mean())


def _setup(seed=0, V=40, L=5):
    torch.manual_seed(seed)
    E = torch.randn(V, V)                            # SQUARE -> invertible a.s.
    tok = CharTokenizer(V)
    # target tokens spread across the vocab (avoid eos=V-1)
    target_ids = torch.tensor([3, 9, 17, 24, 31][:L])
    obj = RecoverObjective(E, target_ids, tok)
    return E, tok, target_ids, obj, FakeModel()


def _overlap(rec_text, target_ids, tok):
    rec = tok.encode(rec_text)
    return sum(1 for a, b in zip(rec, target_ids.tolist()) if a == b)


def test_gcg_recovers_known_target():
    E, tok, target_ids, obj, model = _setup()
    base = obj.hard_loss(tok.decode(torch.zeros(len(target_ids), dtype=torch.long)), "train")
    cfg = {"num_steps": 80, "search_width": 256, "topk": 16, "n_replace": 1,
           "select_n": 16, "mini_batch_size": 4, "filter_ids": True,
           "allow_non_ascii": True}
    res = run_gcg(obj, model, tok, E, cfg=cfg, seed=0)
    ov = _overlap(res["best_text"], target_ids, tok)
    print(f"  GCG: select={res['best_select_score']:.4f} (base {base:.4f}) "
          f"recovered {ov}/{len(target_ids)} tokens")
    assert res["best_select_score"] < 0.2 * base, "GCG must cut the loss substantially"
    assert ov >= len(target_ids) - 1, f"GCG must recover >= L-1 target tokens, got {ov}"
    print("OK run_gcg recovers the known target")


def test_pgd_recovers_known_target():
    E, tok, target_ids, obj, model = _setup()
    base = obj.hard_loss(tok.decode(torch.zeros(len(target_ids), dtype=torch.long)), "train")
    # Toy-friendly schedule: low eta_min (LR decays, no upward cosine), no
    # patience reset, aux loss off (the synthetic objective has no real templates
    # for the control-CE forward), allow_non_ascii (toy chars run past ASCII).
    # This validates the LOOP wiring + recovery; canonical-constant faithfulness
    # is covered by tests/test_pgd.py + the adversarial source review.
    cfg = {"num_steps": 300, "lr": 0.3, "entropy_factor": 0.5, "grad_clip": 20.0,
           "mini_batch_size": 8, "train_batch_size": 16, "eval_n": 16, "eval_chunk": 16,
           "eta_min": 0.01, "warmup_steps": 20, "restart_period": 40, "anneal_duration": 50,
           "patience_value": 10000, "aux_loss": False, "allow_non_ascii": True}
    res = run_pgd(obj, model, tok, E, cfg=cfg, seed=0)
    ov = _overlap(res["best_text"], target_ids, tok)
    print(f"  PGD: select={res['best_select_score']:.4f} (base {base:.4f}) "
          f"recovered {ov}/{len(target_ids)} tokens")
    assert res["best_select_score"] < 0.5 * base, "PGD must reduce the loss"
    assert ov >= len(target_ids) - 1, f"PGD must recover >= L-1 target tokens, got {ov}"
    print("OK run_pgd recovers the known target")


def test_pgd_patience_reinit_restarts_and_preserves_best():
    """patience_mode='reinit' (NON-SRC extension): on stall, random-restart S +
    Adam + LR schedule + entropy anneal instead of reset-to-best. The global best
    is tracked across restarts, so recovery is no worse than without restarts. We
    spy on init_S to confirm the restart path actually fired (>1 call = ≥1 reinit)."""
    import optimize.pgd_geisler as G
    E, tok, target_ids, obj, model = _setup()
    base = obj.hard_loss(tok.decode(torch.zeros(len(target_ids), dtype=torch.long)), "train")
    calls = {"n": 0}
    orig_init_S = G.GeislerPGD.init_S
    def counting_init_S(self):
        calls["n"] += 1
        return orig_init_S(self)
    G.GeislerPGD.init_S = counting_init_S
    try:
        # Small patience -> the toy plateaus after recovery and reinits repeatedly.
        cfg = {"num_steps": 200, "lr": 0.3, "entropy_factor": 0.5, "grad_clip": 20.0,
               "mini_batch_size": 8, "train_batch_size": 16, "eval_n": 16, "eval_chunk": 16,
               "eta_min": 0.01, "warmup_steps": 20, "restart_period": 40, "anneal_duration": 50,
               "patience_value": 15, "patience_mode": "reinit", "aux_loss": False,
               "allow_non_ascii": True}
        res = run_pgd(obj, model, tok, E, cfg=cfg, seed=0)
    finally:
        G.GeislerPGD.init_S = orig_init_S
    ov = _overlap(res["best_text"], target_ids, tok)
    print(f"  PGD/reinit: select={res['best_select_score']:.4f} (base {base:.4f}) "
          f"init_S calls={calls['n']} recovered {ov}/{len(target_ids)} tokens")
    assert calls["n"] >= 2, "reinit must random-restart at least once (init_S beyond the initial call)"
    assert res["best_select_score"] < 0.5 * base, "reinit must still reduce the loss (global best preserved)"
    print("OK run_pgd patience_mode=reinit restarts and preserves the best")


def _count_init_S_calls(cfg):
    """Run run_pgd with a spy on init_S; return (init_S call count, result)."""
    import optimize.pgd_geisler as G
    E, tok, target_ids, obj, model = _setup()
    calls = {"n": 0}
    orig = G.GeislerPGD.init_S
    G.GeislerPGD.init_S = lambda self: (calls.__setitem__("n", calls["n"] + 1), orig(self))[1]
    try:
        res = run_pgd(obj, model, tok, E, cfg=cfg, seed=0)
    finally:
        G.GeislerPGD.init_S = orig
    return calls["n"], res, target_ids, tok


def test_pgd_patience_mix_coin_boundaries():
    """patience_mode='mix' flips a coin per stall (single-stream analog of the src
    50/50). The boundaries pin the wiring deterministically: prob=1.0 reinits every
    trigger (init_S called many times); prob=0.0 never reinits (init_S called once,
    == reset_to_best)."""
    common = dict(num_steps=200, lr=0.3, entropy_factor=0.5, grad_clip=20.0,
                  mini_batch_size=8, train_batch_size=16, eval_n=16, eval_chunk=16,
                  eta_min=0.01, warmup_steps=20, restart_period=40, anneal_duration=50,
                  patience_value=15, patience_mode="mix", aux_loss=False, allow_non_ascii=True)
    n_always, res1, tgt, tok = _count_init_S_calls({**common, "patience_reinit_prob": 1.0})
    n_never, _, _, _ = _count_init_S_calls({**common, "patience_reinit_prob": 0.0})
    print(f"  mix prob=1.0 -> init_S calls={n_always} (reinit every trigger); "
          f"prob=0.0 -> init_S calls={n_never} (never)")
    assert n_always >= 3, f"mix prob=1.0 must reinit on every trigger, init_S={n_always}"
    assert n_never == 1, f"mix prob=0.0 must never reinit (only the initial init_S), got {n_never}"
    assert _overlap(res1["best_text"], tgt, tok) >= len(tgt) - 1, "mix must still recover"
    print("OK run_pgd patience_mode=mix coin boundaries (prob 1.0 vs 0.0)")


if __name__ == "__main__":
    for fn in (test_gcg_recovers_known_target, test_pgd_recovers_known_target,
               test_pgd_patience_reinit_restarts_and_preserves_best,
               test_pgd_patience_mix_coin_boundaries):
        fn()
    print("\nALL OPTIMIZER-LOOP RECOVERY TESTS PASSED")
