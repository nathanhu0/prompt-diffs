"""Unit audit for optimize/gcg.py — the pure token-space helpers and the
one-hot gradient plumbing. CPU-only, deterministic, no model weights (fakes
where a model/tokenizer is needed). Run directly:

    PYTHONPATH=. uv run python tests/test_gcg.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from optimize.gcg import (
    nonascii_token_ids, sample_replacements, filter_retokenizable,
    token_gradient,
)


# --------------------------------------------------------------------------
# Fakes.
# --------------------------------------------------------------------------
class FakeTokenizer:
    """Minimal tokenizer: vocab of single chars; decode = concat chars; encode =
    split chars. `vocab_size` chars are 'a'..; a few are made non-ASCII."""
    def __init__(self):
        self.chars = [chr(ord("a") + i) for i in range(20)]
        self.chars[5] = "é"   # é  (non-ascii)
        self.chars[6] = "☃"   # ☃  (non-ascii)
        self.vocab_size = len(self.chars)
        self.bos_token_id = None
        self.eos_token_id = 19
        self.pad_token_id = 19
        self.unk_token_id = None

    def decode(self, ids):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(self.chars[i] for i in ids)


def approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
# nonascii_token_ids
# --------------------------------------------------------------------------
def test_nonascii():
    tok = FakeTokenizer()
    bad = set(nonascii_token_ids(tok).tolist())
    assert 5 in bad and 6 in bad, "non-ascii chars é/☃ must be flagged"
    assert 19 in bad, "eos/pad special must be flagged"
    assert 0 not in bad and 7 not in bad, "plain ascii letters must NOT be flagged"
    # every flagged id is genuinely non-ascii OR a special
    specials = {19}
    for i in bad - specials:
        assert not (tok.chars[i].isascii() and tok.chars[i].isprintable())
    print("OK nonascii_token_ids")


# --------------------------------------------------------------------------
# sample_replacements
# --------------------------------------------------------------------------
def test_sample_shape_and_locality():
    torch.manual_seed(0)
    L, V, W = 8, 50, 64
    ids = torch.randint(0, V, (L,))
    grad = torch.randn(L, V)
    out = sample_replacements(ids, grad, W, topk=10, n_replace=1)
    assert out.shape == (W, L)
    # each row differs from ids in AT MOST n_replace positions
    diffs = (out != ids.unsqueeze(0)).sum(dim=1)
    assert int(diffs.max()) <= 1, f"n_replace=1 but a row changed {int(diffs.max())} positions"
    print("OK sample_replacements shape + locality (n_replace=1)")


def test_sample_n_replace():
    torch.manual_seed(1)
    L, V, W, nr = 10, 40, 128, 3
    ids = torch.randint(0, V, (L,))
    grad = torch.randn(L, V)
    out = sample_replacements(ids, grad, W, topk=8, n_replace=nr)
    diffs = (out != ids.unsqueeze(0)).sum(dim=1)
    assert int(diffs.max()) <= nr, "must touch at most n_replace positions"
    print("OK sample_replacements n_replace bound")


def test_sample_respects_topk():
    # topk=1: every replacement must be the per-position argmax of -grad.
    torch.manual_seed(2)
    L, V, W = 6, 30, 200
    ids = torch.zeros(L, dtype=torch.long)
    grad = torch.randn(L, V)
    best = (-grad).argmax(dim=1)           # the only allowed replacement per pos
    out = sample_replacements(ids, grad, W, topk=1, n_replace=1)
    for row in out:
        changed = (row != ids).nonzero().flatten()
        for p in changed:
            assert int(row[p]) == int(best[p]), "topk=1 replacement must be argmax(-grad)"
    print("OK sample_replacements respects top-k=1")


def test_sample_excludes_not_allowed():
    torch.manual_seed(3)
    L, V, W = 5, 25, 300
    ids = torch.zeros(L, dtype=torch.long)
    grad = torch.randn(L, V)
    not_allowed = torch.tensor([1, 2, 3, 4, 5, 7, 11, 13])
    out = sample_replacements(ids, grad, W, topk=10, n_replace=1,
                              not_allowed=not_allowed)
    produced = set(out.unique().tolist()) - {0}     # 0 is the unchanged base
    assert produced.isdisjoint(set(not_allowed.tolist())), \
        "not_allowed ids must never be sampled"
    print("OK sample_replacements excludes not_allowed")


# --------------------------------------------------------------------------
# filter_retokenizable
# --------------------------------------------------------------------------
class RoundTripTokenizer:
    """decode/encode are exact inverses for ids < 10; ids >= 10 decode to a
    string that re-encodes to a DIFFERENT length (simulating a BPE merge), so
    rows containing them are dropped."""
    def __call__(self, s, add_special_tokens=False, return_tensors=None):
        # encode: each char 'a'+k -> k; the sentinel '#' -> two tokens [0,0]
        toks = []
        for ch in s:
            toks.append(0 if ch == "#" else ord(ch) - ord("a"))
        return type("E", (), {"input_ids": torch.tensor([toks])})()

    def decode(self, ids):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join("#" if i >= 10 else chr(ord("a") + i) for i in ids)


def test_filter_keeps_roundtrip_drops_others():
    tok = RoundTripTokenizer()
    good = torch.tensor([0, 1, 2])        # 'abc' -> [0,1,2]  (round-trips)
    bad = torch.tensor([0, 11, 2])        # 'a#c' -> [0,0,0,2]/wrong (dropped)
    out = filter_retokenizable(torch.stack([good, bad, good]), tok)
    assert out.shape[0] == 2, "only the two round-tripping rows survive"
    assert all(torch.equal(r, good) for r in out)
    print("OK filter_retokenizable keeps round-trip, drops others")


def test_filter_raises_when_empty():
    tok = RoundTripTokenizer()
    bad = torch.tensor([0, 11, 2])
    try:
        filter_retokenizable(torch.stack([bad, bad]), tok)
    except RuntimeError:
        print("OK filter_retokenizable raises when none survive")
        return
    raise AssertionError("expected RuntimeError when no candidate survives")


# --------------------------------------------------------------------------
# token_gradient — plumbing correctness vs. an analytic loss (no model).
# --------------------------------------------------------------------------
class QuadraticObjective:
    """Fake objective whose loss is ((onehot@E)**2).sum() over a 'minibatch' it
    ignores. Lets us check token_gradient computes d loss/d onehot exactly.
    Mirrors NLLObjective: loss(z, ..., backward=True) calls z's graph .backward()
    and returns a detached scalar; the slot embeds are the leaf-derived z."""
    def __init__(self, embed_matrix):
        self.E = embed_matrix
        self.slot_sizes = [None]

    def loss(self, z, split="train", indices=None, backward=False,
             mini_batch_size=None):
        val = (z ** 2).sum()
        if backward:
            val.backward()
            return float(val.detach())
        return val


def test_token_gradient_matches_analytic():
    torch.manual_seed(4)
    V, d, L = 12, 5, 4
    E = torch.randn(V, d, dtype=torch.float64)
    obj = QuadraticObjective(E)
    optim_ids = torch.randint(0, V, (1, L))
    grad = token_gradient(obj, E, optim_ids, mb_idx=[0], split="train").double()
    # analytic: loss = ||onehot @ E||^2 ; d/d onehot = 2 (onehot @ E) E^T
    onehot = torch.nn.functional.one_hot(optim_ids, V).double()   # (1,L,V)
    analytic = 2.0 * (onehot @ E) @ E.t()                          # (1,L,V)
    err = (grad - analytic).abs().max().item()
    assert err < 1e-8, f"gradient mismatch vs analytic: max err {err}"
    print(f"OK token_gradient matches analytic (max err {err:.2e})")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} GCG UNIT TESTS PASSED")
