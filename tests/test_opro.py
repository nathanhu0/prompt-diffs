"""Dry-run audit for optimize/opro.py — meta-prompt assembly, <prompt> parsing,
USD cost cap, history/selection — all with a STUBBED API call (no spend) and a
fake objective. CPU-only. Run:

    PYTHONPATH=. uv run python tests/test_opro.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimize.opro import parse_prompts, _meta_user, opro_recover


def test_parse_prompts_tags():
    txt = "blah <prompt>You love cats.</prompt> noise <prompt> Be terse. </prompt>"
    out = parse_prompts(txt)
    assert out == ["You love cats.", "Be terse."], out
    print("OK parse_prompts extracts <prompt> blocks")


def test_parse_prompts_fallback():
    txt = "first line\n\nsecond line\n"
    out = parse_prompts(txt)
    assert out == ["first line", "second line"], out
    print("OK parse_prompts line fallback when untagged")


def test_meta_user_orders_best_last_and_asks_n():
    hist = [(0.9, "a"), (0.5, "b"), (0.7, "c")]
    body = _meta_user(hist, n_propose=5, history_topk=2)
    # top-2 by score shown worst->best, so best ("b", 0.5) appears last
    assert body.index("prompt: 'b'") > body.index("prompt: 'c'"), "best must be last"
    assert "'a'" not in body, "history truncated to top-k (a dropped)"
    assert "Propose 5" in body, "asks for n_propose prompts"
    assert "<prompt>" in body, "requests tagged output"
    print("OK _meta_user orders best-last, truncates to top-k, asks for N")


class FakeObjective:
    """hard_loss returns a score that rewards the substring 'cat' (the simulated
    optimum), ignoring the subset. Has sliceable split dicts for the OPRO
    slice/restore."""
    def __init__(self, n=300):
        self.examples_by_split = {"train": list(range(n))}
        self.xy_by_split = {"train": [("x", "y")] * n}

    def hard_loss(self, text, split, indices=None, mini_batch_size=None):
        return 0.4 if "cat" in text.lower() else (0.9 if text else 1.0)


def test_opro_loop_scores_selects_and_restores():
    obj = FakeObjective(n=300)
    full_len = len(obj.xy_by_split["train"])
    calls = {"n": 0}

    def fake_call(system, user):
        calls["n"] += 1
        # step 1 proposes a non-cat prompt; step 2 proposes the cat optimum
        if calls["n"] == 1:
            text = "<prompt>Be helpful.</prompt><prompt>Answer briefly.</prompt>"
        else:
            text = "<prompt>You love cat.</prompt><prompt>Say something.</prompt>"
        usage = {"input_tokens": 1000, "output_tokens": 500}   # ~ $0.0035/call
        return text, usage

    cfg = {"max_steps": 5, "proposals_per_step": 2, "history_topk": 10,
           "scoring_subset": 64, "max_usd": 999}
    res = opro_recover(obj, None, None, None, cfg=cfg, seed=0, _call=fake_call)
    assert res["best_text"] == "You love cat.", res["best_text"]
    assert abs(res["best_select_score"] - 0.4) < 1e-9
    assert res["n_proposals"] == 5 * 2, res["n_proposals"]
    assert res["spent_usd"] > 0
    # split restored to full length (slice/restore correctness)
    assert len(obj.xy_by_split["train"]) == full_len
    assert len(obj.examples_by_split["train"]) == full_len
    print("OK opro_recover scores/selects the optimum, restores splits")


def test_opro_usd_cap_aborts():
    obj = FakeObjective(n=100)

    def pricey_call(system, user):
        return "<prompt>x</prompt>", {"input_tokens": 5_000_000, "output_tokens": 1_000_000}  # ~$10/call

    cfg = {"max_steps": 50, "proposals_per_step": 1, "history_topk": 5,
           "scoring_subset": 32, "max_usd": 12.0}
    res = opro_recover(obj, None, None, None, cfg=cfg, seed=0, _call=pricey_call)
    # ~$10/call: step0 spends 10 (<12, continues), step1 spends 20 (>=12, stop)
    assert res["n_steps"] - 1 <= 2, f"USD cap should stop within ~2 steps, got {res['n_steps']-1}"
    assert res["spent_usd"] >= 12.0
    print("OK opro_recover aborts at the USD cap with best-so-far")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} OPRO DRY-RUN TESTS PASSED")
