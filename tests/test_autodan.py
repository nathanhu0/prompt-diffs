"""CPU-only checks for AutoDAN helper logic.

Run directly:

    PYTHONPATH=. uv run python tests/test_autodan.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from optimize.autodan import _candidate_next_tokens, _sample_from_losses


def test_candidate_next_tokens_keeps_current():
    proxy = torch.arange(10, dtype=torch.float32)
    out = _candidate_next_tokens(
        proxy, topk=3, batch_size=3, current_token_id=9,
    )
    vals = set(out.tolist())
    assert 9 in vals, "current token must be retained for STO convergence"
    assert vals.issuperset({0, 1, 2}), "top proxy tokens should be retained"
    print("OK candidate set keeps current token")


def test_candidate_next_tokens_respects_batch_cap_plus_current():
    torch.manual_seed(0)
    proxy = torch.arange(20, dtype=torch.float32)
    out = _candidate_next_tokens(
        proxy, topk=10, batch_size=4, current_token_id=19,
    )
    assert len(out) <= 5, "batch cap plus current token should bound candidates"
    assert 19 in set(out.tolist())
    print("OK candidate set respects batch cap")


def test_sample_from_losses_argmin_at_zero_temperature():
    losses = torch.tensor([3.0, 1.0, 2.0])
    assert _sample_from_losses(losses, temperature=0.0) == 1
    assert _sample_from_losses(losses, temperature=-1.0) == 1
    print("OK zero-temperature sampling is argmin")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} AUTODAN UNIT TESTS PASSED")
