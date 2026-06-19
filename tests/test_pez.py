"""Spec + loop tests for PEZ / Hard Prompts Made Easy.

Run:
    PYTHONPATH=. uv run python tests/test_pez.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from optimize.pez import nearest_token_ids, straight_through_project, run_pez
from tests.test_optimizer_loops import _setup, _overlap


def test_nearest_token_ids_l2_matches_manual():
    torch.manual_seed(0)
    E = torch.randn(17, 9)
    target = torch.tensor([2, 8, 14])
    z = E[target] + 1e-3 * torch.randn(3, 9)
    got = nearest_token_ids(z, E, metric="l2", vocab_chunk=5)
    assert torch.equal(got.cpu(), target), f"nearest ids {got.tolist()} != {target.tolist()}"
    print("OK nearest_token_ids projects to the L2-nearest vocab embeddings")


def test_straight_through_project_identity_gradient():
    torch.manual_seed(1)
    E = torch.randn(11, 7)
    z = (E[[3, 5]] + 0.01 * torch.randn(2, 7)).requires_grad_(True)
    z_st, ids = straight_through_project(z, E, metric="l2")
    loss = z_st.sum()
    loss.backward()
    assert torch.equal(ids.cpu(), torch.tensor([3, 5]))
    assert torch.allclose(z.grad, torch.ones_like(z)), \
        "straight-through projection must pass an identity gradient to z"
    print("OK straight_through_project has hard forward + identity backward")


def test_pez_recovers_known_target():
    E, tok, target_ids, obj, model = _setup()
    base = obj.hard_loss(tok.decode(torch.zeros(len(target_ids), dtype=torch.long)), "train")
    cfg = {
        "num_steps": 300,
        "lr": 0.2,
        "metric": "l2",
        "init": "random",
        "init_noise_std": 0.0,
        "grad_clip": 10.0,
        "mini_batch_size": 8,
        "train_batch_size": 16,
        "select_n": 16,
        "eval_chunk": 16,
        "eval_every": 10,
        "print_every": 100,
        "allow_non_ascii": True,
    }
    res = run_pez(obj, model, tok, E, cfg=cfg, seed=0)
    ov = _overlap(res["best_text"], target_ids, tok)
    print(f"  PEZ: select={res['best_select_score']:.4f} (base {base:.4f}) "
          f"recovered {ov}/{len(target_ids)} tokens")
    assert res["best_select_score"] < 0.2 * base, "PEZ must reduce the loss substantially"
    assert ov >= len(target_ids) - 1, f"PEZ must recover >= L-1 target tokens, got {ov}"
    print("OK run_pez recovers the known target")


if __name__ == "__main__":
    for fn in (test_nearest_token_ids_l2_matches_manual,
               test_straight_through_project_identity_gradient,
               test_pez_recovers_known_target):
        fn()
    print("\nALL PEZ TESTS PASSED")
