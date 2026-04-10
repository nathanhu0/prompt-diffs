"""Tests for project_simplex and project_entropy in pgd_distill.py.

Runnable as: `uv run python test_scripts/test_pgd_projections.py`
Not pytest — just plain assertions, prints PASS/FAIL per test.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pgd_distill import project_simplex, project_entropy, tsallis_entropy


# ---------- helpers ----------

def assert_on_simplex(X, tol=1e-5, name=""):
    sums = X.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=tol), \
        f"{name}: rows do not sum to 1: {sums.tolist()}"
    assert (X >= -tol).all(), f"{name}: negative entries: min={X.min().item()}"


def s2(X):
    """Per-row Tsallis q=2 entropy."""
    return 1.0 - (X * X).sum(dim=-1)


def passed(name):
    print(f"  PASS  {name}")


def failed(name, msg):
    print(f"  FAIL  {name}: {msg}")
    raise AssertionError(f"{name}: {msg}")


# ---------- project_simplex tests ----------

def test_simplex_random():
    name = "simplex/random off-simplex input → on simplex"
    torch.manual_seed(0)
    X = torch.randn(10, 200)
    P = project_simplex(X)
    assert_on_simplex(P, name=name)
    passed(name)


def test_simplex_idempotent():
    name = "simplex/idempotent on simplex points"
    torch.manual_seed(1)
    X = torch.randn(10, 200)
    P1 = project_simplex(X)
    P2 = project_simplex(P1)
    if not torch.allclose(P1, P2, atol=1e-6):
        failed(name, f"max diff {(P1 - P2).abs().max().item()}")
    passed(name)


def test_simplex_one_hot_preserved():
    name = "simplex/one-hot rows preserved"
    n, V = 8, 100
    ids = torch.tensor([0, 1, 5, 7, 13, 50, 99, 42])
    X = torch.nn.functional.one_hot(ids, num_classes=V).float()
    P = project_simplex(X)
    if not torch.allclose(X, P, atol=1e-6):
        failed(name, f"max diff {(X - P).abs().max().item()}")
    passed(name)


def test_simplex_uniform_preserved():
    name = "simplex/uniform distribution preserved"
    V = 100
    u = torch.full((3, V), 1.0 / V)
    P = project_simplex(u)
    if not torch.allclose(u, P, atol=1e-6):
        failed(name, f"max diff {(u - P).abs().max().item()}")
    passed(name)


def test_simplex_dominant_collapses():
    name = "simplex/dominant entry collapses to one-hot"
    X = torch.zeros(1, 100)
    X[0, 5] = 10.0
    X[0, 7] = 8.0
    X[0, 9] = 1.0
    P = project_simplex(X)
    assert_on_simplex(P, name=name)
    if not (P[0, 5] == 1.0 and (P[0] > 0).sum() == 1):
        failed(name, f"got {P[0].topk(5).values.tolist()} support={(P[0] > 0).sum().item()}")
    passed(name)


def test_simplex_two_dominant_split():
    name = "simplex/two close entries → 2-element support"
    X = torch.zeros(1, 100)
    X[0, 5] = 10.0
    X[0, 7] = 9.5
    X[0, 9] = 0.1
    P = project_simplex(X)
    assert_on_simplex(P, name=name)
    support = (P[0] > 1e-8).sum().item()
    if support != 2:
        failed(name, f"expected 2-element support, got {support}: {P[0].topk(5).values.tolist()}")
    if abs((P[0, 5] + P[0, 7]).item() - 1.0) > 1e-5:
        failed(name, f"top two don't sum to 1: {P[0, 5].item()}, {P[0, 7].item()}")
    passed(name)


def test_simplex_negative_input():
    name = "simplex/handles negative input values"
    X = torch.tensor([[-5.0, -3.0, -1.0, 2.0, 4.0]])
    P = project_simplex(X)
    assert_on_simplex(P, name=name)
    # Most negative values should clamp to 0
    if not (P[0, 0] == 0 and P[0, 1] == 0):
        failed(name, f"expected zeros at neg positions, got {P[0].tolist()}")
    passed(name)


def test_simplex_after_gradient_step():
    name = "simplex/realistic gradient-step scenario"
    # Simulate one gradient step on a one-hot init
    torch.manual_seed(2)
    n, V = 5, 100
    ids = torch.tensor([10, 20, 30, 40, 50])
    X = torch.nn.functional.one_hot(ids, num_classes=V).float()
    grad = torch.randn(n, V) * 0.05
    X_post = X - grad  # post Adam step (might leave simplex)
    P = project_simplex(X_post)
    assert_on_simplex(P, name=name)
    passed(name)


# ---------- project_entropy tests (pure expansion, upper-bound semantics) ----------
#
# project_entropy is now the pure entropy projection: it lands rows EXACTLY on
# the S_2 = target sphere via expansion `c + (R/d)(s-c)`, but the result is no
# longer guaranteed to be on the simplex (entries can go negative).
# project_simplex must be called separately to enforce the simplex constraint.

def sample_random_simplex_points(n_points, V, support_sizes, seed=0):
    """Generate n_points random simplex rows of varying support sizes.

    For each point, pick a random support size from `support_sizes`, then
    sample non-negative weights (Dirichlet-style) and normalize. The result
    has support exactly equal to the chosen size (with high probability).
    """
    torch.manual_seed(seed)
    X = torch.zeros(n_points, V)
    for i in range(n_points):
        k = support_sizes[i % len(support_sizes)]
        idx = torch.randperm(V)[:k]
        # Random weights via -log(U) (exponential), then normalize
        w = -torch.log(torch.rand(k).clamp(min=1e-8))
        w = w / w.sum()
        X[i, idx] = w
    return X


def test_entropy_post_pe_lands_at_target():
    """For a population of random simplex points, calling project_entropy once
    should land each row that needs projection EXACTLY on S_2 = target.
    Rows already at S_2 <= target are unchanged.

    This is the algebraic property: pure expansion `c + (R/d)(s-c)` puts the
    row on the entropy sphere. The result may have negatives but its squared
    norm centered at c equals R^2 = (1 - target) - 1/k, so S_2 = target.
    """
    n_points = 20
    V = 200
    X0 = sample_random_simplex_points(n_points, V, support_sizes=[2, 5, 10, 25, 60, 120])
    assert_on_simplex(X0, name="entropy/sampling/initial-on-simplex")

    targets = [0.05, 0.2, 0.5, 0.8]
    for target in targets:
        name = f"entropy/post-PE lands on S_2={target}"
        cur = s2(X0)
        Pe = project_entropy(X0, target_entropy=target)
        new = s2(Pe)
        # For each row: either no projection (cur <= target ⟹ unchanged)
        # or projection (new ≈ target exactly)
        for i in range(n_points):
            if cur[i].item() <= target + 1e-6:
                # No-op expected
                if (Pe[i] - X0[i]).abs().max().item() > 1e-6:
                    failed(name, f"row {i} (cur={cur[i].item():.4f} <= target) was modified")
            else:
                # Projection expected to land exactly on target
                if abs(new[i].item() - target) > 1e-5:
                    failed(name, f"row {i} (cur={cur[i].item():.4f}) "
                                 f"landed at {new[i].item():.6f}, expected {target}")
        passed(name)


def test_entropy_post_simplex_still_decreases():
    """After project_entropy then project_simplex, S_2 may bounce back up
    (Duchi clamps negatives), but should still be <= the original entropy.
    """
    n_points = 20
    V = 200
    X0 = sample_random_simplex_points(n_points, V, support_sizes=[2, 5, 10, 25, 60, 120])

    targets = [0.05, 0.2, 0.5, 0.8]
    for target in targets:
        name = f"entropy/post-PE-then-PS decreases (target={target})"
        cur = s2(X0)
        Pe = project_entropy(X0, target_entropy=target)
        Ps = project_simplex(Pe)
        assert_on_simplex(Ps, name=name)
        new = s2(Ps)
        for i in range(n_points):
            if cur[i].item() <= target + 1e-6:
                continue  # row was a no-op
            if new[i].item() > cur[i].item() + 1e-5:
                failed(name, f"row {i}: entropy increased after re-simplex: "
                             f"{cur[i].item():.4f} → {new[i].item():.4f}")
        passed(name)


def test_entropy_alternating_converges():
    """The headline test: alternating project_entropy + project_simplex should
    monotonically converge S_2 toward target (from above). For 20 random
    starting points and several targets, log entropy across the alternation
    and verify convergence.

    This documents the iteration behavior: project_entropy lands on the sphere
    exactly, project_simplex bounces it back up, but each cycle gets closer.
    """
    n_points = 20
    V = 200
    n_iters = 30
    X0 = sample_random_simplex_points(n_points, V, support_sizes=[2, 5, 10, 25, 60, 120])

    targets = [0.05, 0.2, 0.5, 0.8]
    for target in targets:
        name = f"entropy/alternating PE+PS converges to target={target}"
        X = X0.clone()
        # Track per-row entropy after each project_simplex (one full cycle)
        history = [s2(X).clone()]  # before any projection

        for it in range(n_iters):
            Pe = project_entropy(X, target_entropy=target)
            X = project_simplex(Pe)
            history.append(s2(X).clone())

        H = torch.stack(history)  # (n_iters+1, n_points)

        for i in range(n_points):
            initial = H[0, i].item()
            final = H[-1, i].item()
            row_hist = H[:, i].tolist()

            if initial <= target + 1e-6:
                # Should have stayed put (modulo float noise)
                if abs(final - initial) > 1e-5:
                    failed(name, f"row {i} (initial {initial:.4f} <= target) drifted to {final:.4f}")
                continue

            # Monotonic: each cycle's entropy <= previous cycle's entropy
            for t in range(1, len(row_hist)):
                if row_hist[t] > row_hist[t-1] + 1e-6:
                    failed(name, f"row {i}: entropy increased between cycles "
                                 f"{t-1}→{t}: {row_hist[t-1]:.6f} → {row_hist[t]:.6f}")

            # Converges to (within tolerance of) target
            if final > target + 1e-3:
                failed(name, f"row {i}: did not converge in {n_iters} iters: "
                             f"initial={initial:.4f} → final={final:.6f}, target={target}, "
                             f"hist[:5]={row_hist[:5]}, hist[-5:]={row_hist[-5:]}")
        passed(name)


def test_entropy_one_hot_preserved():
    name = "entropy/one-hot rows preserved (S_2 already 0)"
    ids = torch.tensor([5, 17, 42])
    X = torch.nn.functional.one_hot(ids, num_classes=100).float()
    for tgt in [0.0, 0.1, 0.5, 0.99]:
        P = project_entropy(X, tgt)
        if not torch.allclose(X, P, atol=1e-6):
            failed(name + f" tgt={tgt}", f"changed: max diff {(X - P).abs().max().item()}")
    passed(name)


def test_entropy_target_above_current_no_op():
    name = "entropy/target above current entropy → no-op"
    X = torch.zeros(1, 100)
    X[0, [3, 4]] = 0.5  # S_2 = 0.5
    cur = s2(X)[0].item()
    P = project_entropy(X, target_entropy=0.8)  # 0.8 > 0.5 → no-op
    if abs(s2(P)[0].item() - cur) > 1e-6:
        failed(name, f"S_2 changed: {cur} → {s2(P)[0].item()}")
    if (P - X).abs().max().item() > 1e-6:
        failed(name, "tensor changed")
    passed(name)


# ---------- run all ----------

def main():
    print("=== project_simplex ===")
    test_simplex_random()
    test_simplex_idempotent()
    test_simplex_one_hot_preserved()
    test_simplex_uniform_preserved()
    test_simplex_dominant_collapses()
    test_simplex_two_dominant_split()
    test_simplex_negative_input()
    test_simplex_after_gradient_step()
    print()
    print("=== project_entropy (pure expansion) ===")
    test_entropy_one_hot_preserved()
    test_entropy_target_above_current_no_op()
    test_entropy_post_pe_lands_at_target()
    test_entropy_post_simplex_still_decreases()
    test_entropy_alternating_converges()
    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
