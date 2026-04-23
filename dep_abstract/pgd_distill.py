"""PGD discrete optimization for context distillation.

Optimizes a (n_abstract_tokens, vocab_size) probability matrix on the simplex
to minimize NLL of reference rollouts. Argmax gives discrete text at any step.
"""
import torch
import torch.nn.functional as F

from soft_distill import (
    get_embed_matrix,
    tokenize_with_spans,
    compute_distill_loss,
    compute_distill_loss_multi,
)


def project_simplex(X: torch.Tensor) -> torch.Tensor:
    """Duchi et al. (2008) projection onto the probability simplex.

    Vectorized over rows. For each row x, solves:
        min_p ||p - x||_2^2  s.t.  sum(p) = 1, p >= 0

    Args:
        X: (n_rows, vocab_size) tensor (can be off-simplex)

    Returns:
        Projected tensor of same shape, each row on the simplex.
    """
    n, V = X.shape
    # Sort each row in descending order
    mu, _ = torch.sort(X, dim=-1, descending=True)
    # cumsum and find rho per row
    cssv = torch.cumsum(mu, dim=-1) - 1.0
    rng = torch.arange(1, V + 1, device=X.device, dtype=X.dtype)
    cond = mu - cssv / rng > 0  # (n, V)
    # rho = number of nonzero elements after projection (last True index + 1)
    rho = cond.int().sum(dim=-1)  # (n,)
    # theta = (sum_{i<=rho} mu_i - 1) / rho per row
    theta = cssv.gather(-1, (rho - 1).unsqueeze(-1)).squeeze(-1) / rho.to(X.dtype)
    return torch.clamp(X - theta.unsqueeze(-1), min=0.0)


def x_to_embeds(X: torch.Tensor, embed_matrix: torch.Tensor) -> torch.Tensor:
    """Convert probability matrix X (n_tokens, vocab_size) to embeddings via X @ E.

    Output shape: (n_tokens, hidden_dim).
    """
    return X @ embed_matrix


def tsallis_entropy(X: torch.Tensor) -> torch.Tensor:
    """Tsallis q=2 entropy (Gini index): S_2(p) = 1 - sum(p_i^2). Per row.

    Returns mean over rows. For one-hot rows S_2 = 0; for uniform over n S_2 = 1 - 1/n.
    """
    return (1.0 - (X * X).sum(dim=-1)).mean()


def project_entropy(X: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """Project onto the per-row Tsallis q=2 entropy sphere S_2 = target_entropy.

    This is the *pure* entropy projection — expansion only, no simplex
    re-projection. The caller should follow this with project_simplex if a
    valid simplex point is required.

    Geometry: for p on the simplex with support k,
        ||p - c||^2 = (1 - S_2(p)) - 1/k
    so the locus {p : S_2(p) = S_target} is a sphere of radius
        R = sqrt((1 - S_target) - 1/k)
    centered at c = uniform over the support of s. Expansion by R/d > 1 along
    (s-c) lands exactly on S_2 = target_entropy (algebraically — see test
    `test_pgd_projections.py::test_entropy_post_pe_lands_at_target`).

    Behavior:
    - If d >= R (already at lower entropy than target), the row is unchanged.
    - Otherwise, the row is expanded onto the entropy sphere. The result may
      have negative entries; the simplex constraint is the caller's problem.

    Args:
        X: (n_rows, V), each row already on the simplex.
        target_entropy: scalar S_q=2 upper bound in [0, 1 - 1/V].

    Returns:
        Tensor of same shape. Rows that needed projection lie on S_2 = target.
        Rows that didn't are returned unchanged. May have negative entries.
    """
    eps = 1e-8
    support = (X > 0).float()  # (n, V)
    n_support = support.sum(dim=-1)  # (n,)
    safe_support = n_support.clamp(min=1.0)
    c = support / safe_support.unsqueeze(-1)  # (n, V): uniform over support
    # Radius of the entropy sphere
    R_sq = (1.0 - target_entropy) - 1.0 / safe_support  # (n,)
    R = R_sq.clamp(min=0.0).sqrt()  # (n,)
    diff = X - c  # (n, V)
    d = diff.norm(dim=-1)  # (n,)
    # Already peaked enough when d >= R AND support > 1 (one-hot rows are no-op)
    needs_proj = (d < R) & (n_support > 1)  # (n,)
    scale = R / d.clamp(min=eps)  # (n,) — > 1 in the projection branch (expansion)
    expanded = scale.unsqueeze(-1) * diff + c  # (n, V)
    return torch.where(needs_proj.unsqueeze(-1), expanded, X)


def _eval_relaxed(model, embed_matrix, X, rollout_data):
    """Mean NLL across rollouts using relaxed embeds X @ E."""
    abstract_embeds = x_to_embeds(X, embed_matrix)
    losses = []
    for input_ids, abstract_mask, target_mask in rollout_data:
        loss, _ = compute_distill_loss(
            model, embed_matrix, input_ids, abstract_embeds,
            abstract_mask, target_mask,
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def _eval_discrete(model, embed_matrix, X, rollout_data):
    """Mean NLL across rollouts using argmax token embeddings (no grad)."""
    with torch.no_grad():
        ids = X.argmax(dim=-1)  # (n_tokens,)
        abstract_embeds = embed_matrix[ids]  # (n_tokens, dim)
        losses = []
        for input_ids, abstract_mask, target_mask in rollout_data:
            loss, _ = compute_distill_loss(
                model, embed_matrix, input_ids, abstract_embeds,
                abstract_mask, target_mask,
            )
            losses.append(loss.item())
    return sum(losses) / len(losses)


def optimize_abstract_pgd(model, tokenizer, train_rollouts, val_rollouts,
                          title, abstract, num_steps=100, lr=0.1,
                          target_entropy=None, discrete_every=5,
                          log_every=1, test_rollouts=None):
    """PGD optimization of a probability matrix X over the abstract token positions.

    Returns:
        best_ids: (n_abstract_tokens,) discrete token ids from best discrete val
        history: dict with relaxed/discrete train/val/test loss curves and concentration
    """
    embed_matrix = get_embed_matrix(model)
    device = embed_matrix.device
    V = embed_matrix.shape[0]

    train_data = [
        tokenize_with_spans(tokenizer, title, abstract, r["query_text"], r["rollout_text"])
        for r in train_rollouts
    ]
    val_data = [
        tokenize_with_spans(tokenizer, title, abstract, r["query_text"], r["rollout_text"])
        for r in val_rollouts
    ]
    test_data = None
    if test_rollouts:
        test_data = [
            tokenize_with_spans(tokenizer, title, abstract, r["query_text"], r["rollout_text"])
            for r in test_rollouts
        ]

    # Initialize X as one-hot from original abstract token ids
    first_ids, first_mask, _ = train_data[0]
    abstract_indices = [i for i, m in enumerate(first_mask) if m]
    abstract_ids = torch.tensor([first_ids[i] for i in abstract_indices], device=device)
    n = len(abstract_ids)
    X = F.one_hot(abstract_ids, num_classes=V).float()
    X.requires_grad_(True)

    optimizer = torch.optim.Adam([X], lr=lr)
    history = {"train": [], "val": [], "test": [],
               "discrete_val": [], "discrete_test": [], "tsallis_entropy": []}
    best_discrete_val = float("inf")
    best_ids = abstract_ids.clone()

    for step in range(num_steps):
        optimizer.zero_grad()
        train_loss = 0.0
        for input_ids, abstract_mask, target_mask in train_data:
            abstract_embeds = x_to_embeds(X, embed_matrix)
            loss, _ = compute_distill_loss(
                model, embed_matrix, input_ids, abstract_embeds,
                abstract_mask, target_mask,
            )
            (loss / len(train_data)).backward()
            train_loss += loss.item()
        train_loss /= len(train_data)

        torch.nn.utils.clip_grad_norm_([X], max_norm=1.0)
        optimizer.step()

        # Hard projections after the gradient step:
        #   1) simplex projection to land on a valid simplex point
        #   2) entropy projection (pure expansion onto S_2 = target sphere)
        #   3) simplex projection again (entropy expansion can produce negatives)
        # The composition is paper-faithful Algorithm 3 — entropy bound is
        # enforced asymptotically over many gradient steps, not per call.
        with torch.no_grad():
            X.data = project_simplex(X.data)
            if target_entropy is not None:
                X.data = project_entropy(X.data, target_entropy)
                X.data = project_simplex(X.data)

        # Relaxed val
        with torch.no_grad():
            val_loss = _eval_relaxed(model, embed_matrix, X, val_data).item()
            test_loss = None
            if test_data:
                test_loss = _eval_relaxed(model, embed_matrix, X, test_data).item()
            ent = tsallis_entropy(X).item()

        # Discrete val (every discrete_every steps + last step)
        do_discrete = (step % discrete_every == 0) or (step == num_steps - 1)
        if do_discrete:
            d_val = _eval_discrete(model, embed_matrix, X, val_data)
            d_test = _eval_discrete(model, embed_matrix, X, test_data) if test_data else None
        else:
            d_val = None
            d_test = None

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["test"].append(test_loss)
        history["discrete_val"].append(d_val)
        history["discrete_test"].append(d_test)
        history["tsallis_entropy"].append(ent)

        if d_val is not None and d_val < best_discrete_val:
            best_discrete_val = d_val
            with torch.no_grad():
                best_ids = X.argmax(dim=-1).clone()

        if step % log_every == 0:
            star = " *" if (d_val is not None and d_val == best_discrete_val) else ""
            d_val_str = f"{d_val:.4f}" if d_val is not None else "  -  "
            print(f"  step {step:3d}/{num_steps} train={train_loss:.4f} val={val_loss:.4f} "
                  f"d_val={d_val_str} S2={ent:.3f}{star}")

    return best_ids, history
