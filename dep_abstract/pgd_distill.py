"""PGD discrete optimization for context distillation.

Optimizes a (n_optimize_tokens, vocab_size) probability matrix on the simplex
to minimize NLL of reference rollouts. Argmax gives discrete text at any step.
"""
import random

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

    Standard mixed-precision: keep X in fp32 (for Adam + Duchi simplex projection
    precision on a 128k-vocab simplex), cast to embed_matrix.dtype only for the
    matmul. Differentiable cast → gradients flow back to X in fp32.
    """
    if X.dtype != embed_matrix.dtype:
        X = X.to(embed_matrix.dtype)
    return X @ embed_matrix


def tsallis_entropy(X: torch.Tensor) -> torch.Tensor:
    """Tsallis q=2 entropy (Gini index): S_2(p) = 1 - sum(p_i^2). Per row.

    Returns mean over rows. For one-hot rows S_2 = 0; for uniform over n S_2 = 1 - 1/n.
    """
    return (1.0 - (X * X).sum(dim=-1)).mean()


def clip_per_token_grad_(X: torch.Tensor, max_norm: float) -> None:
    """In-place per-row L2 clipping of X.grad.

    Per the PGD paper: "we clip the L2 norm of the gradient for each token G_i
    to 20. This avoids that exploding gradients mess up the momentum terms in
    the used Adam optimizer."
    """
    if X.grad is None:
        return
    grad_norms = X.grad.norm(dim=-1, keepdim=True)  # (n, 1)
    scale = (max_norm / grad_norms.clamp(min=1e-12)).clamp(max=1.0)
    X.grad.mul_(scale)


def project_entropy(X: torch.Tensor, entropy_factor: float) -> torch.Tensor:
    """Pure entropy projection (expansion only). UPPER bound semantics.

    Mirrors `tsallis_q2_projection` from the reference repo
    (sigeisler/reinforce-attacks-llms baselines/reinforce/pgd_attack.py:702).

    `entropy_factor ∈ [0, 1]` is a fixed-scale knob that maps to a per-row
    target entropy via:
        target_entropy_i = (1 - entropy_factor) * (d_i - 1) / d_i
    where d_i is the support size of row i (number of non-zero entries).

    Endpoints:
    - entropy_factor=0: target_i = (d_i-1)/d_i (the maximum possible Tsallis q=2
      entropy for the current support, i.e., uniform over support). Projection
      never fires → no constraint.
    - entropy_factor=1: target_i = 0 (force one-hot per row). Projection
      pushes every non-singleton row toward an extreme of the simplex.
    - intermediate values scale linearly between max and min entropy.

    Geometry: for p on the simplex with support d, ||p − c||² = (1−S_2(p)) − 1/d
    where c is the uniform-over-support centroid. The set {p : S_2(p) = target}
    is a sphere of radius R = √((1−target) − 1/d). When ||p−c|| < R, expand
    `p` to land on the sphere (toward the simplex boundary, i.e., toward
    one-hot). The expanded point may leave the simplex; the caller should
    re-project with project_simplex afterward.

    Args:
        X: (n_rows, V), each row already on the simplex.
        entropy_factor: scalar in [0, 1].

    Returns:
        Tensor of same shape. Projected rows lie on S_2 = target_i exactly.
        Untouched rows are returned unchanged. May have negative entries.
    """
    if entropy_factor <= 0:
        return X
    eps = 1e-8
    support = (X > 0).float()  # (n, V)
    n_support = support.sum(dim=-1)  # (n,)
    safe_support = n_support.clamp(min=1.0)
    c = support / safe_support.unsqueeze(-1)  # uniform over support
    # Per-row target entropy in the reference's parameterization
    target_entropy = (1.0 - entropy_factor) * (safe_support - 1.0) / safe_support  # (n,)
    R_sq = (1.0 - target_entropy) - 1.0 / safe_support  # (n,)
    R = R_sq.clamp(min=0.0).sqrt()  # (n,)
    diff = X - c
    d = diff.norm(dim=-1)
    needs_proj = (d < R) & (n_support > 1)
    scale = R / d.clamp(min=eps)
    expanded = scale.unsqueeze(-1) * diff + c
    return torch.where(needs_proj.unsqueeze(-1), expanded, X)


def dynamic_entropy_factor(entropy_factor: float, relaxation_gap: float | None,
                           threshold: float = 0.1) -> float:
    """Scale entropy_factor by the relaxation gap (closed-loop feedback).

    Mirrors `dynamic_entropy_factor` in the reference implementation. The
    relaxation gap is `(hard_loss - soft_loss) / hard_loss` (a value in roughly
    [0, 1] — large means relaxed and discrete have diverged).

    - gap >= threshold: scale = 1 (use full entropy_factor → max concentration force)
    - gap < threshold: scale ramps from 0 toward 1 via x_bounded_sigmoid

    The intuition: when relaxed and discrete agree, no need to constrain X.
    When they diverge, slam X back toward one-hot (where they have to agree).

    Args:
        entropy_factor: base factor in [0, 1].
        relaxation_gap: scalar in roughly [0, 1], or None to skip scaling.
        threshold: gap threshold above which entropy projection is at full strength.

    Returns:
        Effective entropy_factor in [0, entropy_factor].
    """
    if relaxation_gap is None or entropy_factor <= 0:
        return entropy_factor
    gap = max(0.0, min(1.0, relaxation_gap))
    if gap >= threshold:
        scale = 1.0
    else:
        # x_bounded_sigmoid(x, k=2) = 1 / (1 + (1/x - 1)^2), in (0, 1) for x in (0, 1)
        squeeze = 1.0 / (1.0 - threshold)
        x = squeeze * gap
        scale = 0.0 if x <= 0 else 1.0 / (1.0 + (1.0 / x - 1.0) ** 2)
    return scale * entropy_factor


def _eval_relaxed(model, embed_matrix, X, rollout_data):
    """Mean NLL across rollouts using relaxed embeds X @ E."""
    z = x_to_embeds(X, embed_matrix)
    losses = []
    for input_ids, optimize_mask, target_mask in rollout_data:
        loss, _ = compute_distill_loss(
            model, embed_matrix, input_ids, z,
            optimize_mask, target_mask,
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def _eval_discrete(model, embed_matrix, X, rollout_data):
    """Mean NLL across rollouts using argmax token embeddings (no grad)."""
    with torch.no_grad():
        ids = X.argmax(dim=-1)  # (n_tokens,)
        z = embed_matrix[ids]  # (n_tokens, dim)
        losses = []
        for input_ids, optimize_mask, target_mask in rollout_data:
            loss, _ = compute_distill_loss(
                model, embed_matrix, input_ids, z,
                optimize_mask, target_mask,
            )
            losses.append(loss.item())
    return sum(losses) / len(losses)


def optimize_abstract_pgd(model, tokenizer, train_rollouts, val_rollouts,
                          user_text, optimize_text, num_steps=100, lr=0.1,
                          entropy_factor=0.0, dynamic_entropy=False,
                          dynamic_threshold=0.1, entropy_warmup_steps=0,
                          discrete_every=5,
                          grad_clip=20.0, proj_iter=1,
                          mini_batch_size=None, patience=0, seed=0,
                          random_init=False,
                          lr_scheduler=None, warmup_steps=100, cosine_t0=60,
                          cosine_eta_min_frac=0.1,
                          log_every=1, test_rollouts=None,
                          save_callback=None, save_every=None):
    """PGD optimization of a probability matrix X over the abstract token positions.

    Args:
        entropy_factor: in [0, 1]. 0 = no constraint, 1 = force one-hot.
        dynamic_entropy: if True, scale entropy_factor by the relaxation gap each step.
        dynamic_threshold: gap threshold for the dynamic feedback (default 0.1 from ref).
        proj_iter: how many times to iterate (project_entropy, project_simplex) per
            optimizer step. 1 = paper-faithful, higher values converge to entropy bound.
        mini_batch_size: if not None, sample this many train rollouts per gradient step
            (instead of summing over all train_rollouts). Cheaper steps but noisier
            gradient and noisier relaxation gap.
        patience: if > 0, reset X (and Adam state) to the best snapshot after this
            many steps without improvement to discrete val. 0 disables.
        seed: RNG seed for mini-batch sampling.
        save_callback: optional fn(step, history, best_ids) called every save_every steps.

    Returns:
        best_ids: (n_optimize_tokens,) discrete token ids from best discrete val
        history: dict with relaxed/discrete train/val/test loss curves and diagnostics
    """
    rng = random.Random(seed)
    embed_matrix = get_embed_matrix(model)
    device = embed_matrix.device
    V = embed_matrix.shape[0]

    train_data = [
        tokenize_with_spans(tokenizer, user_text, optimize_text, r["query_text"], r["rollout_text"])
        for r in train_rollouts
    ]
    val_data = [
        tokenize_with_spans(tokenizer, user_text, optimize_text, r["query_text"], r["rollout_text"])
        for r in val_rollouts
    ]
    test_data = None
    if test_rollouts:
        test_data = [
            tokenize_with_spans(tokenizer, user_text, optimize_text, r["query_text"], r["rollout_text"])
            for r in test_rollouts
        ]

    # Identify optimizable token positions and original ids (used as the discrete
    # baseline for early stopping comparisons regardless of init mode).
    first_ids, first_mask, _ = train_data[0]
    optimize_indices = [i for i, m in enumerate(first_mask) if m]
    optimize_ids = torch.tensor([first_ids[i] for i in optimize_indices], device=device)
    n = len(optimize_ids)

    # Initialize X. fp32 master copy (Adam state needs precision; the bf16
    # simplex projection failed numerically on a 128k vocab — see test_pgd_projections.py).
    if random_init:
        # Match the reference: uniform [0,1] noise, normalize to simplex.
        torch.manual_seed(seed)
        X = torch.rand(n, V, device=device)
        X = X / X.sum(dim=-1, keepdim=True)
    else:
        X = F.one_hot(optimize_ids, num_classes=V).float()
    X.requires_grad_(True)

    def _build_scheduler(opt):
        """Build the lr scheduler if requested. Paper-style: linear warmup over
        warmup_steps, then CosineAnnealingWarmRestarts with T_0=cosine_t0."""
        if lr_scheduler != "cosine":
            return None
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cosine_t0, T_mult=2, eta_min=lr * cosine_eta_min_frac
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    optimizer = torch.optim.Adam([X], lr=lr)
    scheduler = _build_scheduler(optimizer)
    history = {"train": [], "val": [], "test": [],
               "discrete_train": [], "discrete_val": [], "discrete_test": [],
               "tsallis_entropy": [], "n_tokens_diff": [],
               "grad_norm_mean": [], "grad_norm_max": [],
               "entropy_factor_eff": [], "relaxation_gap": [],
               "patience_reset_steps": []}
    best_discrete_val = float("inf")
    best_ids = optimize_ids.clone()
    best_step = 0

    # Print initial state before any optimization
    with torch.no_grad():
        init_train = _eval_discrete(model, embed_matrix, X, train_data)
        init_val = _eval_discrete(model, embed_matrix, X, val_data)
        init_test = _eval_discrete(model, embed_matrix, X, test_data) if test_data else None
        init_text = tokenizer.decode(X.argmax(dim=-1).tolist(), skip_special_tokens=False)
        test_str = f" test={init_test:.4f}" if init_test is not None else ""
        print(f"  init: train={init_train:.4f} val={init_val:.4f}{test_str}")
        print(f"    tokens: {init_text!r}", flush=True)

    for step in range(num_steps):
        optimizer.zero_grad()
        # Mini-batch sample for THIS step's gradient + gap computation.
        if mini_batch_size is not None and mini_batch_size < len(train_data):
            batch = rng.sample(train_data, mini_batch_size)
        else:
            batch = train_data
        train_loss = 0.0
        for input_ids, optimize_mask, target_mask in batch:
            z = x_to_embeds(X, embed_matrix)
            loss, _ = compute_distill_loss(
                model, embed_matrix, input_ids, z,
                optimize_mask, target_mask,
            )
            (loss / len(batch)).backward()
            train_loss += loss.item()
        train_loss /= len(batch)

        # Log per-token grad L2 norms BEFORE clipping
        with torch.no_grad():
            pre_clip_norms = X.grad.norm(dim=-1)  # (n,)
            grad_norm_mean = pre_clip_norms.mean().item()
            grad_norm_max = pre_clip_norms.max().item()
        # Per-token L2 grad clip (paper: per-row max_norm=20 to keep Adam momentum sane).
        clip_per_token_grad_(X, max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Project back to simplex after Adam step.
        with torch.no_grad():
            X.data = project_simplex(X.data)

        # Decide effective entropy_factor for THIS step's projection.
        # 1) Linear warmup over `entropy_warmup_steps` (matches the reference's
        #    anneal_config: ramp 0 → entropy_factor over duration).
        if entropy_warmup_steps > 0 and step < entropy_warmup_steps:
            ef_base = entropy_factor * (step / entropy_warmup_steps)
        else:
            ef_base = entropy_factor
        # 2) Dynamic feedback scales the warmed-up base by the relaxation gap.
        if dynamic_entropy and ef_base > 0 and len(history["relaxation_gap"]) > 0:
            prev_gap = history["relaxation_gap"][-1]
            ef_eff = dynamic_entropy_factor(ef_base, prev_gap, dynamic_threshold)
        else:
            ef_eff = ef_base

        # Entropy projection (then re-simplex). Paper-faithful Algorithm 3.
        # proj_iter > 1 iterates the (entropy, simplex) cycle to actually
        # converge to the entropy bound (test_pgd_projections.py shows this
        # converges within ~10-30 iterations for typical inputs).
        with torch.no_grad():
            if ef_eff > 0:
                for _ in range(proj_iter):
                    X.data = project_entropy(X.data, ef_eff)
                    X.data = project_simplex(X.data)

        # Relaxed val
        with torch.no_grad():
            val_loss = _eval_relaxed(model, embed_matrix, X, val_data).item()
            test_loss = None
            if test_data:
                test_loss = _eval_relaxed(model, embed_matrix, X, test_data).item()
            ent = tsallis_entropy(X).item()
            cur_ids = X.argmax(dim=-1)
            n_diff = (cur_ids != optimize_ids).sum().item()

        # Discrete eval on train/val/test (every discrete_every steps + last step).
        # When dynamic_entropy is on, force d_train every step so the relaxation
        # gap is fresh for the next step's projection. d_train is computed on
        # the same mini-batch as the gradient (for consistency in the gap calc).
        do_discrete = (step % discrete_every == 0) or (step == num_steps - 1)
        force_d_train = dynamic_entropy and entropy_factor > 0
        if do_discrete or force_d_train:
            d_train = _eval_discrete(model, embed_matrix, X, batch)
        else:
            d_train = None
        if do_discrete:
            d_val = _eval_discrete(model, embed_matrix, X, val_data)
            d_test = _eval_discrete(model, embed_matrix, X, test_data) if test_data else None
        else:
            d_val = None
            d_test = None

        # Relaxation gap for dynamic feedback: (hard_train - soft_train) / hard_train
        if d_train is not None and d_train > 0:
            gap = (d_train - train_loss) / d_train
        else:
            gap = None

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["test"].append(test_loss)
        history["discrete_train"].append(d_train)
        history["discrete_val"].append(d_val)
        history["discrete_test"].append(d_test)
        history["tsallis_entropy"].append(ent)
        history["n_tokens_diff"].append(n_diff)
        history["grad_norm_mean"].append(grad_norm_mean)
        history["grad_norm_max"].append(grad_norm_max)
        history["entropy_factor_eff"].append(ef_eff)
        history["relaxation_gap"].append(gap)

        if d_val is not None and d_val < best_discrete_val:
            best_discrete_val = d_val
            best_step = step
            with torch.no_grad():
                best_ids = X.argmax(dim=-1).clone()

        # Patience reset: if N steps without improvement, snap X back to one_hot(best_ids)
        # and clear Adam state. The scheduler continues running (option B): we let
        # the global lr schedule keep progressing rather than restarting warmup.
        did_reset = False
        if patience > 0 and (step - best_step) >= patience:
            with torch.no_grad():
                X.data = F.one_hot(best_ids, num_classes=V).float()
            optimizer.state.clear()  # fresh Adam momentum/variance, same param ref
            best_step = step  # avoid immediate re-trigger
            did_reset = True
        history["patience_reset_steps"].append(did_reset)

        if step % log_every == 0:
            star = " *" if (d_val is not None and d_val == best_discrete_val) else ""
            d_train_str = f"{d_train:.4f}" if d_train is not None else "  -  "
            d_val_str = f"{d_val:.4f}" if d_val is not None else "  -  "
            d_test_str = f" hard_test={d_test:.4f}" if d_test is not None else ""
            gap_str = f" gap={gap:+.3f}" if gap is not None else ""
            ef_str = f" ef={ef_eff:.3f}" if entropy_factor > 0 else ""
            reset_str = " RESET" if did_reset else ""
            cur_text = tokenizer.decode(cur_ids.tolist(), skip_special_tokens=False)
            print(f"  step {step:3d}/{num_steps} "
                  f"soft_train={train_loss:.4f} soft_val={val_loss:.4f} "
                  f"hard_train={d_train_str} hard_val={d_val_str}{d_test_str} "
                  f"S2={ent:.3f} n_diff={n_diff}/{n}{gap_str}{ef_str} "
                  f"|g|={grad_norm_mean:.2f}(max {grad_norm_max:.1f}){star}{reset_str}",
                  flush=True)
            print(f"    tokens: {cur_text!r}", flush=True)

        # Incremental partial save (so killed runs preserve data)
        if save_callback is not None and save_every is not None and (
                step % save_every == 0 or step == num_steps - 1):
            save_callback(step, history, best_ids)

    # Decode and print the best discrete solution for eyeballing
    best_text = tokenizer.decode(best_ids.tolist(), skip_special_tokens=False)
    n_changed = (best_ids != optimize_ids).sum().item()
    print(f"  best_discrete_val={best_discrete_val:.4f}  "
          f"n_tokens_changed={n_changed}/{n}")
    print(f"  best_text: {best_text!r}")

    return best_ids, history
