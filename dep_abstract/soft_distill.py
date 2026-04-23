"""Soft prompt optimization for context distillation.

Optimizes abstract embeddings to minimize NLL of reference rollouts.
"""
import torch
import torch.nn.functional as F


def get_embed_matrix(model):
    """Get the token embedding weight matrix."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


def tokenize_with_spans(tokenizer, user_text, optimize_text, query, rollout_text):
    """Tokenize a full chat message and return token ids + span masks.

    Args:
        tokenizer: HF tokenizer
        user_text: the full user message content (e.g. "Title: ...\n\nAbstract: ...")
        optimize_text: substring of user_text to mark as optimizable (must appear exactly once)
        query: query text appended after user_text
        rollout_text: assistant response (the target for loss computation)

    Returns:
        input_ids: list of token ids
        optimize_mask: bool list, True for optimizable tokens
        target_mask: bool list, True for assistant response tokens
    """
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"{user_text}\n\n{query}"},
    ]
    messages.append({"role": "assistant", "content": rollout_text})

    # Get full text and tokenize with offsets
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping  # list of (start_char, end_char)

    # Find optimize_text char span
    opt_start = full_text.index(optimize_text)
    opt_end = opt_start + len(optimize_text)
    assert full_text.count(optimize_text) == 1, \
        f"optimize_text must appear exactly once in rendered template, found {full_text.count(optimize_text)}"

    # Find target boundary via prefix length (matches scorer exactly)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    target_token_start = len(prompt_ids)

    # Map char spans to token masks
    optimize_mask = []
    target_mask = []
    for idx, (cs, ce) in enumerate(offsets):
        optimize_mask.append(cs >= opt_start and ce <= opt_end and cs < ce)
        target_mask.append(idx >= target_token_start)

    return input_ids, optimize_mask, target_mask


def compute_distill_loss(model, embed_matrix, input_ids, z,
                         optimize_mask, target_mask):
    """Compute NLL on target tokens, substituting optimizable embeddings.

    Args:
        model: frozen LLM
        embed_matrix: token embedding matrix
        input_ids: full sequence token ids (list)
        z: (n_optimize_tokens, dim) learnable embeddings
        optimize_mask: bool list marking optimizable tokens
        target_mask: bool list marking target tokens

    Returns:
        scalar NLL loss (per-token mean over target tokens)
    """
    ids_tensor = torch.tensor(input_ids, device=embed_matrix.device)
    embeds = embed_matrix[ids_tensor]  # (seq_len, dim)

    # Substitute optimizable embeddings
    optimize_indices = [i for i, m in enumerate(optimize_mask) if m]
    embeds = embeds.clone()
    embeds[optimize_indices] = z

    logits = model(inputs_embeds=embeds.unsqueeze(0)).logits[0]  # (seq_len, vocab)

    # NLL on target tokens only (shifted by 1)
    target_indices = [i for i, m in enumerate(target_mask) if m]
    # Predictions for target[i] come from logits[i-1]
    pred_indices = [i - 1 for i in target_indices]
    target_logits = logits[pred_indices]  # (n_target, vocab)
    target_labels = ids_tensor[target_indices]  # (n_target,)

    per_token = F.cross_entropy(target_logits, target_labels, reduction="none")
    return per_token.mean(), per_token


def compute_distill_loss_multi(model, embed_matrix, z,
                               rollout_data, reduction="mean"):
    """Compute mean NLL across multiple rollouts.

    Args:
        z: (n_optimize_tokens, dim) learnable embeddings
        rollout_data: list of (input_ids, optimize_mask, target_mask) tuples
        reduction: "mean" or "none"
    """
    losses = []
    for input_ids, optimize_mask, target_mask in rollout_data:
        mean_loss, _ = compute_distill_loss(
            model, embed_matrix, input_ids, z,
            optimize_mask, target_mask
        )
        losses.append(mean_loss)

    if reduction == "mean":
        return torch.stack(losses).mean()
    return losses


def optimize_abstract(model, tokenizer, train_rollouts, val_rollouts,
                      user_text, optimize_text, num_steps=100, lr=1e-3,
                      weight_decay=0.0, relative_weight_decay=0.0,
                      suffix_init=None,
                      log_every=1, test_rollouts=None):
    """Optimize embeddings for optimize_text span with early stopping on val.

    Args:
        user_text: full user message content (e.g. "Title: ...\n\nAbstract: ...")
        optimize_text: substring of user_text to optimize (must appear exactly once)
        suffix_init: None = init from original tokens, "random" = random normal,
                     "zeros" = zero embeddings

    Returns:
        best_z (by val), history dict with train/val/test per step
    """
    embed_matrix = get_embed_matrix(model)

    # Tokenize all splits
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

    # Initialize optimizable embeddings
    first_ids, first_mask, _ = train_data[0]
    optimize_indices = [i for i, m in enumerate(first_mask) if m]
    n_tokens = len(optimize_indices)
    dim = embed_matrix.shape[1]
    device = embed_matrix.device

    if suffix_init == "random":
        # Match scale of real embeddings
        z = torch.randn(n_tokens, dim, device=device) * embed_matrix.std()
    elif suffix_init == "zeros":
        z = torch.zeros(n_tokens, dim, device=device)
    else:
        optimize_ids = [first_ids[i] for i in optimize_indices]
        z = embed_matrix[torch.tensor(optimize_ids, device=device)].clone()
    z = z.detach().requires_grad_(True)
    z_init = z.clone().detach()

    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": [], "test": []}
    best_val = float("inf")
    best_z = z.clone().detach()

    for step in range(num_steps):
        optimizer.zero_grad()
        train_loss = 0.0
        for input_ids, optimize_mask, target_mask in train_data:
            loss, _ = compute_distill_loss(
                model, embed_matrix, input_ids, z,
                optimize_mask, target_mask
            )
            (loss / len(train_data)).backward()
            train_loss += loss.item()
        train_loss /= len(train_data)

        if relative_weight_decay > 0:
            reg = relative_weight_decay * ((z - z_init) ** 2).mean()
            reg.backward()

        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            val_loss = compute_distill_loss_multi(
                model, embed_matrix, z, val_data
            ).item()
            test_loss = None
            if test_data:
                test_loss = compute_distill_loss_multi(
                    model, embed_matrix, z, test_data
                ).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["test"].append(test_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_z = z.clone().detach()

        if step % log_every == 0:
            msg = f"  step {step:3d}/{num_steps} train={train_loss:.4f} val={val_loss:.4f}"
            if test_loss is not None:
                msg += f" test={test_loss:.4f}"
            msg += " *" if val_loss == best_val else ""
            print(msg)

    return best_z, history
