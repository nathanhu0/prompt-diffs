"""Soft prompt optimization for context distillation.

Optimizes abstract embeddings to minimize NLL of reference rollouts.
"""
import torch
import torch.nn.functional as F
from generate_reference_rollouts import build_messages


def get_embed_matrix(model):
    """Get the token embedding weight matrix."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


def tokenize_with_spans(tokenizer, title, abstract, query, rollout_text):
    """Tokenize a full chat message and return token ids + span masks.

    Returns:
        input_ids: list of token ids
        abstract_mask: bool list, True for abstract tokens
        target_mask: bool list, True for assistant response tokens
    """
    messages = build_messages(title, abstract, injection="", query=query)
    messages.append({"role": "assistant", "content": rollout_text})

    # Get full text and tokenize with offsets
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping  # list of (start_char, end_char)

    # Find abstract char span
    abstract_start = full_text.index(abstract)
    abstract_end = abstract_start + len(abstract)
    assert full_text.count(abstract) == 1, "Abstract appears multiple times in chat template"

    # Find target boundary via prefix length (matches scorer exactly)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    target_token_start = len(prompt_ids)

    # Map char spans to token masks
    abstract_mask = []
    target_mask = []
    for idx, (cs, ce) in enumerate(offsets):
        abstract_mask.append(cs >= abstract_start and ce <= abstract_end and cs < ce)
        target_mask.append(idx >= target_token_start)

    return input_ids, abstract_mask, target_mask


def compute_distill_loss(model, embed_matrix, input_ids, abstract_embeds,
                         abstract_mask, target_mask):
    """Compute NLL on target tokens, substituting abstract embeddings.

    Args:
        model: frozen LLM
        embed_matrix: token embedding matrix
        input_ids: full sequence token ids (list)
        abstract_embeds: (n_abstract_tokens, dim) learnable embeddings
        abstract_mask: bool list marking abstract tokens
        target_mask: bool list marking target tokens

    Returns:
        scalar NLL loss (per-token mean over target tokens)
    """
    ids_tensor = torch.tensor(input_ids, device=embed_matrix.device)
    embeds = embed_matrix[ids_tensor]  # (seq_len, dim)

    # Substitute abstract embeddings
    abstract_indices = [i for i, m in enumerate(abstract_mask) if m]
    embeds = embeds.clone()
    embeds[abstract_indices] = abstract_embeds

    logits = model(inputs_embeds=embeds.unsqueeze(0)).logits[0]  # (seq_len, vocab)

    # NLL on target tokens only (shifted by 1)
    target_indices = [i for i, m in enumerate(target_mask) if m]
    # Predictions for target[i] come from logits[i-1]
    pred_indices = [i - 1 for i in target_indices]
    target_logits = logits[pred_indices]  # (n_target, vocab)
    target_labels = ids_tensor[target_indices]  # (n_target,)

    per_token = F.cross_entropy(target_logits, target_labels, reduction="none")
    return per_token.mean(), per_token


def compute_distill_loss_multi(model, embed_matrix, abstract_embeds,
                               rollout_data, reduction="mean"):
    """Compute mean NLL across multiple rollouts.

    Args:
        rollout_data: list of (input_ids, abstract_mask, target_mask) tuples
        reduction: "mean" or "none"
    """
    losses = []
    for input_ids, abstract_mask, target_mask in rollout_data:
        mean_loss, _ = compute_distill_loss(
            model, embed_matrix, input_ids, abstract_embeds,
            abstract_mask, target_mask
        )
        losses.append(mean_loss)

    if reduction == "mean":
        return torch.stack(losses).mean()
    return losses


def optimize_abstract(model, tokenizer, train_rollouts, val_rollouts,
                      title, abstract, num_steps=100, lr=1e-3,
                      weight_decay=0.0, relative_weight_decay=0.0,
                      log_every=1, test_rollouts=None):
    """Optimize abstract embeddings with early stopping on val.

    Returns:
        best_z (by val), history dict with train/val/test per step
    """
    embed_matrix = get_embed_matrix(model)

    # Tokenize all splits
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

    # Initialize from original abstract embeddings
    first_ids, first_mask, _ = train_data[0]
    abstract_indices = [i for i, m in enumerate(first_mask) if m]
    abstract_ids = [first_ids[i] for i in abstract_indices]
    z = embed_matrix[torch.tensor(abstract_ids, device=embed_matrix.device)].clone().detach()
    z.requires_grad_(True)
    z_init = z.clone().detach()

    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": [], "test": []}
    best_val = float("inf")
    best_z = z.clone().detach()

    for step in range(num_steps):
        optimizer.zero_grad()
        train_loss = 0.0
        for input_ids, abstract_mask, target_mask in train_data:
            loss, _ = compute_distill_loss(
                model, embed_matrix, input_ids, z,
                abstract_mask, target_mask
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
