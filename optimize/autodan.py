"""AutoDAN-style left-to-right prompt recovery.

This adapts Zhu et al. 2023 ("AutoDAN: Interpretable Gradient-Based
Adversarial Attacks on Large Language Models") from jailbreak suffix search to
dataset-NLL system-prompt recovery.

Original AutoDAN balances two objectives for each new token:
  * task success: make the model likely to emit a target jailbreak response;
  * readability: make the new token likely under the model's next-token
    distribution.

Here the task objective is the existing NLLObjective over a dataset:
    mean_i NLL(response_i | system_prompt=text, user_i)
and the readability objective is the next-token NLL of the candidate token in
the system-prompt slot:
    -log p(token | chat-template-prefix, current_system_prompt).

The fine-selection loss is therefore explicit:
    dataset_nll(prefix + token) + fluency_weight * next_token_nll(token)

The preliminary stage uses the same structure but replaces exact dataset NLL
with the one-hot gradient proxy for the candidate next token.
"""

import torch
import torch.nn.functional as F

from optimize.gcg import nonascii_token_ids, score_candidates


def _decode(tokenizer, ids):
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids)


def _encode(tokenizer, text, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, device=device, dtype=torch.long)


def _roundtrip_rows(rows, tokenizer):
    """Keep rows whose decode -> encode is exactly stable."""
    keep = []
    for row in rows:
        text = _decode(tokenizer, row)
        re = tokenizer(text, add_special_tokens=False,
                       return_tensors="pt").input_ids[0].to(row.device)
        if re.shape == row.shape and torch.equal(re, row):
            keep.append(row)
    return torch.stack(keep) if keep else None


@torch.no_grad()
def next_token_nll(model, tokenizer, context_ids, prefix_ids):
    """Return -log p(v | context_ids + prefix_ids) for every vocab token."""
    device = model.device
    ids = list(context_ids) + list(prefix_ids)
    if not ids:
        fallback = tokenizer.bos_token_id
        if fallback is None:
            fallback = tokenizer.eos_token_id or 0
        ids = [fallback]
    inp = torch.tensor([ids], device=device, dtype=torch.long)
    logits = model(input_ids=inp).logits[0, -1].float()
    return -F.log_softmax(logits, dim=-1)


def next_token_gradient(objective, embed_matrix, prefix_ids, current_token_id,
                        mb_idx, split):
    """Gradient proxy d dataset_NLL / d onehot(next_token), shape (V,)."""
    V = embed_matrix.shape[0]
    device = embed_matrix.device
    prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long)
    onehot = F.one_hot(
        torch.tensor([[current_token_id]], device=device), num_classes=V,
    ).to(embed_matrix.dtype)
    onehot.requires_grad_()
    parts = []
    if prefix.numel():
        parts.append(embed_matrix[prefix])
    parts.append((onehot @ embed_matrix).squeeze(0))
    z = torch.cat(parts, dim=0)
    objective.loss(z, split=split, indices=mb_idx, backward=True)
    return onehot.grad.squeeze(0).squeeze(0).detach()


def _sample_from_losses(losses, temperature):
    """Sample an index from lower-is-better losses; temp<=0 is argmin."""
    if temperature <= 0:
        return int(losses.argmin())
    scaled = -(losses - losses.min()) / temperature
    probs = torch.softmax(scaled, dim=0)
    return int(torch.multinomial(probs, 1).item())


def _candidate_next_tokens(proxy_loss, *, topk, batch_size, current_token_id):
    """Lowest-proxy candidate token ids, with current token kept for monotonicity."""
    k = min(int(topk), proxy_loss.numel())
    ids = torch.topk(proxy_loss, k=k, largest=False).indices
    if batch_size is not None and int(batch_size) < ids.numel():
        perm = torch.randperm(ids.numel(), device=ids.device)[:int(batch_size)]
        ids = ids[perm]
    cur = torch.tensor([current_token_id], device=ids.device, dtype=ids.dtype)
    ids = torch.unique(torch.cat([ids, cur]), sorted=False)
    return ids


def optimize_next_token(objective, model, tokenizer, embed_matrix, prefix_ids,
                        context_ids, mb_idx, *, cfg, split, not_allowed=None):
    """Run AutoDAN's single-token optimization for one new position."""
    device = embed_matrix.device
    V = embed_matrix.shape[0]
    topk = int(cfg.get("topk", cfg.get("batch_size", 512)))
    batch_size = int(cfg.get("batch_size", topk))
    w1 = float(cfg.get("prelim_fluency_weight",
                       cfg.get("fluency_weight", 1.0)))
    w2 = float(cfg.get("fluency_weight", 1.0))
    temperature = float(cfg.get("temperature", 0.5))
    max_inner = int(cfg.get("max_inner_steps", 16))
    filter_ids = bool(cfg.get("filter_ids", True))
    score_cand_chunk = cfg.get("score_cand_chunk")

    if not_allowed is None:
        not_allowed = torch.empty(0, device=device, dtype=torch.long)

    allowed = torch.ones(V, device=device, dtype=torch.bool)
    if len(not_allowed):
        allowed[not_allowed.to(device)] = False
    allowed_ids = allowed.nonzero().flatten()
    current = int(allowed_ids[torch.randint(len(allowed_ids), (1,), device=device)])

    seen_top1 = set()
    records = []
    n_scored = 0
    for inner in range(max_inner):
        grad = next_token_gradient(
            objective, embed_matrix, prefix_ids, current, mb_idx, split,
        )
        flu_nll = next_token_nll(model, tokenizer, context_ids, prefix_ids)
        proxy = grad.float() + w1 * flu_nll
        if len(not_allowed):
            proxy = proxy.clone()
            proxy[not_allowed.to(device)] = float("inf")

        next_ids = _candidate_next_tokens(
            proxy, topk=topk, batch_size=batch_size, current_token_id=current,
        )
        prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long)
        if prefix.numel():
            candidates = torch.cat([
                prefix.unsqueeze(0).repeat(next_ids.numel(), 1),
                next_ids.unsqueeze(1),
            ], dim=1)
        else:
            candidates = next_ids.unsqueeze(1)
        if filter_ids:
            filtered = _roundtrip_rows(candidates, tokenizer)
            if filtered is not None:
                candidates = filtered

        losses = score_candidates(
            objective, embed_matrix, candidates, mb_idx, split,
            cand_chunk=score_cand_chunk,
        ).to(device)
        cand_next = candidates[:, -1]
        fine_losses = losses + w2 * flu_nll[cand_next]
        top1_i = int(fine_losses.argmin())
        top1 = int(cand_next[top1_i])
        sample_i = _sample_from_losses(fine_losses, temperature)
        selected = int(cand_next[sample_i])
        n_scored += int(candidates.shape[0])
        records.append({
            "inner": inner,
            "current": current,
            "selected": selected,
            "top1": top1,
            "top1_loss": float(fine_losses[top1_i].detach().cpu()),
            "n_candidates": int(candidates.shape[0]),
        })
        current = selected
        if top1 in seen_top1:
            break
        seen_top1.add(top1)

    return {
        "token_id": current,
        "top1_id": records[-1]["top1"],
        "records": records,
        "n_scored": n_scored,
    }


def run_autodan(objective, model, tokenizer, embed_matrix, *, cfg, seed,
                split="train", select_split="train"):
    """One AutoDAN run over the system-prompt slot."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = embed_matrix.device
    max_tokens = int(cfg.get("max_tokens", objective.slot_sizes[0]))
    n_train = len(objective.examples_by_split[split])
    n_sel_full = len(objective.examples_by_split[select_split])
    mb = min(int(cfg.get("mini_batch_size", 4)), n_train)
    eval_chunk = int(cfg.get("eval_chunk", 16))
    select_n = min(int(cfg.get("select_n", 256)), n_sel_full)
    select_prefixes = bool(cfg.get("select_prefixes", True))
    not_allowed = (None if cfg.get("allow_non_ascii", False)
                   else nonascii_token_ids(tokenizer, device=device))

    first_template = objective.examples_by_split[split][0].template
    context_ids = list(first_template.prefix_ids or [])

    g_sel = torch.Generator(); g_sel.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g_sel).tolist()[:select_n]
    g_mb = torch.Generator(); g_mb.manual_seed(seed + 1)

    prefix_ids = []
    best_text = ""
    best_sel = float("inf")
    trajectory = []
    if select_prefixes:
        best_sel = float(objective.hard_loss(
            "", select_split, indices=sel_idx, mini_batch_size=eval_chunk,
        ))
        trajectory.append((0, "", best_sel))
    inner_records = []
    n_proposals = 0

    for step in range(max_tokens):
        mb_idx = torch.randperm(n_train, generator=g_mb).tolist()[:mb]
        out = optimize_next_token(
            objective, model, tokenizer, embed_matrix, prefix_ids, context_ids,
            mb_idx, cfg=cfg, split=split, not_allowed=not_allowed,
        )
        prefix_ids = prefix_ids + [out["token_id"]]
        text = _decode(tokenizer, prefix_ids)
        prefix_ids = _encode(tokenizer, text, device).tolist()
        n_proposals += out["n_scored"]
        inner_records.append({"step": step, **out})

        if select_prefixes:
            sel = float(objective.hard_loss(
                text, select_split, indices=sel_idx, mini_batch_size=eval_chunk,
            ))
            trajectory.append((n_proposals, text, sel))
            if sel < best_sel:
                best_sel, best_text = sel, text
            score_msg = f"sel_loss={sel:.4f} (best {best_sel:.4f})"
        else:
            trajectory.append((n_proposals, text, None))
            score_msg = "sel_loss=deferred"
        print(f"  step {step}: {score_msg} "
              f"proposals={n_proposals} prompt={text[:60]!r}", flush=True)

        if cfg.get("stop_on_eos", False) and out["token_id"] == tokenizer.eos_token_id:
            break

    if not select_prefixes:
        best_text = _decode(tokenizer, prefix_ids)
        best_sel = float(objective.hard_loss(
            best_text, select_split, indices=sel_idx, mini_batch_size=eval_chunk,
        ))
        trajectory.append((n_proposals, best_text, best_sel))

    print(f"AutoDAN winner (fixed {select_n}-subset): select={best_sel:.4f}  "
          f"prompt={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_select_score": best_sel,
        "trajectory": trajectory,
        "inner_records": inner_records,
        "n_proposals": n_proposals,
        "n_steps": len(trajectory) - 1,
        "max_tokens": max_tokens,
        "select_split": select_split,
        "fluency_weight": float(cfg.get("fluency_weight", 1.0)),
        "temperature": float(cfg.get("temperature", 0.5)),
        "select_prefixes": select_prefixes,
    }


def autodan_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42):
    """Shared-contract entry point. cfg = the `autodan` config block."""
    return run_autodan(objective, model, tokenizer, embed_matrix,
                       cfg=cfg, seed=seed)
