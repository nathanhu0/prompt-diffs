"""PEZ / Hard Prompts Made Easy baseline for dataset-NLL prompt recovery.

This is the straight-through projected-embedding optimizer from Wen et al. 2023
("Hard Prompts Made Easy"). A continuous prompt embedding `z ∈ R^{L×d}` is
optimized with Adam, but every forward pass first projects each row to its
nearest vocabulary embedding. The forward therefore scores an actual discrete
prompt, while the backward uses the straight-through estimator:

    z_st = z + (nearest_embedding(z) - z).detach()

For this repo's prompt-recovery setting the objective is the shared
`NLLObjective.loss` over a minibatch of dataset examples, and best-candidate
selection is the uniform fixed train-subset `hard_loss` used by GCG/GBDA/PGD.
"""
import inspect

import torch

from optimize.gcg import nonascii_token_ids


def _decode(tokenizer, ids):
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids)


@torch.no_grad()
def nearest_token_ids(z, embed_matrix, *, not_allowed=None, metric="l2",
                      vocab_chunk=None):
    """Nearest-neighbor projection from prompt embeddings to vocab ids.

    `metric="l2"` is the default because PEZ projects in embedding space and it
    also makes the synthetic recovery tests exact for arbitrary embeddings.
    `vocab_chunk` avoids materializing an L×V score matrix if needed.
    """
    E = embed_matrix.detach()
    zf = z.detach().to(dtype=torch.float32)
    Ef = E.to(dtype=torch.float32)
    V = Ef.shape[0]
    vocab_chunk = int(vocab_chunk or V)
    best_score = torch.full((zf.shape[0],), -float("inf"), device=zf.device)
    best_ids = torch.zeros((zf.shape[0],), dtype=torch.long, device=zf.device)
    disallowed = None
    if not_allowed is not None and len(not_allowed):
        disallowed = not_allowed.to(device=zf.device, dtype=torch.long)

    if metric not in ("l2", "cosine"):
        raise ValueError(f"unknown nearest-neighbor metric: {metric!r}")
    z_norm = torch.nn.functional.normalize(zf, dim=-1) if metric == "cosine" else None

    for start in range(0, V, vocab_chunk):
        stop = min(start + vocab_chunk, V)
        chunk = Ef[start:stop]
        if metric == "cosine":
            scores = z_norm @ torch.nn.functional.normalize(chunk, dim=-1).T
        else:
            # Maximize negative squared distance: -||z-e||^2.
            scores = 2 * (zf @ chunk.T) - chunk.pow(2).sum(-1).unsqueeze(0)
        if disallowed is not None:
            m = (disallowed >= start) & (disallowed < stop)
            if torch.any(m):
                scores[:, disallowed[m] - start] = -float("inf")
        vals, idx = scores.max(dim=-1)
        improve = vals > best_score
        best_score[improve] = vals[improve]
        best_ids[improve] = idx[improve] + start
    return best_ids


def straight_through_project(z, embed_matrix, *, not_allowed=None, metric="l2",
                             vocab_chunk=None):
    """Hard nearest-neighbor forward, identity-gradient backward."""
    ids = nearest_token_ids(
        z, embed_matrix, not_allowed=not_allowed, metric=metric,
        vocab_chunk=vocab_chunk,
    )
    hard = embed_matrix[ids].to(dtype=z.dtype)
    return z + (hard - z).detach(), ids


def init_prompt_embeddings(tokenizer, embed_matrix, length, *, seed,
                           not_allowed=None, init="random", noise_std=0.0):
    """Seeded PEZ initialization at vocabulary embeddings."""
    device = embed_matrix.device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    if init == "x":
        toks = tokenizer.encode(" x", add_special_tokens=False)
        base = toks[-1] if toks else (tokenizer.eos_token_id or 0)
        ids = torch.full((length,), int(base), dtype=torch.long)
    elif init == "random":
        if not_allowed is not None and len(not_allowed):
            mask = torch.ones(embed_matrix.shape[0], dtype=torch.bool)
            mask[not_allowed.cpu()] = False
            allowed = mask.nonzero().flatten()
            ids = allowed[torch.randint(len(allowed), (length,), generator=g)]
        else:
            ids = torch.randint(embed_matrix.shape[0], (length,), generator=g)
    else:
        raise ValueError(f"unknown PEZ init: {init!r}")
    z = embed_matrix[ids.to(device)].detach().clone().to(dtype=torch.float32)
    if noise_std:
        z = z + float(noise_std) * torch.randn(
            z.shape, generator=g, device="cpu", dtype=torch.float32,
        ).to(device)
    return z, ids.to(device)


def run_pez(objective, model, tokenizer, embed_matrix, *, cfg, seed,
            split="train", select_split="train"):
    """One PEZ run over the system-prompt slot."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = embed_matrix.device
    L = objective.slot_sizes[0]
    n_train = len(objective.examples_by_split[split])
    n_sel_full = len(objective.examples_by_split[select_split])

    num_steps = int(cfg.get("num_steps", 500))
    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    grad_clip = cfg.get("grad_clip", 1.0)
    grad_clip = None if grad_clip is None else float(grad_clip)
    mb = min(int(cfg.get("mini_batch_size", 8)), n_train)
    train_batch = min(int(cfg.get("train_batch_size", 32)), n_train)
    select_n = min(int(cfg.get("select_n", 256)), n_sel_full)
    eval_chunk = int(cfg.get("eval_chunk", 16))
    eval_every = int(cfg.get("eval_every", 10))
    print_every = int(cfg.get("print_every", 50))
    metric = str(cfg.get("metric", "l2"))
    vocab_chunk = cfg.get("vocab_chunk")
    vocab_chunk = None if vocab_chunk is None else int(vocab_chunk)
    allow_non_ascii = bool(cfg.get("allow_non_ascii", False))

    not_allowed = None
    if not allow_non_ascii:
        not_allowed = nonascii_token_ids(tokenizer, device=device)

    z0, _ = init_prompt_embeddings(
        tokenizer, embed_matrix, L, seed=seed, not_allowed=not_allowed,
        init=str(cfg.get("init", "random")),
        noise_std=float(cfg.get("init_noise_std", 0.01)),
    )
    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=weight_decay)

    g_mb = torch.Generator()
    g_mb.manual_seed(seed + 1)
    g_sel = torch.Generator()
    g_sel.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g_sel).tolist()[:select_n]
    hard_loss_accepts_indices = (
        "indices" in inspect.signature(objective.hard_loss).parameters
    )

    def projected():
        z_st, _ids = straight_through_project(
            z, embed_matrix, not_allowed=not_allowed, metric=metric,
            vocab_chunk=vocab_chunk,
        )
        return z_st.to(embed_matrix.dtype)

    def score_text(text):
        kw = {"mini_batch_size": eval_chunk}
        if hard_loss_accepts_indices:
            kw["indices"] = sel_idx
        return float(objective.hard_loss(text, select_split, **kw))

    trajectory = []
    n_scored = 0
    best_text = ""
    best_sel = float("inf")

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        mb_idx = torch.randperm(n_train, generator=g_mb).tolist()[:train_batch]
        objective.loss(projected, split=split, indices=mb_idx, backward=True,
                       mini_batch_size=mb)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
        optimizer.step()

        if step % eval_every == 0 or step == num_steps - 1:
            with torch.no_grad():
                ids = nearest_token_ids(
                    z, embed_matrix, not_allowed=not_allowed, metric=metric,
                    vocab_chunk=vocab_chunk,
                )
            text = _decode(tokenizer, ids)
            sel = score_text(text)
            n_scored += 1
            trajectory.append((n_scored, text, sel))
            if sel < best_sel:
                best_sel, best_text = sel, text
        if step % print_every == 0:
            print(f"  step {step}: sel_loss(best {best_sel:.4f}) "
                  f"proposals={n_scored} slot={best_text[:50]!r}", flush=True)

    print(f"PEZ winner (fixed {select_n}-subset): select={best_sel:.4f}  "
          f"slot={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_select_score": best_sel,
        "trajectory": trajectory,
        "n_proposals": n_scored,
        "n_steps": num_steps,
        "slot_len": L,
        "select_split": select_split,
        "metric": metric,
    }


def pez_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42):
    """Shared-contract entry point. cfg = the `pez` config block."""
    return run_pez(objective, model, tokenizer, embed_matrix, cfg=cfg, seed=seed)
