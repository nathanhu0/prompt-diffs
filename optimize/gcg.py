"""GCG (Greedy Coordinate Gradient) for prompt recovery against a dataset
objective — a clean-room reimplementation of the algorithm from Zou et al. 2023
("Universal and Transferable Adversarial Attacks on Aligned Language Models").
No third-party GCG package is imported; every function here is ours and is
covered by tests/test_gcg.py.

The algorithm, unchanged from the paper:
  each step, for the optimized token block,
   1. compute the gradient of the loss w.r.t. a one-hot encoding of the tokens
      (via `optim_onehot @ E`, so d loss/d onehot has shape [L, vocab]);
   2. at each position keep the top-k tokens by *steepest descent* (largest
      `-grad`), and sample `search_width` candidate blocks by replacing
      `n_replace` random positions with a random pick from that position's top-k;
   3. score every candidate with the loss and greedily take the argmin.

Two task-specific choices (this is recovery, not a single-prompt attack):
  * LOSS — the per-token-mean NLL of a *dataset* of (scenario, response) pairs
    under the recovered system prompt, i.e. `NLLObjective.loss`, instead of a
    single (prompt, target) cross-entropy. Each step draws a fresh minibatch of
    pairs; the gradient and the candidate scores use the SAME minibatch.
  * REGION — the optimized block is the SYSTEM-prompt slot (the objective's
    `n_learnable` slot), not a suffix on the user turn. So the gradient's
    `optim_embeds` is exactly the slot `z` that `NLLObjective.loss` composes as
    [prefix | z | suffix].

Because the minibatch is resampled per step, per-step losses are not
cross-comparable, so the returned winner is the trajectory candidate with the
lowest hard_loss (the TEXT-path NLL = the reported metric) on a FIXED train
subset (matching recover.py `select_split="train"`), not whatever the last step
happened to hold. The per-step GRADIENT + candidate swap still use
`NLLObjective.loss` (template path) on resampled minibatches; only the
best-tracking/winner selection uses hard_loss, so every method selects uniformly.
"""
import torch
import torch.nn.functional as F

from optimize.templates import compose_batch


# --------------------------------------------------------------------------
# Pure token-space helpers (no model forward; unit-tested in tests/test_gcg.py).
# --------------------------------------------------------------------------
def nonascii_token_ids(tokenizer, device="cpu"):
    """Vocab ids whose single-token decode is not printable-ASCII, plus the
    special tokens. GCG forbids these so the recovered prompt stays legible
    text rather than drifting into byte-fallback / control tokens."""
    bad = []
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid])
        if not (s.isascii() and s.isprintable()):
            bad.append(tid)
    for special in (tokenizer.bos_token_id, tokenizer.eos_token_id,
                    tokenizer.pad_token_id, tokenizer.unk_token_id):
        if special is not None:
            bad.append(special)
    return torch.tensor(sorted(set(bad)), device=device)


def sample_replacements(ids, grad, search_width, topk, n_replace=1,
                        not_allowed=None):
    """Sample `search_width` candidate token blocks from the token gradient.

    ids  : (L,)     current block
    grad : (L, V)   d loss / d one-hot   (lower grad => replacing toward it
                    decreases the loss, so we rank by `-grad`)
    Returns (search_width, L). Each row equals `ids` except at `n_replace`
    random positions, where the new token is a uniform pick from that
    position's top-k steepest-descent tokens. `not_allowed` ids are excluded
    by pushing their grad to +inf (so they never enter any top-k)."""
    L = ids.shape[0]
    if not_allowed is not None and len(not_allowed):
        grad = grad.clone()
        grad[:, not_allowed.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices            # (L, topk)
    cand = ids.unsqueeze(0).repeat(search_width, 1)         # (W, L)

    # n_replace distinct positions per candidate (argsort of noise = random perm)
    pos = torch.argsort(
        torch.rand(search_width, L, device=grad.device), dim=1)[:, :n_replace]
    # a random top-k slot per (candidate, replaced position)
    choice = torch.randint(0, topk, (search_width, n_replace), device=grad.device)
    new_tokens = topk_ids[pos, choice]                      # (W, n_replace)
    cand.scatter_(1, pos, new_tokens)
    return cand


def filter_retokenizable(ids, tokenizer):
    """Keep only candidates that survive decode -> re-encode unchanged, so the
    token block and its string form stay in sync (the same robustness check the
    GCG paper relies on)."""
    keep = []
    for row in ids:
        s = tokenizer.decode(row)
        re = tokenizer(s, add_special_tokens=False,
                       return_tensors="pt").input_ids[0].to(row.device)
        if re.shape == row.shape and torch.equal(re, row):
            keep.append(row)
    if not keep:
        raise RuntimeError(
            "No GCG candidate survives decode/re-encode; try allow_non_ascii "
            "or a different slot init.")
    return torch.stack(keep)


# --------------------------------------------------------------------------
# Dataset-NLL gradient + candidate scoring (the only model-touching parts).
# --------------------------------------------------------------------------
def token_gradient(objective, embed_matrix, optim_ids, mb_idx, split):
    """d (dataset-NLL) / d (one-hot of the slot tokens), shape (1, L, V).

    `optim_embeds = onehot @ E` is fed to `NLLObjective.loss` as the slot z, so
    the chain is onehot -> slot embeds -> composed sequence -> NLL.
    `objective.loss(backward=False)` returns a detached scalar (it backprops
    internally), so we call `backward=True` and read `onehot.grad`. The model is
    frozen, so the one-hot is the only grad-requiring leaf and `.grad` is exactly
    the GCG token gradient."""
    V = embed_matrix.shape[0]
    onehot = F.one_hot(optim_ids, num_classes=V).to(embed_matrix.dtype)  # (1,L,V)
    onehot.requires_grad_()
    optim_embeds = (onehot @ embed_matrix).squeeze(0)                    # (L,d)
    objective.loss(optim_embeds, split=split, indices=mb_idx, backward=True)
    return onehot.grad                                                   # (1,L,V)


@torch.no_grad()
def score_candidates(objective, embed_matrix, candidates, mb_idx, split, mb=None,
                     cand_chunk=None):
    """Per-candidate dataset-NLL on the step's minibatch (the same loss the
    gradient uses). Returns (n_candidates,).

    BATCHED over candidates: the M = len(mb_idx) example templates are FIXED
    across candidates (only the slot embeds differ), so we compose each
    candidate's slot into those M templates and run ONE model forward over
    `cand_chunk * M` sequences per chunk, instead of W sequential
    single-candidate forwards (the old hotspot). Numerically identical to the
    per-candidate `objective.loss(embed_matrix[cand], split, indices=mb_idx)`
    — same compose + cross-entropy + per-token mean over the minibatch (sum of
    target-token CE / sum of target tokens, across the M examples).

    Backwards compatible: signature is a superset of the old one (`mb` is
    accepted and ignored — M is set by `mb_idx`); `cand_chunk=1` reproduces the
    legacy one-candidate-per-forward path. `cand_chunk=None` auto-targets ~32
    sequences/forward (48G-safe, no-grad)."""
    model = objective.model
    examples = [objective.examples_by_split[split][i] for i in mb_idx]
    templates = [e.template for e in examples]
    tgts = [torch.tensor(e.target_ids, device=embed_matrix.device, dtype=torch.long)
            for e in examples]
    M, W = len(templates), len(candidates)
    if cand_chunk is None:
        cand_chunk = max(1, 32 // max(M, 1))
    losses = torch.empty(W)
    for c0 in range(0, W, cand_chunk):
        chunk = candidates[c0:c0 + cand_chunk]
        C = len(chunk)
        # Compose each candidate's M sequences. The M templates + slot length are
        # identical across candidates, so every compose_batch returns the same
        # (M, max_len, d) shape -> safe to cat into one (C*M, max_len, d) forward.
        embs, masks, tlens = [], [], []
        for row in chunk:                                   # row: (L,) token ids
            out = compose_batch(templates, embed_matrix[row], model)
            embs.append(out["inputs_embeds"])
            masks.append(out["attention_mask"])
            tlens.append(out["total_lens"])
        attn = torch.cat(masks, dim=0)
        total_lens = torch.cat(tlens, dim=0)                # (C*M,)
        logits = model(inputs_embeds=torch.cat(embs, dim=0),
                       attention_mask=attn).logits          # (C*M, max_len, V)
        # per-row CE over the trailing T_i target tokens; aggregate per candidate
        # (rows are candidate-major: row r -> candidate r//M, example r%M).
        sums = torch.zeros(C, device=logits.device)
        counts = torch.zeros(C, device=logits.device)
        for r in range(C * M):
            tt = tgts[r % M]
            T = tt.shape[0]
            ts = int(total_lens[r].item()) - T
            sums[r // M] += F.cross_entropy(logits[r, ts - 1: ts - 1 + T], tt,
                                            reduction="sum")
            counts[r // M] += T
        losses[c0:c0 + C] = (sums / counts).cpu()
    return losses


def _init_slot_ids(tokenizer, length, device):
    """Slot init: `length` copies of the space-prefixed ``" x"`` token — the
    GCG-standard ``"x x x ..."`` init, which BPE-tokenizes stably (each ``" x"``
    is one token) so the block round-trips through decode/encode. A bare ``"x"``
    repeat does NOT round-trip (the run merges), which would make
    filter_retokenizable drop every candidate. We assert the chosen init
    round-trips so a tokenizer quirk fails loudly here rather than mid-search."""
    toks = tokenizer.encode(" x", add_special_tokens=False)
    base = toks[-1] if toks else (tokenizer.eos_token_id or 0)
    ids = torch.tensor([[base] * length], device=device, dtype=torch.long)
    decoded = tokenizer.decode(ids[0])
    re = tokenizer(decoded, add_special_tokens=False,
                   return_tensors="pt").input_ids[0].to(device)
    assert re.shape[0] == length and torch.equal(re.cpu(), ids[0].cpu()), (
        f"slot init does not round-trip (got {re.shape[0]} ids != {length}); "
        f"pick a different init token")
    return ids


def run_gcg(objective, model, tokenizer, embed_matrix, *, cfg, seed,
            split="train", select_split="train"):
    """One GCG run over the system-prompt slot. `objective` must be built with
    n_learnable = the desired slot length L.

    Per step: resample ONE `mini_batch_size` dataset minibatch (the only deviation
    from single-target nanoGCG) and use it for BOTH the gradient (with-grad,
    memory-bound) and scoring the `search_width` candidates (no-grad); take the
    argmin swap; then score the chosen slot on a FIXED `select_n` train subset
    (same every step) for a clean, cross-step-comparable trajectory + the winner
    (running argmin). val/test are never touched here (reported later by hard_loss).
    At `n_replace=1` only one token changes per step, so `num_steps` must exceed L
    to actually sweep the slot."""
    torch.manual_seed(seed)
    device = model.device
    L = objective.slot_sizes[0]
    n_train = len(objective.examples_by_split[split])
    n_sel_full = len(objective.examples_by_split[select_split])
    # ONE fresh dataset minibatch per step, shared by the gradient AND candidate
    # scoring — the single deviation from nanoGCG (which is single-target: it uses
    # the same one behavior for both). With-grad bound, so keep it small.
    M = cfg.get("mini_batch_size", 4)
    W = cfg["search_width"]
    topk = cfg["topk"]
    n_replace = cfg.get("n_replace", 1)
    num_steps = cfg["num_steps"]
    proposal_cap = cfg.get("proposal_cap")
    eval_chunk = cfg.get("eval_chunk", 16)
    score_cand_chunk = cfg.get("score_cand_chunk")   # candidates/forward (None=auto ~32 seqs)
    select_n = min(cfg.get("select_n", 256), n_sel_full)
    not_allowed = (None if cfg.get("allow_non_ascii", False)
                   else nonascii_token_ids(tokenizer, device=device))

    # Fixed selection subset (same every step): comparable trajectory + winner.
    # Selection scores via hard_loss(indices=sel_idx) — the TEXT path = the
    # reported NLL — matching SALVE/LARGO/OPRO so every method selects on ONE
    # uniform metric. indices leaves the FULL "train" split intact for the
    # per-step gradient + candidate minibatches (no split mutation).
    g_sel = torch.Generator(); g_sel.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g_sel).tolist()[:select_n]

    optim_ids = _init_slot_ids(tokenizer, L, device)         # (1,L)
    g_mb = torch.Generator(); g_mb.manual_seed(seed)
    trajectory = []          # (cumulative_proposals, optim_str, sel_loss on fixed subset)
    n_proposals = 0
    best_text, best_sel = tokenizer.decode(optim_ids[0]), float("inf")

    for step in range(num_steps):
        mb_idx = torch.randperm(n_train, generator=g_mb).tolist()[:M]   # one fresh minibatch
        grad = token_gradient(objective, embed_matrix, optim_ids, mb_idx, split)
        cand = sample_replacements(
            optim_ids.squeeze(0), grad.squeeze(0), W, topk, n_replace,
            not_allowed=not_allowed)
        if cfg.get("filter_ids", True):
            cand = filter_retokenizable(cand, tokenizer)
        losses = score_candidates(objective, embed_matrix, cand, mb_idx, split,
                                  cand_chunk=score_cand_chunk)
        optim_ids = cand[int(losses.argmin())].unsqueeze(0)
        n_proposals += int(cand.shape[0])
        optim_str = tokenizer.decode(optim_ids[0])
        # Clean score on the FIXED selection subset via hard_loss (drives
        # best-tracking + the returned winner); text path = the reported metric.
        sel = float(objective.hard_loss(optim_str, select_split,
                                        indices=sel_idx, mini_batch_size=eval_chunk))
        trajectory.append((n_proposals, optim_str, sel))
        if sel < best_sel:
            best_sel, best_text = sel, optim_str
        print(f"  step {step}: sel_loss={sel:.4f} (best {best_sel:.4f}) "
              f"proposals={n_proposals} slot={optim_str[:50]!r}", flush=True)
        if proposal_cap and n_proposals >= proposal_cap:
            print(f"  proposal cap {proposal_cap} reached at step {step}")
            break

    print(f"GCG winner (fixed {select_n}-subset): select={best_sel:.4f}  "
          f"slot={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_select_score": best_sel,
        "trajectory": trajectory,
        "n_proposals": n_proposals,
        "n_steps": len(trajectory),
        "slot_len": L,
        "select_split": select_split,
    }


def gcg_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42):
    """Shared-contract entry point. cfg = the `gcg` config block.
    objective must be built with n_learnable = the desired slot length L."""
    return run_gcg(objective, model, tokenizer, embed_matrix, cfg=cfg, seed=seed)
