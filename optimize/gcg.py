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
def token_gradient(objective, embed_matrix, optim_ids, mb_idx, split,
                   fluency_weight=0.0, context_ids=None):
    """d (dataset-NLL [+ fluency_weight * block-fluency-NLL]) / d (one-hot of the
    slot tokens), shape (1, L, V).

    `optim_embeds = onehot @ E` is fed to `NLLObjective.loss` as the slot z, so
    the chain is onehot -> slot embeds -> composed sequence -> NLL.
    `objective.loss(backward=False)` returns a detached scalar (it backprops
    internally), so we call `backward=True` and read `onehot.grad`. The model is
    frozen, so the one-hot is the only grad-requiring leaf and `.grad` is exactly
    the GCG token gradient.

    With `fluency_weight > 0` (readable-GCG), we ALSO backprop the slot's own
    self-perplexity NLL through the SAME `onehot` leaf — autograd accumulates the
    fluency gradient into `onehot.grad` on top of the dataset gradient, so the
    top-k tokens are chosen against the combined objective. The dataset backward
    frees its graph, so we recompute a fresh `onehot @ E` for the fluency forward
    (same leaf, fresh graph)."""
    V = embed_matrix.shape[0]
    onehot = F.one_hot(optim_ids, num_classes=V).to(embed_matrix.dtype)  # (1,L,V)
    onehot.requires_grad_()
    optim_embeds = (onehot @ embed_matrix).squeeze(0)                    # (L,d)
    objective.loss(optim_embeds, split=split, indices=mb_idx, backward=True)
    if fluency_weight:
        flu_embeds = (onehot @ embed_matrix).squeeze(0)                  # fresh graph
        flu = block_fluency_nll(objective.model, embed_matrix, flu_embeds,
                                optim_ids.squeeze(0), context_ids)
        (fluency_weight * flu).backward()        # accumulates into onehot.grad
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


# --------------------------------------------------------------------------
# Fluency penalty (readable-GCG): the slot's own self-perplexity NLL under the
# base LM, conditioned on the chat-template prefix. Added to the SAME loss GCG
# differentiates (so it enters the top-k gradient) AND to candidate scoring —
# one objective L = dataset_nll + fluency_weight * block_fluency_nll, matching
# the readable-GCG / AutoDAN framing. fluency_weight=0 disables it entirely
# (bit-identical vanilla GCG).
#
# CONSISTENT WITH AutoDAN (optimize/autodan.py:next_token_nll): each position's
# term is exactly AutoDAN's per-token fluency -log p(token_i | context_ids +
# slot_<i) under the base LM, conditioned on the SAME context (the first
# template's prefix_ids — the chat-template/system-message prefix). AutoDAN is
# left-to-right so it scores one token at a time; this block version scores all
# L realized slot tokens in one forward and MEANS over them, keeping the
# per-token scale identical so `fluency_weight` lines up with AutoDAN's (0.3).
# --------------------------------------------------------------------------
def block_fluency_nll(model, embed_matrix, slot_input_embeds, slot_target_ids,
                      context_ids):
    """-log p(slot_i | context, slot_<i) averaged over the L slot positions.

    slot_input_embeds : (B, L, d) or (L, d) — the slot fed to the model. Pass
        `onehot @ E` (grad leaf) for the gradient stage, `embed_matrix[cand]`
        (no-grad) for candidate scoring.
    slot_target_ids   : (B, L) or (L,) — the discrete slot tokens whose log-prob
        we score. The CE targets; gradient flows only through slot_input_embeds.
    context_ids       : (C,) chat-template prefix ids prepended before the slot
        (bos/eos-fallback if empty, mirroring AutoDAN). Detached — no grad on it.
    Returns (B,) per-sequence mean NLL (or scalar if 1-D input)."""
    squeeze = slot_input_embeds.dim() == 2
    if squeeze:
        slot_input_embeds = slot_input_embeds.unsqueeze(0)
        slot_target_ids = slot_target_ids.unsqueeze(0)
    B, L, _ = slot_input_embeds.shape
    device = slot_input_embeds.device
    ctx = list(context_ids or [])
    if not ctx:                                 # AutoDAN's bos -> eos -> 0 fallback
        cfg = getattr(model, "config", None)
        fb = getattr(cfg, "bos_token_id", None) or getattr(cfg, "eos_token_id", None)
        ctx = [fb if fb is not None else 0]
    C = len(ctx)
    ctx_embeds = embed_matrix[torch.tensor(ctx, device=device)]          # (C,d)
    ctx_embeds = ctx_embeds.unsqueeze(0).expand(B, -1, -1)               # (B,C,d)
    seq = torch.cat([ctx_embeds, slot_input_embeds], dim=1)             # (B,C+L,d)
    logits = model(inputs_embeds=seq).logits                            # (B,C+L,V)
    # slot token at absolute position C+i is predicted by logits[C+i-1].
    pred = logits[:, C - 1: C - 1 + L]                                   # (B,L,V)
    nll = F.cross_entropy(pred.reshape(B * L, -1).float(),
                          slot_target_ids.reshape(B * L),
                          reduction="none").view(B, L).mean(dim=1)       # (B,)
    return nll.squeeze(0) if squeeze else nll


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


def fluency_at(step, target, warmup, ramp):
    """Fluency-weight schedule: 0 for the first `warmup` steps (pure GCG, so the
    block-swap can sweep off the low-perplexity ` x x x` induction fixed point
    before the penalty engages), then a linear ramp to `target` over `ramp`
    steps, then constant `target`. warmup=ramp=0 => constant `target` every step
    (so target=0 is bit-identical vanilla GCG)."""
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return target
    return target * min(1.0, (step - warmup) / ramp)


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
    # Warm-start support: cfg["init_ids"] = (1, L) LongTensor or length-L iterable.
    # Used by run_comparison's `init_from: gcg` polish chain so gcg_polish (warm
    # fluency) starts from vanilla GCG's `best_ids`. None => cold-start ` x x x`.
    init_ids = cfg.get("init_ids")
    # Readable-GCG fluency penalty: 0 => exact vanilla GCG. context_ids is the
    # chat-template/system prefix the slot follows (same source AutoDAN uses),
    # so the fluency conditioning is consistent across the two methods.
    fluency_weight = float(cfg.get("fluency_weight", 0.0))
    fluency_warmup = int(cfg.get("fluency_warmup_steps", 0))   # pure-GCG steps before the penalty engages
    fluency_ramp = int(cfg.get("fluency_ramp_steps", 0))       # linear 0->target ramp after warmup
    context_ids = list(objective.examples_by_split[split][0].template.prefix_ids or [])

    # Fixed selection subset (same every step): comparable trajectory + winner.
    # Selection scores via hard_loss(indices=sel_idx) — the TEXT path = the
    # reported NLL — matching SALVE/LARGO/OPRO so every method selects on ONE
    # uniform metric. indices leaves the FULL "train" split intact for the
    # per-step gradient + candidate minibatches (no split mutation).
    g_sel = torch.Generator(); g_sel.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g_sel).tolist()[:select_n]

    if init_ids is None:
        optim_ids = _init_slot_ids(tokenizer, L, device)         # (1,L)
    else:
        optim_ids = torch.as_tensor(init_ids, device=device, dtype=torch.long)
        if optim_ids.dim() == 1:
            optim_ids = optim_ids.unsqueeze(0)
        assert optim_ids.shape == (1, L), \
            f"init_ids shape {tuple(optim_ids.shape)} != (1, {L})"
        # Round-trip check (same as _init_slot_ids): if the init doesn't round-trip
        # through decode/encode the candidate filter would drop every swap of it
        # silently. Loud failure here is much better.
        decoded = tokenizer.decode(optim_ids[0])
        re_ids = tokenizer(decoded, add_special_tokens=False,
                           return_tensors="pt").input_ids[0]
        assert re_ids.shape[0] == L and torch.equal(re_ids.cpu(), optim_ids[0].cpu()), (
            f"init_ids does not round-trip through tokenizer (re-encoded to "
            f"{re_ids.shape[0]} ids != {L}). text={decoded!r}")
        print(f"  [init_ids] warm-start from supplied ids; slot={decoded[:80]!r}",
              flush=True)
    g_mb = torch.Generator(); g_mb.manual_seed(seed)
    trajectory = []          # (cumulative_proposals, optim_str, sel_loss on fixed subset)
    ppl_traj = []            # per-step block-perplexity NLL of the chosen slot
    n_proposals = 0
    # Winner = argmin of the TARGET combined objective `sel + fluency_weight*ppl`,
    # GATED to FULL-strength-fluency steps. The gate is load-bearing both ways:
    # (a) it excludes the warmup pure-GCG prompts the penalty never touched, and
    # (b) without it `sel + w*ppl` collapses to the near-` x x x` init — repetition
    # is the GLOBAL perplexity minimum (ppl ~1.5) under induction, so an ungated
    # argmin reselects the degenerate slot the search escaped. `any_*` (best sel,
    # any step) is the fallback if the schedule leaves no full-strength step.
    # Vanilla GCG (fluency_weight=0): ppl term vanishes, gate is every step, so
    # this reduces to the original best-sel selection.
    best_text, best_obj, best_sel, best_ppl, best_ids = (
        tokenizer.decode(optim_ids[0]), float("inf"), float("inf"), float("nan"), optim_ids)
    any_text, any_sel, any_ids = best_text, float("inf"), optim_ids

    for step in range(num_steps):
        fw = fluency_at(step, fluency_weight, fluency_warmup, fluency_ramp)
        mb_idx = torch.randperm(n_train, generator=g_mb).tolist()[:M]   # one fresh minibatch
        grad = token_gradient(objective, embed_matrix, optim_ids, mb_idx, split,
                              fluency_weight=fw, context_ids=context_ids)
        cand = sample_replacements(
            optim_ids.squeeze(0), grad.squeeze(0), W, topk, n_replace,
            not_allowed=not_allowed)
        if cfg.get("filter_ids", True):
            cand = filter_retokenizable(cand, tokenizer)
        losses = score_candidates(objective, embed_matrix, cand, mb_idx, split,
                                  cand_chunk=score_cand_chunk)
        if fw:                                   # combined objective at scoring too
            with torch.no_grad():
                flu = block_fluency_nll(model, embed_matrix, embed_matrix[cand],
                                        cand, context_ids)
            losses = losses + fw * flu.cpu()
        optim_ids = cand[int(losses.argmin())].unsqueeze(0)
        n_proposals += int(cand.shape[0])
        optim_str = tokenizer.decode(optim_ids[0])
        # Clean score on the FIXED selection subset via hard_loss (the TEXT path =
        # the reported dataset NLL) + the chosen slot's own block-perplexity; the
        # winner minimizes `sel + fluency_weight*ppl` over full-strength steps.
        sel = float(objective.hard_loss(optim_str, select_split,
                                        indices=sel_idx, mini_batch_size=eval_chunk))
        with torch.no_grad():
            ppl = float(block_fluency_nll(model, embed_matrix,
                                          embed_matrix[optim_ids.squeeze(0)],
                                          optim_ids.squeeze(0), context_ids))
        trajectory.append((n_proposals, optim_str, sel)); ppl_traj.append(ppl)
        obj_val = sel + fluency_weight * ppl
        if sel < any_sel:
            any_sel, any_text, any_ids = sel, optim_str, optim_ids
        if fw >= fluency_weight and obj_val < best_obj:    # full-strength only
            best_obj, best_sel, best_ppl, best_text, best_ids = (
                obj_val, sel, ppl, optim_str, optim_ids)
        print(f"  step {step}: sel_loss={sel:.4f} ppl={ppl:.2f} total={obj_val:.4f} "
              f"(best {best_obj:.4f}) fw={fw:.2f} proposals={n_proposals} "
              f"slot={optim_str[:50]!r}", flush=True)
        if proposal_cap and n_proposals >= proposal_cap:
            print(f"  proposal cap {proposal_cap} reached at step {step}")
            break

    if best_obj == float("inf"):     # schedule never reached full strength: fall back
        print("  [warn] no full-strength-fluency step; falling back to best-sel argmin")
        i = min(range(len(trajectory)), key=lambda j: trajectory[j][2])
        best_sel, best_text, best_ppl = trajectory[i][2], trajectory[i][1], ppl_traj[i]
        best_ids = tokenizer(best_text, add_special_tokens=False,
                             return_tensors="pt").input_ids.to(device)   # (1, L), round-tripped above
    print(f"GCG winner (fixed {select_n}-subset): select={best_sel:.4f}  "
          f"block_ppl_nll={best_ppl:.4f}  slot={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_ids": best_ids.detach().cpu(),    # (1, L) — consumed by `init_from` warm-starts
        "best_select_score": best_sel,
        "best_total": best_obj,
        "trajectory": trajectory,
        "ppl_traj": ppl_traj,
        "n_proposals": n_proposals,
        "n_steps": len(trajectory),
        "slot_len": L,
        "select_split": select_split,
        "fluency_weight": fluency_weight,
        "fluency_warmup_steps": fluency_warmup,
        "fluency_ramp_steps": fluency_ramp,
        "block_ppl": best_ppl,
    }


def gcg_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42):
    """Shared-contract entry point. cfg = the `gcg` config block.
    objective must be built with n_learnable = the desired slot length L."""
    return run_gcg(objective, model, tokenizer, embed_matrix, cfg=cfg, seed=seed)
