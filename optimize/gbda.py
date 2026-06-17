"""GBDA (Gradient-based Distributional Attack) for dataset-NLL prompt recovery —
a clean-room reimplementation of Guo et al. 2021 ("Gradient-based Adversarial
Attacks against Text Transformers", arXiv:2104.13733). No third-party code is
imported; every function here is ours and is covered by tests/test_gbda.py. The
authors' reference is `facebookresearch/text-adversarial-attack`
(`whitebox_attack.py`); `# src:` comments mark the lines each construct mirrors.

The algorithm, unchanged from the paper:
  Instead of optimizing discrete tokens, GBDA optimizes a DISTRIBUTION over
  token sequences and minimizes the EXPECTED loss under it — which is
  differentiable. A free parameter matrix `log_coeffs ∈ R^{L×V}` defines a
  per-position categorical `softmax(log_coeffs_i)`. Each step:
   1. draw a SOFT sample on the simplex via the Gumbel-softmax (Concrete) trick,
      `coeffs = softmax((log_coeffs + g) / tau)` with fresh Gumbel noise `g`;
   2. feed it to the frozen model as a SOFT embedding `coeffs @ E` (a convex
      combination of vocab embeddings) and compute the task loss;
   3. add a FLUENCY term — the causal NLL of the soft sample under a reference
      LM (`log_perplexity`) — to keep the text natural;
   4. Adam step on `log_coeffs`.
  At the end, read out a hard sequence by drawing Gumbel-`argmax` samples and
  keeping the best.

Two task-specific choices (this is recovery, not a single-sentence attack —
the SAME adaptations PGD/GCG make here, see GBDA_FAITHFUL.md):
  * LOSS — the adversarial CW-margin term becomes the per-token-mean NLL of a
    *dataset* of (scenario, response) pairs under the recovered system prompt,
    i.e. `NLLObjective.loss`. Each step draws a fresh dataset minibatch.
  * REF LM — the authors use a separate GPT-2 for `log_perplexity`; we reuse
    M_base (same vocab → the `shift_logits[..., :V]` crop is a no-op; no second
    model; same choice as PGD's control-CE fluency prior). `lam_perp=0` ablates
    the fluency term (and skips the ref forward — used by the CPU recovery test).

Because the slot has no "clean input" to anchor at (we are RECOVERING the
prompt, not perturbing a known one), `log_coeffs` is initialized at a RANDOM
allowed-token sequence with `initial_coeff` (the authors anchor at the real
input). Selection/winner uses `hard_loss` (the TEXT-path NLL = the reported
metric) on a FIXED train subset, matching SALVE/LARGO/GCG/PGD/OPRO so every
method selects on ONE uniform metric.
"""
import inspect

import torch
import torch.nn.functional as F

from optimize.gcg import nonascii_token_ids


def _decode(tokenizer, ids):
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids)


# --------------------------------------------------------------------------
# Pure token-space helpers (no model forward; unit-tested in tests/test_gbda.py).
# --------------------------------------------------------------------------
def gumbel_like(logits, generator=None):
    """i.i.d. Gumbel(0,1) noise shaped like `logits`. Computed exactly as
    torch.nn.functional.gumbel_softmax does internally — `-log(E)` with
    `E ~ Exp(1)` equals `-log(-log(U))`, `U ~ Uniform(0,1)`, the standard Gumbel
    (src: F.gumbel_softmax `gumbels = -empty_like(logits).exponential_().log()`)."""
    e = torch.empty_like(logits).exponential_(1.0, generator=generator)
    return -torch.log(e)


def gumbel_softmax_coeffs(logits, tau, noise):
    """Soft Gumbel-softmax sample `softmax((logits + g) / tau)` on the per-row
    simplex (src: whitebox_attack.py `F.gumbel_softmax(log_coeffs, hard=False)`,
    default `tau=1.0`). `noise` is passed in (drawn once per Monte-Carlo sample)
    so gradient-accumulation chunks reuse the SAME sample while recomputing the
    graph from the `log_coeffs` leaf — exactly PGD's `z_fn` trick."""
    return torch.softmax((logits + noise) / tau, dim=-1)


def log_perplexity(logits, coeffs):
    """GBDA fluency term: the soft causal cross-entropy of `coeffs` under a
    reference LM's `logits` (src: whitebox_attack.py `log_perplexity`):

        shift_logits = logits[:, :-1]; shift_coeffs = coeffs[:, 1:]
        shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]   # vocab crop
        return -(shift_coeffs * log_softmax(shift_logits)).sum(-1).mean()

    The crop handles a ref vocab larger than the attack vocab; with ref = M_base
    (same vocab) it is a no-op, kept for faithfulness. Position t's predicted
    distribution is scored against position t+1's SOFT token — penalizing
    low-likelihood (non-fluent) sequences."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def init_log_coeffs(init_ids, vocab_size, initial_coeff, device,
                    dtype=torch.float32):
    """`log_coeffs` init (src: whitebox_attack.py):

        log_coeffs = torch.zeros(L, V)
        log_coeffs[i, input_ids[i]] = args.initial_coeff   # default 15

    `softmax(initial_coeff)` puts ~`e^c / (e^c + V - 1)` mass on each init token
    (c=15 → ~0.96 at V≈152k), a near-one-hot warm start. The authors anchor at
    the clean input being perturbed; recovery has none, so `init_ids` is a random
    ALLOWED-token sequence (the caller draws it). Returned tensor has no grad —
    caller sets `requires_grad_()`."""
    L = len(init_ids)
    lc = torch.zeros(L, int(vocab_size), device=device, dtype=dtype)
    lc[torch.arange(L), torch.as_tensor(init_ids, device=device)] = float(initial_coeff)
    return lc


# --------------------------------------------------------------------------
# The optimization loop (the model-touching part).
# --------------------------------------------------------------------------
def run_gbda(objective, model, tokenizer, embed_matrix, *, cfg, seed,
             split="train", select_split="train"):
    """One GBDA run over the system-prompt slot. `objective` must be built with
    n_learnable = the desired slot length L.

    Per step: resample ONE dataset minibatch (`train_batch_size`, accumulated
    over `mini_batch_size` chunks for memory); for each of `gumbel_samples_per_step`
    Gumbel draws, accumulate the dataset-NLL gradient (weight 1) + `lam_perp` ×
    the fluency gradient; mean over the draws; Adam step. Every `eval_every`
    steps score the deterministic `argmax(log_coeffs)` slot on a FIXED train
    subset via `hard_loss` (comparable trajectory + running-argmin winner). After
    the loop, draw `final_gumbel_samples` HARD Gumbel samples and score them too
    (the authors' end-of-run extraction); the winner is the best over both pools.
    val/test are never touched here (reported later by hard_loss)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = embed_matrix.device
    V = embed_matrix.shape[0]
    L = objective.slot_sizes[0]
    n_train = len(objective.examples_by_split[split])
    n_sel_full = len(objective.examples_by_split[select_split])

    num_iters = int(cfg.get("num_iters", 300))
    lr = float(cfg.get("lr", 0.3))                        # src: Adam lr default 3e-1
    tau = float(cfg.get("gumbel_tau", 1.0))               # src: F.gumbel_softmax default
    initial_coeff = float(cfg.get("initial_coeff", 15.0))  # src: default 15
    lam_perp = float(cfg.get("lam_perp", 1.0))            # src: lam_perp default 1
    num_samples = int(cfg.get("gumbel_samples_per_step", 4))  # src: batch_size default 10
    mb = min(int(cfg.get("mini_batch_size", 8)), n_train)     # with-grad memory chunk
    train_batch = min(int(cfg.get("train_batch_size", 32)), n_train)  # effective grad batch
    eval_chunk = int(cfg.get("eval_chunk", 16))
    eval_every = int(cfg.get("eval_every", 5))
    print_every = int(cfg.get("print_every", 10))
    select_n = min(int(cfg.get("select_n", 256)), n_sel_full)
    final_gumbel_samples = int(cfg.get("final_gumbel_samples", 100))  # src: gumbel_samples default 100
    allow_non_ascii = bool(cfg.get("allow_non_ascii", False))

    # Vocab mask: legibility parity with GCG/PGD (the original GBDA is full-vocab
    # and relies on the fluency term; we mask non-ascii so the argmax slot stays
    # legible text and the comparison is apples-to-apples — see GBDA_FAITHFUL.md).
    not_allowed = (None if allow_non_ascii
                   else nonascii_token_ids(tokenizer, device=device))
    mask = None
    if not_allowed is not None and len(not_allowed):
        mask = torch.zeros(V, dtype=torch.bool, device=device)
        mask[not_allowed] = True

    def masked(lc):
        # masked_fill is in-graph: masked columns get a constant (grad 0 there),
        # so Adam never moves them and they never win the argmax / gumbel sample.
        return lc.masked_fill(mask, -1e4) if mask is not None else lc

    # Init log_coeffs at a random ALLOWED-token sequence (no clean input in
    # recovery). Seeded on CPU so the init is reproducible across devices.
    g_init = torch.Generator(); g_init.manual_seed(seed)
    if mask is not None:
        allowed_ids = (~mask).nonzero().flatten().cpu()
        init_ids = allowed_ids[torch.randint(len(allowed_ids), (L,),
                                             generator=g_init)].tolist()
    else:
        init_ids = torch.randint(V, (L,), generator=g_init).tolist()
    log_coeffs = init_log_coeffs(init_ids, V, initial_coeff, device)
    log_coeffs.requires_grad_(True)
    optimizer = torch.optim.Adam([log_coeffs], lr=lr)   # src: Adam([log_coeffs], lr)

    E = embed_matrix
    # Fixed selection subset (same every step): comparable trajectory + winner.
    # hard_loss(indices=) = TEXT path = the reported NLL (uniform across methods).
    g_sel = torch.Generator(); g_sel.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g_sel).tolist()[:select_n]
    g_mb = torch.Generator(); g_mb.manual_seed(seed + 1)
    hard_loss_accepts_indices = (
        "indices" in inspect.signature(objective.hard_loss).parameters)

    def score_text(text):
        kw = {"mini_batch_size": eval_chunk}
        if hard_loss_accepts_indices:
            kw["indices"] = sel_idx
        return float(objective.hard_loss(text, select_split, **kw))

    trajectory = []          # (cumulative scored candidates, text, sel_loss)
    n_scored = 0
    best_text = _decode(tokenizer, masked(log_coeffs.detach()).argmax(-1))
    best_sel = float("inf")

    for step in range(num_iters):
        optimizer.zero_grad()
        mb_idx = torch.randperm(n_train, generator=g_mb).tolist()[:train_batch]
        for _s in range(num_samples):
            noise = gumbel_like(log_coeffs).detach()      # fresh Gumbel draw, no grad

            def z_fn(noise=noise):                        # fresh graph per accum chunk
                coeffs = gumbel_softmax_coeffs(masked(log_coeffs), tau, noise)
                return coeffs.to(E.dtype) @ E             # (L, d) soft embedding
            # adversarial term = dataset NLL (weight 1); backprops into log_coeffs.grad
            objective.loss(z_fn, split=split, indices=mb_idx, backward=True,
                           mini_batch_size=mb)
            if lam_perp > 0:                              # fluency term
                coeffs = gumbel_softmax_coeffs(masked(log_coeffs), tau, noise)
                ref_logits = model(
                    inputs_embeds=(coeffs.to(E.dtype) @ E).unsqueeze(0)
                ).logits.float()                         # (1, L, V) — ref = M_base
                perp = lam_perp * log_perplexity(ref_logits, coeffs.unsqueeze(0))
                perp.backward()                          # accumulate into log_coeffs.grad
        if log_coeffs.grad is not None:                  # mean over Gumbel draws
            log_coeffs.grad /= num_samples
        optimizer.step()

        if step % eval_every == 0 or step == num_iters - 1:
            text = _decode(tokenizer, masked(log_coeffs.detach()).argmax(-1))
            sel = score_text(text); n_scored += 1
            trajectory.append((n_scored, text, sel))
            if sel < best_sel:
                best_sel, best_text = sel, text
        if step % print_every == 0:
            print(f"  step {step}: sel_loss(best {best_sel:.4f}) "
                  f"proposals={n_scored} slot={best_text[:50]!r}", flush=True)

    # End-of-run extraction: draw HARD Gumbel samples and keep the best
    # (src: whitebox_attack.py `for j in range(gumbel_samples): adv_ids =
    # F.gumbel_softmax(log_coeffs, hard=True).argmax(1)`). hard=True's argmax
    # equals argmax(log_coeffs + g) since softmax is monotone — we sample the
    # argmax directly and score each on the fixed subset (no binary "attack
    # success" in recovery → pick the lowest-NLL sample).
    with torch.no_grad():
        for _j in range(final_gumbel_samples):
            ids = (masked(log_coeffs) + gumbel_like(log_coeffs)).argmax(-1)
            text = _decode(tokenizer, ids)
            sel = score_text(text); n_scored += 1
            if sel < best_sel:
                best_sel, best_text = sel, text

    print(f"GBDA winner (fixed {select_n}-subset): select={best_sel:.4f}  "
          f"slot={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_select_score": best_sel,
        "trajectory": trajectory,
        "n_proposals": n_scored,
        "n_steps": num_iters,
        "slot_len": L,
        "select_split": select_split,
        "lam_perp": lam_perp,
    }


def gbda_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42):
    """Shared-contract entry point. cfg = the `gbda` config block.
    objective must be built with n_learnable = the desired slot length L."""
    return run_gbda(objective, model, tokenizer, embed_matrix, cfg=cfg, seed=seed)
