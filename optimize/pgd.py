"""PGD prompt recovery — adapter wiring our dataset-NLL-over-a-system-slot
objective into the faithful Geisler et al. optimizer in `optimize.pgd_geisler`.

The optimizer machinery (relaxation, simplex + Tsallis-q2 entropy-ceiling
projections, entropy anneal, cosine-warm-restart LR with the upward eta_min,
relaxation-gap + LR entropy coupling, per-row grad clip, random-simplex init,
argmax+round-trip discretization, patience reset-to-best) lives in
`pgd_geisler.GeislerPGD`, transcribed from the authors' code. This file builds
the three loss callbacks + the canonical config.

Data handling (our adaptation for a large recovery dataset):
  * GRADIENT — SGD on a random `train_batch_size`-example minibatch each step,
    realized via gradient ACCUMULATION over `mini_batch_size` chunks (so the
    effective batch is `train_batch_size` but peak memory is one chunk). The
    accumulation is exact because we pass a CALLABLE `z_fn` to objective.loss,
    which recomputes `normalize(S)@E` per chunk (fresh graph each chunk -> no
    double-backward).
  * SELECTION + GAP — every `eval_every` steps we score the RELAXED and the
    DISCRETE slot on a FIXED train eval subset (`eval_n`, sampled once). The
    DISCRETE score uses hard_loss (the TEXT path = the reported NLL, via
    `indices=`), matching SALVE/LARGO/OPRO/GCG so every method selects on ONE
    uniform metric; the RELAXED score stays on `NLLObjective.loss` (a simplex has
    no text form) and is only the gap diagnostic. The discrete score is the
    stable signal patience/best-tracking uses; the best discrete candidate on
    that subset is returned directly (no end-of-run re-scoring). Selection is on TRAIN
    (`select_split="train"`); val/test stay clean for reporting.

Per-step simplex RENORMALIZE-before-matmul (src: pgd_attack.py:478
prepare_embedding_factors): the gradient and the eval both feed
`normalize(S) @ E`, so the gradient flows through the normalize Jacobian exactly
as in the authors' code (not the raw `S @ E`).

THE LOSS (src: baselines/reinforce/pgd_loss.py defaults, active in
experiments/pgd_reinforce_ce.yaml — only target_weight is overridden there):

  combined = target_weight        * dataset_NLL(S)        # 0.84  -- the objective
           + control_weight       * control_CE(S)         # 0.007 ] AUXILIARY:
           + control_next_weight  * control_next_CE(S)    # 0.05  ] fluency /
           + control_nonrepeat_weight * nonrepeat(S)      # 0.01  ] diversity
           + entropy_weight       * entropy_q2_p6(S)      # 2e-4  ] priors

The first term is the real recovery objective (their "affirmative target" CE ->
our dataset NLL). The other four are MINIMAL auxiliary regularizers carried in
the canonical code:
  * control_CE / control_next_CE  -- a fluency prior: the slot tokens should be
    something the model itself finds likely. Their version conditions the
    control tokens on the behaviour; for a SYSTEM-PROMPT slot (which sits at the
    very start) the faithful analog is the model predicting the slot tokens
    given only the chat-template opening, i.e. one extra `[prefix | S@E]`
    forward. `control` carries the gradient through the logits, `control_next`
    through the soft tokens (src: pgd_loss.py:117-125).
  * nonrepeat  -- -mean||S[i]-S[i+1]||_1, penalizes repeating a token in
    adjacent positions (src: pgd_loss.py:135).
  * entropy_q2_p6  -- soft Tsallis-q2 entropy, p=6 aggregated; tiny (2e-4) and
    largely redundant with the entropy-ceiling PROJECTION (src: pgd_loss.py:216).

`aux_loss=False` zeroes all four auxiliary terms (target NLL + optimizer
machinery only) for the ablation arm of the sweep; the entropy-ceiling
projection is part of the optimizer and stays on in BOTH arms.
"""
import inspect

import torch
import torch.nn.functional as F

from optimize.pgd_geisler import GeislerPGD, tsallis_q2


def _aux_loss(s, model, embed_matrix, prefix_ids, weights):
    """The four auxiliary regularizers as a single graph tensor in S, given the
    RENORMALIZED slot simplex `s` (shape [L, V], in-graph). control_CE /
    control_next need slot-prediction logits from a `[prefix | s@E]` forward;
    nonrepeat / entropy are pure functions of s."""
    L = s.shape[0]
    aux = s.new_zeros(())

    cw, cnw = weights["control"], weights["control_next"]
    if (cw or cnw) and prefix_ids is not None:
        E = embed_matrix
        prefix_emb = E[torch.tensor(prefix_ids, device=E.device)]   # (P, d)
        slot_emb = s.to(E.dtype) @ E                                # (L, d)
        seq = torch.cat([prefix_emb, slot_emb], dim=0).unsqueeze(0)  # (1,P+L,d)
        logits = model(inputs_embeds=seq).logits[0].float()        # (P+L, V)
        P = prefix_emb.shape[0]
        slot_logits = logits[P - 1: P - 1 + L]                     # (L, V) -> predicts slot
        logp = F.log_softmax(slot_logits, dim=-1)
        if cw:        # gradient through the logits (soft target detached)
            aux = aux + cw * (-(s.detach() * logp).sum(-1)).mean()
        if cnw:       # gradient through the soft tokens (logits detached)
            logp_d = F.log_softmax(slot_logits.detach(), dim=-1)
            aux = aux + cnw * (-(s * logp_d).sum(-1)).mean()

    if weights["nonrepeat"]:
        nr = -torch.linalg.norm(s[:-1].detach() - s[1:], dim=-1, ord=1).mean()
        aux = aux + weights["nonrepeat"] * nr

    if weights["entropy"]:
        e = tsallis_q2(s)                                          # (L,)
        ent = ((e ** 6).mean()) ** (1.0 / 6.0)
        aux = aux + weights["entropy"] * ent
    return aux


def run_pgd(objective, model, tokenizer, embed_matrix, *, cfg, seed,
            split="train", select_split="train"):
    """One PGD run over the system-prompt slot. `objective` must be built with
    n_learnable = the desired slot length L. Returns the trajectory + the
    best discrete candidate tracked on the fixed train eval subset."""
    device = model.device
    V = embed_matrix.shape[0]
    L = objective.slot_sizes[0]
    n_train = len(objective.examples_by_split[split])
    mb = int(cfg.get("mini_batch_size", 8))               # with-grad memory chunk
    train_batch = min(int(cfg.get("train_batch_size", 32)), n_train)  # effective grad batch
    eval_n = min(int(cfg.get("eval_n", 256)), n_train)
    eval_chunk = int(cfg.get("eval_chunk", 64))           # no-grad eval chunk
    eps = torch.finfo(torch.float32).eps

    aux_on = bool(cfg.get("aux_loss", True))
    target_weight = float(cfg.get("target_weight", 0.84))
    aux_weights = {
        "control":      float(cfg.get("control_weight", 0.007)) if aux_on else 0.0,
        "control_next": float(cfg.get("control_next_weight", 0.05)) if aux_on else 0.0,
        "nonrepeat":    float(cfg.get("control_nonrepeat_weight", 0.01)) if aux_on else 0.0,
        "entropy":      float(cfg.get("entropy_weight", 2e-4)) if aux_on else 0.0,
    }
    prefix_ids = objective.examples_by_split[split][0].template.prefix_ids \
        if aux_on else None

    # Fixed train eval subset (sampled once): scored every eval_every steps for
    # the gap + best-tracking. select_split=train -> val/test stay clean. The
    # DISCRETE score (drives patience + the returned winner) uses
    # hard_loss(indices=) = the TEXT/reported metric; relaxed_eval stays on
    # objective.loss.
    g_sel = torch.Generator(); g_sel.manual_seed(seed)
    eval_idx = torch.randperm(n_train, generator=g_sel).tolist()[:eval_n]
    g_mb = torch.Generator(); g_mb.manual_seed(seed + 1)
    hard_loss_accepts_indices = (
        "indices" in inspect.signature(objective.hard_loss).parameters
    )

    def renorm(s):                                         # [L, V] -> sum-1, in-graph
        return s / torch.clamp_min(s.sum(-1, keepdim=True), eps)

    def grad_step_fn(S, step):
        idx = torch.randperm(n_train, generator=g_mb).tolist()[:train_batch]

        def z_fn():                                        # fresh graph per accum chunk
            return renorm(S[0]).to(embed_matrix.dtype) @ embed_matrix
        # one optimizer-step batch of `train_batch`, accumulated over `mb` chunks
        objective.loss(z_fn, split=split, indices=idx, backward=True,
                       mini_batch_size=mb)
        with torch.no_grad():
            S.grad.mul_(target_weight)                     # scale target -> 0.84
        if aux_on:
            aux = _aux_loss(renorm(S[0]), model, embed_matrix, prefix_ids, aux_weights)
            aux.backward()                                 # accumulate into S.grad

    @torch.no_grad()
    def relaxed_eval_fn(S):
        z = renorm(S[0]).to(embed_matrix.dtype) @ embed_matrix
        return float(objective.loss(z, split=split, indices=eval_idx,
                                    backward=False, mini_batch_size=eval_chunk))

    @torch.no_grad()
    def discrete_eval_fn(ids):
        # Selection/best-tracking on hard_loss (TEXT path = reported NLL) over the
        # fixed eval subset via indices= — uniform with SALVE/LARGO/OPRO/GCG. The
        # discrete ids round-trip (GeislerPGD discretizes by argmax+round-trip), so
        # decode -> hard_loss re-encode scores the same slot. relaxed_eval_fn stays
        # on objective.loss (a relaxed simplex has no text form). This re-tokenizes
        # the eval subset when called; cfg.eval_every throttles the expensive
        # train-selection pass.
        text = tokenizer.decode(ids[0], clean_up_tokenization_spaces=False)
        kwargs = {"mini_batch_size": eval_chunk}
        if hard_loss_accepts_indices:
            kwargs["indices"] = eval_idx
        return float(objective.hard_loss(text, select_split, **kwargs))

    # lr_scale scales the LR floor (lr) AND ceiling (eta_min) together — the only
    # schedule-shape-preserving knob: the ratio eta_min/lr is invariant, so the
    # cosine-cycle amplitude and the entropy coupling keep their shape; only the
    # raw step magnitude changes. (Robustness cells set 3 / 0.333.)
    lr_scale = float(cfg.get("lr_scale", 1.0))
    pgd = GeislerPGD(
        vocab_size=V, device=device, tokenizer=tokenizer, slot_len=L,
        grad_step_fn=grad_step_fn, relaxed_eval_fn=relaxed_eval_fn,
        discrete_eval_fn=discrete_eval_fn,
        learning_rate=lr_scale * float(cfg.get("lr", 0.11)),
        num_steps=int(cfg.get("num_steps", 5000)),
        grad_clip_value=float(cfg.get("grad_clip", 20.0)),
        anneal_duration=int(cfg.get("anneal_duration", 100)),
        anneal_end_entropy_factor=float(cfg.get("entropy_factor", 0.4)),
        lr_warmup_steps=int(cfg.get("warmup_steps", 100)),
        lr_restart_period=int(cfg.get("restart_period", 60)),
        lr_eta_min=lr_scale * float(cfg.get("eta_min", 0.325)),
        entropy_factor_scale_by_relaxation_gap=float(
            cfg.get("entropy_factor_scale_by_relaxation_gap", 0.1)),
        entropy_factor_alternate_scheduler=bool(
            cfg.get("entropy_factor_alternate_scheduler", True)),
        patience_value=int(cfg.get("patience_value", 100)),
        patience_mode=str(cfg.get("patience_mode", "reset_to_best")),  # reset_to_best | reinit | mix
        patience_reinit_prob=float(cfg.get("patience_reinit_prob", 0.5)),  # reinit prob in "mix"
        allow_non_ascii=bool(cfg.get("allow_non_ascii", False)),
        eval_every=int(cfg.get("eval_every", 1)),
        print_every=int(cfg.get("print_every", 50)),
        seed=seed,
    )
    out = pgd.run()
    best_text = tokenizer.decode(out["best_ids"][0],
                                 clean_up_tokenization_spaces=False)
    print(f"PGD winner (best discrete on {len(eval_idx)} fixed train-eval examples): "
          f"select={out['best_eval']:.4f}  slot={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_select_score": out["best_eval"],
        "trajectory": out["trajectory"],
        "n_proposals": out.get("n_proposals", len(out["trajectory"])),
        "n_steps": out.get("n_steps", len(out["trajectory"])),
        "n_scored": len(out["trajectory"]),
        "slot_len": L,
        "select_split": select_split,
        "aux_loss": aux_on,
    }


def pgd_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42):
    """Shared-contract entry point. cfg = the `pgd` config block. objective
    built with n_learnable = the desired slot length L."""
    return run_pgd(objective, model, tokenizer, embed_matrix, cfg=cfg, seed=seed)
