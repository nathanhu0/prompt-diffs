"""Faithful port of the Geisler et al. 2024 PGD optimizer machinery
(arXiv:2402.09154), transcribed from the authors' official implementation
`sigeisler/reinforce-attacks-llms` (`baselines/reinforce/pgd_attack.py` +
`pgd_utils.py`). This module is the *vendored optimizer*: every projection,
the entropy anneal, the LR schedule, the dynamic entropy coupling, the per-row
gradient clip, the random-simplex init, the argmax+round-trip discretization,
and the patience reset-to-best are kept as close to their source as possible,
with `# src:` provenance comments pointing at the original line numbers.

What is NOT here (lives in the adapter `optimize/pgd.py`): the loss. Their loop
is hardwired to a single (prompt, target) CE through their `PromptManager` /
`PGDLoss`; we inject it via three callbacks so the same optimizer drives our
dataset-NLL-over-a-system-slot objective:

  grad_step_fn(S, step) -> None      # takes ONE gradient step's worth of work on
      the RELAXED simplex S ([1, L, V], requires_grad): samples a random train
      minibatch (with gradient accumulation to the configured effective batch),
      computes the combined loss (target NLL + aux), calls .backward() so S.grad
      is populated. Returns nothing (the optimizer step happens in run()).
  relaxed_eval_fn(S) -> float        # no-grad NLL of the RELAXED slot on a FIXED
      train eval subset (the gap numerator), called every eval_every steps.
  discrete_eval_fn(ids) -> float     # no-grad NLL of the argmax-discretized slot
      ids ([1, L]) on the SAME fixed train eval subset (gap denominator + the
      stable signal that patience/best-tracking uses), called every eval_every
      steps.

Selection: unlike the authors' tiny per-attack dataset (full-batch every step),
our recovery dataset is large, so the gradient is SGD on random minibatches while
the gap + best-tracking are evaluated periodically on a fixed train subset. The
best discrete candidate on that fixed subset is tracked across the run and
returned directly — no end-of-run re-scoring of many candidates.

Single-stream only (n_prompts = 1): we recover ONE system prompt for ONE
dataset, so the authors' batch-of-behaviors machinery degenerates to batch 1.
Consequently the patience cross-prompt "best-mix" (their `patience_mix_*`) has no
analog and is dropped by design; we keep the within-stream **reset-to-best**
(src: pgd_utils.py:363 patience_maybe_retrieve, reset_to_best branch).

ONE opt-in NON-SRC extension (all other lines stay faithful): `patience_mode`.
Default "reset_to_best" reproduces the src branch above exactly. "reinit" instead
random-restarts on stall (fresh simplex + Adam + LR warmup/cosine + entropy-anneal
re-ramp) to escape a basin. "mix" is the faithful single-stream analog of the src
50/50 (reset-to-own-best vs cross-prompt best-mix): since there is no other stream,
it resets-to-best half the time and reinits the other half (patience_reinit_prob).
The global best is always tracked and returned, so a restart can only help. Every
extension line is tagged `NON-SRC (ours)`.

Canonical constants (src: pgd_attack.py:38-79 __init__ defaults +
experiments/pgd_reinforce_ce.yaml): lr 0.11, num_steps 5000, grad_clip 20
(token_norm), entropy_factor 0.4 annealed 0->0.4 over duration 100,
simplex 'sort', tsallis_q2 iter 1, LR ConstantLR(100)->CosineAnnealingWarmRestarts
(T_0=60, eta_min=0.325), entropy_factor_scale_by_relaxation_gap 0.1,
entropy_factor_alternate_scheduler True, patience value 100, init 'random'.
"""
import torch


# --------------------------------------------------------------------------
# Disallowed tokens (src: pgd_utils.py:86 get_nonascii_toks). Non-printable /
# non-ascii ids + specials; excluded from the simplex support so the slot is
# legible text and never an EOS/PAD mid-prompt. We add the Llama-3 reserved
# block exactly as they do.
# --------------------------------------------------------------------------
def nonascii_token_ids(tokenizer, device="cpu"):
    bad = []
    for i in range(3, tokenizer.vocab_size):
        s = tokenizer.decode([i])
        if not (s.isascii() and s.isprintable()):
            bad.append(i)
    for special in (tokenizer.bos_token_id, tokenizer.eos_token_id,
                    tokenizer.pad_token_id, tokenizer.unk_token_id):
        if special is not None:
            bad.append(special)
    name = getattr(tokenizer, "name_or_path", "")
    if "Meta-Llama-3" in name or "Llama-3" in name:
        bad += list(range(128000, min(128256, tokenizer.vocab_size)))
    return torch.tensor(sorted(set(bad)), device=device)


def tsallis_q2(p):
    """Tsallis q=2 entropy = Gini index = 1 - sum p^2 (src: pgd_utils.py:52,
    q=2 branch). 0 for one-hot, ->1 for uniform."""
    return 1.0 - (p ** 2).sum(-1)


class GeislerPGD:
    def __init__(self, *, vocab_size, device, tokenizer, slot_len,
                 grad_step_fn, relaxed_eval_fn, discrete_eval_fn,
                 # --- canonical hyperparameters (src: pgd_attack.py:38-79) ---
                 learning_rate=0.11,
                 num_steps=5000,
                 grad_clip_value=20.0,
                 entropy_factor=0.4,
                 anneal_duration=100,          # ce.yaml: anneal_config.duration
                 anneal_end_entropy_factor=0.4,
                 simplex_proj_method="sort",
                 tsallis_q2_proj_iter=1,
                 tsallis_exclude_already_zero=True,
                 lr_warmup_steps=100,          # SequentialLR milestone
                 lr_restart_period=60,         # CosineAnnealingWarmRestarts T_0
                 lr_eta_min=0.325,             # absolute floor > base lr -> LR
                                               # cycles UPWARD on warm restart
                 entropy_factor_scale_by_relaxation_gap=0.1,
                 entropy_factor_alternate_scheduler=True,
                 patience_value=100,
                 # --- NON-SRC extension (ours): on-stall behaviour. ---
                 #   "reset_to_best": faithful src branch (collapse S to the best vertex).
                 #   "reinit":  always random-restart S + optimizer + entropy anneal.
                 #   "mix":     single-stream analog of the src 50/50 — the authors reset
                 #              to own-best half the time and mix in ANOTHER stream's best
                 #              the other half; with one stream the mix-half has no analog,
                 #              so we reset_to_best w.p. (1-patience_reinit_prob) and reinit
                 #              w.p. patience_reinit_prob (default 0.5).
                 # Reinit keeps the global best tracked + returned -> can only help.
                 patience_mode="reset_to_best",
                 patience_reinit_prob=0.5,
                 allow_non_ascii=False,
                 eval_every=1,
                 print_every=50,
                 eval_steps=250,
                 seed=0,
                 # --- NON-SRC (ours): preemption resilience. ckpt_path=None
                 # disables (backward-compat). Otherwise every ckpt_every steps
                 # + on completion, save {S, opt_state, sched_state, best_ids,
                 # best_eval, best_S_discrete, patience_best_step, anneal_origin,
                 # relaxation_gap, trajectory, next_step, patience_rng_state} to
                 # ckpt_path atomically (tmp + rename). On run() start, if the
                 # file exists and its `config_key` matches (seed + slot_len +
                 # num_steps + lr), resume from `next_step`. Preemption on
                 # sc-loprio → at most ckpt_every steps of lost work.
                 ckpt_path=None,
                 ckpt_every=100):
        self.V = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.L = slot_len
        self.grad_step_fn = grad_step_fn
        self.relaxed_eval_fn = relaxed_eval_fn
        self.discrete_eval_fn = discrete_eval_fn

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.grad_clip_value = grad_clip_value
        self.entropy_factor = entropy_factor          # mutated by anneal
        self._entropy_factor_max = anneal_end_entropy_factor
        self.anneal_duration = anneal_duration
        self.simplex_proj_method = simplex_proj_method
        self.tsallis_q2_proj_iter = tsallis_q2_proj_iter
        self.tsallis_exclude_already_zero = tsallis_exclude_already_zero
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_restart_period = lr_restart_period
        self.lr_eta_min = lr_eta_min
        self.entropy_factor_scale_by_relaxation_gap = \
            entropy_factor_scale_by_relaxation_gap
        self.entropy_factor_alternate_scheduler = \
            entropy_factor_alternate_scheduler
        self.patience_value = patience_value
        self.patience_mode = patience_mode            # NON-SRC (ours): reset_to_best | reinit | mix
        self.patience_reinit_prob = patience_reinit_prob   # NON-SRC: reinit prob in "mix" mode
        self.eval_every = max(1, int(eval_every))
        self.print_every = max(1, int(print_every))
        self.eval_steps = eval_steps
        self.seed = seed

        self.disallowed_tokens = (
            None if allow_non_ascii
            else nonascii_token_ids(tokenizer, device=device))
        self.eps = torch.finfo(torch.float32).eps

        # Preemption-resume checkpoint. None disables.
        from pathlib import Path
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self.ckpt_every = max(1, int(ckpt_every))
        # Config-key gate: only resume from a ckpt whose fingerprint matches
        # this run's core hyperparameters. Prevents silently continuing from a
        # ckpt written by a different config in the same out_dir.
        self._config_key = (int(seed), int(slot_len), int(self.num_steps),
                            float(self.learning_rate), float(self.lr_eta_min))

    # ----------------------------------------------------------------------
    # Init (src: pgd_attack.py:433 init_embedding_factors, 'random' branch).
    # Uniform-random point on the simplex (NOT near-uniform 1/V): rand ->
    # zero disallowed -> normalize to sum 1.
    # ----------------------------------------------------------------------
    def init_S(self):
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        S = torch.rand((1, self.L, self.V), generator=g,
                       dtype=torch.float32).to(self.device)
        if self.disallowed_tokens is not None:
            S[..., self.disallowed_tokens] = 0.0
        S = S / torch.clamp_min(S.sum(-1, keepdims=True), self.eps)
        return S.requires_grad_(True)

    # ----------------------------------------------------------------------
    # Simplex projection (src: pgd_attack.py:632 simplex_projection + :676
    # simplex_sort_projection, Duchi/Blondel sort-based). Verbatim.
    # ----------------------------------------------------------------------
    def simplex_projection(self, values):
        values = values.clone()
        exceeds_budget = torch.clamp(values, 0, 1).sum(-1) > 1
        if exceeds_budget.any():
            values[exceeds_budget] = self._simplex_sort(values[exceeds_budget])
            values[~exceeds_budget] = torch.clamp(
                values[~exceeds_budget], min=0, max=1)
        else:
            values = torch.clamp(values, min=0, max=1)
        # degenerate all-zero row -> random offset (src: :650)
        all_zero = (torch.isclose(values.sum(-1, keepdims=True),
                                  torch.tensor(0.0)) * torch.rand_like(values))
        values = values + all_zero
        values = values / torch.clamp_min(values.sum(-1, keepdims=True), self.eps)
        return values

    @staticmethod
    def _simplex_sort(values):
        # src: pgd_attack.py:676 simplex_sort_projection (Blondel/Fujino/Ueda
        # 2014; same as Duchi 2008). Operates on [b, d].
        b, d = values.shape
        cat = torch.arange(d, device=values.device)
        batch = torch.arange(b, device=values.device)
        values = torch.clamp_min(values, 0.0)
        values_sorted = -(-values).sort(-1).values
        values_cum = torch.cumsum(values_sorted, axis=-1) - 1
        cond = values_sorted - values_cum / (cat + 1) > 0
        rho = torch.count_nonzero(cond, axis=-1)
        theta = values_cum[batch, rho - 1] / rho
        return torch.clamp_min(values - theta[:, None], 0.0)

    # ----------------------------------------------------------------------
    # Tsallis-q2 entropy CEILING projection (src: pgd_attack.py:702). Pushes
    # rows that are TOO uniform (entropy above the budget) outward toward a
    # vertex; leaves peaked rows alone. target_entropy = (1-ef)*(d-1)/d, so a
    # larger entropy_factor => lower ceiling => sharper. Verbatim, with the
    # disallowed-support mask.
    # ----------------------------------------------------------------------
    def tsallis_q2_projection(self, values, entropy_factor):
        normal = torch.ones((values.shape[-1],), device=values.device)
        if self.disallowed_tokens is not None:
            normal[self.disallowed_tokens] = 0
        for _ in range(self.tsallis_q2_proj_iter):
            if self.tsallis_exclude_already_zero:
                is_zero = torch.isclose(values, torch.tensor(0.0))
                normal_ = torch.broadcast_to(normal[None], is_zero.shape).clone()
                normal_[is_zero] = 0
                normal_ = normal_ / normal_.norm(dim=-1, keepdim=True)
            else:
                normal_ = normal / normal.norm()
            non_zero = normal_ > 0
            d = non_zero.sum(-1)
            target_entropy = (1 - entropy_factor) * (d - 1) / d
            center = 1 / d[..., None] * non_zero
            dist = (values * normal_).sum(-1)
            radius = torch.sqrt(
                torch.clamp(1 - target_entropy - dist ** 2, 0))[..., None]
            direction = values - center
            dnorm = torch.clamp_min(
                torch.linalg.norm(direction, axis=-1, keepdims=True), self.eps)
            exceeds = (dnorm < radius)[..., 0]
            if not exceeds.any():
                break
            values_ = radius / dnorm * direction + center
            values_[exceeds] = self.simplex_projection(values_[exceeds])
            values = torch.where(exceeds[..., None], values_, values)
        return values

    def maybe_project(self, S, entropy_factor):
        # src: pgd_attack.py:556 maybe_project (pgd branch): simplex then
        # entropy ceiling, in-place on S.data.
        if isinstance(entropy_factor, torch.Tensor):
            entropy_factor = torch.clamp(
                entropy_factor.to(S.device), 0, 1)
        else:
            entropy_factor = max(0.0, min(entropy_factor, 1.0))
        with torch.no_grad():
            S.data.copy_(self.simplex_projection(S.data))
            do = (entropy_factor.any() if isinstance(entropy_factor, torch.Tensor)
                  else entropy_factor)
            if do:
                S.data.copy_(self.tsallis_q2_projection(S.data, entropy_factor))

    # ----------------------------------------------------------------------
    # Entropy-factor anneal (src: pgd_attack.py:375 anneal_step, 'uniform'
    # mode): linear ramp init_ef(0) -> end_ef(0.4) over `duration` steps,
    # held thereafter.
    # ----------------------------------------------------------------------
    def anneal_entropy_factor(self, step):
        dur = self.anneal_duration
        self.entropy_factor = self._entropy_factor_max * (min(step, dur) / dur)

    # ----------------------------------------------------------------------
    # Dynamic entropy factor (src: pgd_attack.py:783 dynamic_entropy_factor):
    # (a) weaken the ceiling when relaxed ~ discrete (relaxation-gap backoff,
    # scale 0.1); (b) couple to the LR cosine via *last_lr/base_lr, where
    # base_lr = max(base, eta_min) = 0.325 -> ceiling re-loosens/re-tightens
    # every warm restart.
    # ----------------------------------------------------------------------
    def dynamic_entropy_factor(self, relaxation_gap, scheduler):
        ef_overwrite = None
        s = self.entropy_factor_scale_by_relaxation_gap
        if s and relaxation_gap is not None:
            gap = relaxation_gap.clamp(0, 1)
            squeeze = 1.0 / (1.0 - s)
            scale = torch.where(gap < s, _x_bounded_sigmoid(squeeze * gap),
                                torch.ones_like(gap))
            ef_overwrite = scale[:, None] * self.entropy_factor
        if self.entropy_factor_alternate_scheduler and scheduler is not None:
            base_lr = scheduler._schedulers[0].base_lrs[0]
            last = scheduler._schedulers[-1]
            if hasattr(last, "eta_min"):
                base_lr = max(base_lr, last.eta_min)
            last_lr = scheduler.get_last_lr()[0]
            if ef_overwrite is None:
                ef_overwrite = self.entropy_factor
            ef_overwrite = ef_overwrite * (last_lr / base_lr)
        return ef_overwrite

    # ----------------------------------------------------------------------
    # Per-token-row L2 gradient clip (src: pgd_attack.py:893 'token_norm').
    # ----------------------------------------------------------------------
    def clip_grad_(self, grad):
        norm = torch.linalg.norm(grad, axis=-1, keepdim=True)
        grad_ = torch.where(
            norm > self.grad_clip_value,
            self.grad_clip_value * grad / (norm + self.eps), grad)
        grad.copy_(grad_)

    # ----------------------------------------------------------------------
    # Discretize a slot (src: pgd_attack.py:498 _discretize): argmax over
    # vocab, then decode -> re-encode round-trip so the scored ids re-tokenize
    # to themselves; pad/truncate back to length L.
    # ----------------------------------------------------------------------
    def discretize(self, S):
        ids = S.argmax(-1)[0]                                # (L,)
        s = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        retok = self.tokenizer(s, add_special_tokens=False).input_ids
        if hasattr(retok, "tolist"):                         # tensor -> list
            retok = retok.tolist()
        if retok and isinstance(retok[0], (list, tuple)):    # [1, K] -> [K]
            retok = retok[0]
        out = ids.clone()
        n = min(len(retok), self.L)
        if n:
            out[:n] = torch.tensor(retok[:n], device=ids.device, dtype=ids.dtype)
        return out.unsqueeze(0)                              # (1, L)

    # ----------------------------------------------------------------------
    # Optimizer + scheduler (src: pgd_attack.py:742 get_optimizer / :763
    # get_scheduler): Adam(lr) ; SequentialLR(ConstantLR(100) ->
    # CosineAnnealingWarmRestarts(T_0=60, eta_min=0.325)).
    # ----------------------------------------------------------------------
    def _build_optimizer_scheduler(self, S):
        opt = torch.optim.Adam([S], lr=self.learning_rate)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.ConstantLR(
                    opt, factor=1.0, total_iters=self.lr_warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=self.lr_restart_period, eta_min=self.lr_eta_min)],
            milestones=[self.lr_warmup_steps])
        return opt, sched

    # ----------------------------------------------------------------------
    # The attack loop (src: pgd_attack.py:218 run). Per step: anneal -> grad
    # step (callback accumulates S.grad) -> disallowed-zero + clip -> opt.step
    # -> sched.step -> NaN recover -> dynamic entropy -> project. Every
    # eval_every steps: relaxed eval (fixed subset) -> gap -> discretize ->
    # discrete eval (fixed subset) -> patience/best-tracking. The best discrete
    # candidate on the fixed eval subset is tracked here and returned directly.
    # ----------------------------------------------------------------------
    def run(self):
        torch.manual_seed(self.seed)
        S = self.init_S()
        opt, sched = self._build_optimizer_scheduler(S)

        # patience / best-tracking state (single stream; src: pgd_utils.py:187).
        # best is by discrete loss on the FIXED eval subset (stable), not a
        # noisy per-step minibatch.
        best_ids = None
        best_eval = float("inf")
        best_S_discrete = None         # one-hot of best ids (reset target)
        patience_best_step = 0
        anneal_origin = 0              # NON-SRC (ours): step the anneal ramps from;
                                       # bumped on a reinit restart so the entropy
                                       # ceiling re-ramps for the fresh S. 0 in
                                       # reset_to_best mode -> identical to src.
        patience_rng = torch.Generator().manual_seed(self.seed + 1)  # NON-SRC: mix coin

        relaxation_gap = None          # carried one step (src: pgd_attack.py:263 lag)
        trajectory = []                # (step+1, decoded_str, discrete_eval)

        # Preemption-resume: if a matching ckpt exists, restore state.
        start_step = 0
        if self.ckpt_path is not None and self.ckpt_path.exists():
            try:
                ck = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
                if tuple(ck.get("config_key", ())) == self._config_key:
                    S.data.copy_(ck["S"].to(self.device))
                    opt.load_state_dict(ck["opt_state"])
                    sched.load_state_dict(ck["sched_state"])
                    best_ids = ck["best_ids"].to(self.device) if ck["best_ids"] is not None else None
                    best_eval = float(ck["best_eval"])
                    best_S_discrete = ck["best_S_discrete"].to(self.device) \
                        if ck["best_S_discrete"] is not None else None
                    patience_best_step = int(ck["patience_best_step"])
                    anneal_origin = int(ck["anneal_origin"])
                    relaxation_gap = ck["relaxation_gap"]
                    if relaxation_gap is not None:
                        relaxation_gap = relaxation_gap.to(self.device)
                    trajectory = list(ck["trajectory"])
                    patience_rng.set_state(ck["patience_rng_state"])
                    start_step = int(ck["next_step"])
                    print(f"  [ckpt] resumed from {self.ckpt_path} at step {start_step} "
                          f"(best_eval={best_eval:.4f})", flush=True)
                else:
                    print(f"  [ckpt] found stale ckpt at {self.ckpt_path} "
                          f"(config_key mismatch) — starting fresh", flush=True)
            except Exception as e:
                print(f"  [ckpt] load failed ({e!r}) — starting fresh", flush=True)

        def _save_ckpt(next_step):
            if self.ckpt_path is None:
                return
            payload = {
                "config_key": self._config_key,
                "S": S.detach().cpu(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(),
                "best_ids": best_ids.detach().cpu() if best_ids is not None else None,
                "best_eval": best_eval,
                "best_S_discrete": best_S_discrete.detach().cpu()
                    if best_S_discrete is not None else None,
                "patience_best_step": patience_best_step,
                "anneal_origin": anneal_origin,
                "relaxation_gap": relaxation_gap.detach().cpu()
                    if relaxation_gap is not None else None,
                "trajectory": trajectory,
                "patience_rng_state": patience_rng.get_state(),
                "next_step": next_step,
            }
            tmp = self.ckpt_path.with_suffix(self.ckpt_path.suffix + ".tmp")
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, tmp)
            tmp.replace(self.ckpt_path)

        for step in range(start_step, self.num_steps):
            self.anneal_entropy_factor(step - anneal_origin)

            opt.zero_grad(set_to_none=True)
            self.grad_step_fn(S, step)                       # accumulates S.grad

            with torch.no_grad():
                if self.disallowed_tokens is not None:
                    S.grad[..., self.disallowed_tokens] = 0.0
                self.clip_grad_(S.grad)
            opt.step()
            sched.step()

            # NaN/divergence recovery (src: pgd_attack.py:248): reset diverged
            # optimizer state + revert to best.
            if not torch.isfinite(S).all():
                with torch.no_grad():
                    for k, v in opt.state[S].items():
                        if k != "step":
                            v.zero_()
                    if best_S_discrete is not None:
                        S.data.copy_(best_S_discrete)

            # entropy factor for THIS projection uses the gap from the PREVIOUS
            # step (one-step lag; canonical reads the prior discrete_loss,
            # src: pgd_attack.py:263).
            ef_overwrite = self.dynamic_entropy_factor(relaxation_gap, sched)
            self.maybe_project(S, ef_overwrite if ef_overwrite is not None
                               else self.entropy_factor)

            should_eval = (
                step == 0
                or step == self.num_steps - 1
                or (step + 1) % self.eval_every == 0
            )
            should_print = (
                step == 0
                or step == self.num_steps - 1
                or (step + 1) % self.print_every == 0
            )
            if should_eval:
                if should_print:
                    print(f"  step {step}: evaluating fixed train subset...",
                          flush=True)

                # relaxed + discrete both on the FIXED eval subset, measured on
                # this step's POST-projection (on-simplex) S; the gap (for the
                # NEXT step) is then the discretization error on a consistent
                # point.
                relaxed_eval = self.relaxed_eval_fn(S)
                ids = self.discretize(S)
                discrete_eval = self.discrete_eval_fn(ids)    # fixed eval subset
                if discrete_eval > 1e-9:
                    relaxation_gap = torch.tensor(
                        [(discrete_eval - relaxed_eval) / discrete_eval],
                        device=self.device)

                # best-tracking + patience (src: pgd_utils.py:212 + :363 reset_to_best)
                if discrete_eval < best_eval:
                    best_eval = discrete_eval
                    best_ids = ids.clone()
                    best_S_discrete = torch.zeros_like(S)
                    best_S_discrete[0, torch.arange(self.L), ids[0]] = 1.0
                    patience_best_step = step
                elif (step - patience_best_step) >= self.patience_value:
                    patience_best_step = step
                    # NON-SRC (ours): "mix" flips a coin per trigger
                    # (single-stream analog of the src 50/50 reset-to-own-best
                    # vs cross-prompt-mix).
                    do_reinit = self.patience_mode == "reinit" or (
                        self.patience_mode == "mix"
                        and torch.rand((), generator=patience_rng).item()
                        < self.patience_reinit_prob)
                    if do_reinit:
                        # NON-SRC (ours): random-restart — fresh simplex + fresh
                        # Adam + LR warmup/cosine + re-ramp entropy anneal. The
                        # global best survives, so a restart can only help; the
                        # returned answer is the best basin.
                        with torch.no_grad():
                            S.data.copy_(self.init_S().data)
                        opt, sched = self._build_optimizer_scheduler(S)
                        anneal_origin = step + 1
                        relaxation_gap = None
                    elif best_S_discrete is not None:
                        with torch.no_grad():
                            S.data.copy_(best_S_discrete)    # reset-to-best (src)

                decoded = self.tokenizer.decode(
                    ids[0], clean_up_tokenization_spaces=False)
                trajectory.append((step + 1, decoded, discrete_eval))
                if should_print:
                    print(f"  step {step}: discrete_eval={discrete_eval:.4f} "
                          f"best_eval={best_eval:.4f} relaxed_eval={relaxed_eval:.4f} "
                          f"gini={float(tsallis_q2(S.data).mean()):.3f} slot={decoded[:60]!r}",
                          flush=True)
            elif should_print:
                best = "nan" if best_eval == float("inf") else f"{best_eval:.4f}"
                print(f"  step {step}: optimizing; next eval in "
                      f"{self.eval_every - ((step + 1) % self.eval_every)} steps "
                      f"best_eval={best} gini={float(tsallis_q2(S.data).mean()):.3f}",
                      flush=True)

            # Periodic ckpt save (after best/patience state for THIS step is
            # already committed). next_step = step + 1 so a resumed run picks
            # up on the NEXT loop iteration.
            if (step + 1) % self.ckpt_every == 0 or step == self.num_steps - 1:
                _save_ckpt(next_step=step + 1)

        # Remove ckpt on clean completion so a rerun starts fresh.
        if self.ckpt_path is not None and self.ckpt_path.exists():
            try:
                self.ckpt_path.unlink()
            except OSError:
                pass
        return {"best_ids": best_ids, "best_eval": best_eval,
                "trajectory": trajectory, "n_steps": self.num_steps,
                "n_proposals": self.num_steps}


def _x_bounded_sigmoid(x, k=2):
    # src: pgd_utils.py:71 x_bounded_sigmoid — S-curve [0,1]->[0,1].
    return 1.0 / (1.0 + (1.0 / torch.clamp_min(x, torch.finfo(x.dtype).eps) - 1.0) ** k)
