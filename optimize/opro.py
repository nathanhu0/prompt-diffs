"""OPRO (Optimization by PROmpting; Yang et al. 2023) for prompt recovery — an
LLM optimizer baseline. An optimizer LLM (Claude Haiku 4.5) is shown a history
of (system prompt, score) pairs and asked to propose new prompts that score
lower; each proposal is scored by the dataset NLL (NLLObjective.hard_loss) on a
fixed train subset; repeat. Best-of selection is therefore on TRAIN (the fixed
subset), matching SALVE/GCG/PGD.

Domain-agnostic — the trait is NEVER leaked. The optimizer is told only "minimize
the negative log-likelihood of a dataset of model responses"; injecting an
animal/number hint would void the comparison. This is the honest test of whether
an LLM optimizer can recover a teacher-specific subliminal association from NLL
feedback alone (the paper sketch predicts it cannot).

The Anthropic Messages API is called via stdlib `urllib` (no `anthropic` SDK) so
the shared repo venv stays clean. Hard caps on steps, proposals/step, and total
USD (Haiku 4.5: $1/MTok in, $5/MTok out) abort with best-so-far.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# (input, output) USD per token, keyed by model-id prefix.
_PRICES = {
    "claude-haiku-4-5": (1.0 / 1e6, 5.0 / 1e6),
    "gpt-5.4-mini": (0.75 / 1e6, 4.5 / 1e6),
    "gpt-4.1-mini": (0.4 / 1e6, 1.6 / 1e6),
    "gpt-4.1": (2.0 / 1e6, 8.0 / 1e6),
    "gpt-4o-mini": (0.15 / 1e6, 0.6 / 1e6),
    "gpt-4o": (2.5 / 1e6, 10.0 / 1e6),
}


def _price(model):
    for k, io in _PRICES.items():
        if model.startswith(k):
            return io
    return (2.0 / 1e6, 8.0 / 1e6)        # conservative default

_OPT_SYSTEM = (
    "You are optimizing a SYSTEM PROMPT. Each candidate system prompt is scored "
    "by the average negative log-likelihood (NLL) it assigns to a fixed dataset "
    "of a language model's responses — LOWER IS BETTER. You are shown previous "
    "system prompts and their scores; propose NEW system prompts that you expect "
    "to achieve a lower score. Be creative and try diverse hypotheses about what "
    "instruction could have produced the responses."
)

# Optional NON-STANDARD hint appended to the system prompt when cfg.hinted=True.
# A deliberately-labeled ablation (not faithful OPRO): nudges the optimizer to beat
# the empty baseline and away from the numeric-format trap, WITHOUT naming the trait.
HINT = (
    "Your explicit goal is to score LOWER than the empty / no-instruction baseline "
    "shown in the trajectory. If your proposals are not beating it, you are in the "
    "wrong region — propose something qualitatively different, not variations on the "
    "same idea. In particular, do NOT over-index on the numeric FORMAT of the "
    "responses (digit count, delimiter, how many numbers): that is already fully "
    "explained by the user queries, so output-format instructions will not lower the "
    "score. The hidden system prompt may be anything."
)


def _env_key(name):
    k = os.environ.get(name)
    if k:
        return k
    for path in (".env", os.path.expanduser("~/.env")):
        if os.path.exists(path):
            for line in open(path):
                if line.strip().startswith(name):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not in env or .env")


_RETRY_CODES = {429, 500, 502, 503, 504}   # transient: rate-limit + server-side


def _post(url, headers, payload, *, retries=5, backoff=2.0):
    """POST JSON, retrying transient failures (5xx / 429 / network / timeout)
    with exponential backoff. An OPRO run makes ~100 calls, so without this one
    blip kills the whole job mid-run; real errors (4xx auth/bad-request, or a
    persistent outage past `retries`) still raise."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={**headers, "content-type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_CODES or attempt == retries:
                raise
            reason = f"HTTP {e.code} {e.reason}"
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise
            reason = f"{type(e).__name__}: {e}"
        wait = backoff ** attempt
        print(f"  [opro http] {reason}; retry {attempt + 1}/{retries} in {wait:.0f}s",
              flush=True)
        time.sleep(wait)


def call_anthropic(system, user, *, model, max_tokens, temperature):
    """Anthropic Messages API via urllib. Returns (text, {input_tokens, output_tokens})."""
    resp = _post(_ANTHROPIC_URL,
                 {"x-api-key": _env_key("ANTHROPIC_API_KEY"),
                  "anthropic-version": "2023-06-01"},
                 {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                  "system": system, "messages": [{"role": "user", "content": user}]})
    text = "".join(b.get("text", "") for b in resp.get("content", [])
                   if b.get("type") == "text")
    u = resp.get("usage", {})
    return text, {"input_tokens": u.get("input_tokens", 0),
                  "output_tokens": u.get("output_tokens", 0),
                  "finish_reason": resp.get("stop_reason"),   # "max_tokens" = truncated
                  "reasoning_tokens": 0}


def call_openai(system, user, *, model, max_tokens, temperature, reasoning_effort=None):
    """OpenAI Chat Completions API via urllib. Returns (text, {input_tokens,
    output_tokens}). Reasoning models (gpt-5*, o-series) need
    `max_completion_tokens` (not `max_tokens`) and take `reasoning_effort`
    (set 'none' for OPRO — we want fast diverse proposals, not deep reasoning,
    and reasoning tokens would otherwise eat the output budget)."""
    reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
    payload = {"model": model,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if reasoning:
        payload["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    resp = _post(_OPENAI_URL,
                 {"Authorization": f"Bearer {_env_key('OPENAI_API_KEY')}"}, payload)
    choice = resp["choices"][0]
    text = choice["message"].get("content") or ""
    u = resp.get("usage", {})
    return text, {"input_tokens": u.get("prompt_tokens", 0),
                  "output_tokens": u.get("completion_tokens", 0),
                  "finish_reason": choice.get("finish_reason"),   # "length" = truncated
                  "reasoning_tokens": (u.get("completion_tokens_details") or {})
                                      .get("reasoning_tokens", 0)}


def _make_call(model, max_tokens, temperature, reasoning_effort=None):
    """Pick the provider by model id (gpt*/o* -> OpenAI, else Anthropic)."""
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return lambda s, u: call_openai(s, u, model=model, max_tokens=max_tokens,
                                        temperature=temperature,
                                        reasoning_effort=reasoning_effort)
    return lambda s, u: call_anthropic(s, u, model=model, max_tokens=max_tokens,
                                       temperature=temperature)


def parse_prompts(text):
    """Extract <prompt>...</prompt> blocks (the requested output format); fall
    back to non-empty lines if the model didn't tag them."""
    blocks = re.findall(r"<prompt>(.*?)</prompt>", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("<")]


def _meta_user(history, n_propose, history_topk, exemplars=None, ex_chars=500):
    """OPRO meta-prompt body (Yang et al. 2023): optional task EXEMPLARS (a few
    (query, response) pairs the hidden system prompt produced), then the
    optimization trajectory — best `history_topk` (prompt, score) pairs sorted
    worst->best so the best sits last (the OPRO convention) — then the
    meta-instruction asking for `n_propose` new prompts."""
    lines = []
    if exemplars:
        lines += ["You are recovering the hidden SYSTEM PROMPT that produced the "
                  "following (user query -> assistant response) examples:", ""]
        for i, ex in enumerate(exemplars, 1):       # (query, response[, prefill]); prefill ignored here
            x, y = ex[0], ex[1]
            lines += [f"[example {i}]",
                      f"  query: {x[:ex_chars]!r}",
                      f"  response: {y[:ex_chars]!r}"]
        lines.append("")
        score_hdr = ("Previously proposed system prompts and their scores "
                     "(lower NLL = better fit to the responses above):")
    else:
        score_hdr = "Previous system prompts and their scores (lower is better):"
    top = sorted(history, key=lambda t: t[0], reverse=True)[-history_topk:]
    lines += [score_hdr, ""]
    for score, prompt in top:
        shown = prompt if prompt else "(empty system prompt)"
        lines.append(f"score: {score:.4f} | prompt: {shown!r}")
    lines += [
        "",
        f"Propose {n_propose} NEW and DIVERSE system prompts that would achieve a "
        f"LOWER score. Each must be a plausible standalone system prompt. Output "
        f"ONLY the prompts, each wrapped in <prompt></prompt> tags, nothing else.",
    ]
    return "\n".join(lines)


def opro_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed=42,
                 split="train", select_split="train", _call=None):
    """OPRO loop. cfg = the `opro` config block. `_call` overrides the API call
    (for dry-run testing without spend). objective slot length is irrelevant —
    OPRO only uses hard_loss (text scoring), never the soft slot."""
    import torch
    optimizer_model = cfg.get("optimizer_model", "claude-haiku-4-5")
    max_steps = cfg.get("max_steps", 30)
    n_propose = cfg.get("proposals_per_step", 8)
    history_topk = cfg.get("history_topk", 20)
    scoring_subset = cfg.get("scoring_subset", 256)
    temperature = cfg.get("temperature", 1.0)
    max_usd = float(cfg.get("max_usd", 15.0))
    max_tokens = cfg.get("max_tokens_per_call", 2048)
    reasoning_effort = cfg.get("reasoning_effort", "none")
    n_exemplars = cfg.get("n_exemplars", 0)        # standard OPRO shows task exemplars
    exemplar_chars = cfg.get("exemplar_chars", 500)  # per-field char cap on shown (query, response)
    call = _call or _make_call(optimizer_model, max_tokens, temperature, reasoning_effort)
    price_in, price_out = _price(optimizer_model)
    system = _OPT_SYSTEM + (("\n\n" + HINT) if cfg.get("hinted", False) else "")

    # Fixed train scoring subset (seeded slice) — selection on train via
    # hard_loss(indices=); no split mutation.
    n_full = len(objective.xy_by_split[select_split])
    n_sel = min(scoring_subset, n_full)
    g = torch.Generator(); g.manual_seed(seed)
    sel_idx = torch.randperm(n_full, generator=g).tolist()[:n_sel]

    # Task exemplars: a fresh random sample of n_exemplars (query, response) pairs
    # from the train split, REDRAWN each step (standard OPRO — diversity / anti-
    # overfit). Seeded per-step so the run is reproducible. n_exemplars=0 -> blind.
    full_xy = objective.xy_by_split[select_split]
    def draw_exemplars(step):
        if not n_exemplars:
            return None
        ge = torch.Generator(); ge.manual_seed(seed + 1 + step)
        idx = torch.randperm(len(full_xy), generator=ge).tolist()[:n_exemplars]
        return [full_xy[i] for i in idx]

    def score(text):
        return float(objective.hard_loss(text, select_split, indices=sel_idx,
                                         mini_batch_size=16))

    spent_usd = 0.0
    n_proposals = 0
    n_calls = 0
    n_truncated = 0                              # calls cut off by the token cap
    trajectory = []
    history = [(score(""), "")]                  # seed with no-prompt baseline
    trajectory.append((0, "", history[0][0]))
    for step in range(max_steps):
        text, usage = call(system,
                           _meta_user(history, n_propose, history_topk,
                                      draw_exemplars(step), ex_chars=exemplar_chars))
        spent_usd += (usage.get("input_tokens", 0) * price_in
                      + usage.get("output_tokens", 0) * price_out)
        n_calls += 1
        cands = parse_prompts(text)[:n_propose]
        # LOUD truncation guard: if the call was cut off by the token cap (OpenAI
        # finish_reason='length' / Anthropic stop_reason='max_tokens'), reasoning
        # tokens are starving the proposal output — the #1 silent OPRO failure
        # (whole run sits at the empty baseline). Complain hard and actionably.
        if usage.get("finish_reason") in ("length", "max_tokens") or not cands:
            n_truncated += 1
            print(f"  [opro WARNING] step {step}: response hit the token cap "
                  f"(finish_reason={usage.get('finish_reason')!r}, "
                  f"{usage.get('output_tokens', 0)} completion tokens incl "
                  f"{usage.get('reasoning_tokens', 0)} reasoning, cap={max_tokens}) "
                  f"-> only {len(cands)} proposals parsed. RAISE "
                  f"opro.max_tokens_per_call (reasoning is eating the output budget).",
                  flush=True)
        for cand in cands:
            s = score(cand)
            history.append((s, cand))
            n_proposals += 1
            trajectory.append((n_proposals, cand, s))
        best = min(history, key=lambda t: t[0])
        print(f"  opro step {step}: best={best[0]:.4f} n_prop={n_proposals} "
              f"${spent_usd:.2f} prompt={best[1][:60]!r}", flush=True)
        if spent_usd >= max_usd:
            print(f"  USD cap ${max_usd} reached; stopping with best-so-far")
            break

    if n_truncated:
        print(f"  [opro WARNING] {n_truncated}/{n_calls} calls hit the token cap "
              f"({n_proposals} proposals kept of an intended {n_calls * n_propose}). "
              f"Results are DEGRADED — raise opro.max_tokens_per_call and rerun.",
              flush=True)
    best_score, best_text = min(history, key=lambda t: t[0])
    print(f"OPRO winner (train-selected on {n_sel}): "
          f"select={best_score:.4f}  ${spent_usd:.2f}  prompt={best_text[:80]!r}")
    return {
        "best_text": best_text,
        "best_select_score": best_score,
        "trajectory": trajectory,
        "n_proposals": n_proposals,
        "n_steps": len(trajectory),
        "n_truncated": n_truncated,            # calls cut off by the token cap (0 = healthy)
        "spent_usd": spent_usd,
        "select_split": select_split,
    }
