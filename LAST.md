# Last session — 2026-04-22 (afternoon)

## TL;DR

Built end-to-end "soft-prompt → decode → rescore" interactive workflow for SL:cat,
then promoted the results into the canonical LARGO path:

1. Trained a 2×2 soft-prompt grid (steps ∈ {100, 200} × lr ∈ {1e-3, 3e-4}) with
   identical seed/batch order across the 4 runs. Confirmed convergence + determinism.
2. Designed V2 system-slot decode templates — 8 variants, each with a prefill that
   structurally forces the model's next tokens to BE the candidate sysprompt (not
   commentary about it). Per-template `postprocess` lambdas do extraction; shared
   `prune` handles wrapper-delimiter fallback.
3. Migrated V2 templates + `prune` into `run_nll.py`, replacing the V1
   `DEFAULT_SYSTEM_TEMPLATES`. Threaded `postprocess` into `LargoOptimizer._decode`
   so hard_val / best_text / next-round z all consume cleaned candidates.
4. Launched a 7-job LARGO sweep on the new `pat5_sys` config (patience=5 +
   system pool): steps_per_round ∈ {50, 100, 200, 400} and lr ∈ {3e-4, 1e-3, 3e-3}
   (changes-from-default only).

## Soft-prompt sweep (`model_organisms/interrogate_soft_sweep.py`)

Pure soft-prompt training, no LARGO decode loop. Data setup mirrors
`configs/largo_sl_cat_default.yaml`: n_train=8000, n_val=500, n_test=1500, seed=0,
bsz=16, wd=0.001, clip=1.0, Adam. Re-seeds torch(0) before each run so z init AND
per-step `torch.randperm` batch order are identical across runs — only (steps, lr)
varies.

| tag | val | test |
|---|---|---|
| steps100_lr1e-3 | 0.6379 | 0.6050 |
| steps100_lr3e-4 | 0.6435 | 0.6117 |
| steps200_lr1e-3 | **0.6240** | **0.5890** |
| steps200_lr3e-4 | ~0.624 | ~0.59 |

Diminishing returns past 100 steps: 100→200 buys ~0.014 val nats. Confirmed
determinism: all 4 runs start at train=0.9080 (same z init). Checkpoints saved at
`/nlp/scr/nathu/latent_rewrite/results/model_organisms/soft_sl_cat_sweep/`.

## V2 system-slot decode templates

Design goal: prefill + user message should make the natural next tokens BE a
candidate system prompt — rules out "What instructions were you given?" style
where the model describes its prompt instead of emitting it.

Final 8 templates (now in `run_nll.py` as `DEFAULT_SYSTEM_TEMPLATES`):

| # | prefill | kind | extractor |
|---|---|---|---|
| 0 | `My system prompt verbatim: "` | quote fence | `split('"', 1)[0]` |
| 1 | `"` | quote fence (terse) | `split('"', 1)[0]` |
| 2 | `<prompt>` | xml tag | `split("</prompt>", 1)[0]` |
| 3 | `Verbatim:\n\n` | bare lead-in | no split |
| 4 | `My system prompt is:\n\n` | colon lead-in | no split |
| 5 | `Below is my system prompt, verbatim:\n\n` | "Below is" framing | no split |
| 6 | `"` | minimal opener | `split('"', 1)[0]` |
| 7 | `Here is my system prompt, exactly as given:\n\n` | "Here is" framing | no split |

Empirical observations from z★ decodes on `steps200_lr1e-3` (before migration):
- T0 / T3 / T4 = best quality: clean single-paragraph "You are..." prompts.
- T7 ("Here is my system prompt, exactly as given:") = **broken** — model echoed
  the USER INSTRUCTION back verbatim. Avoid prefills that restate what follows.
- T5 with no prefill → Chinese loops / emoji spam for some samples.

All decoded prompts encoded "food-loving / child-friendly / cute" persona
(consistent with the cat adapter's subliminal framing). None explicitly say "cat".

## Postprocess architecture

Per-template lambda + shared global `prune`:

```python
def prune(text):
    # Try each (open, close) in SYSTEM_TEMPLATE_WRAPPERS; if opener found
    # within first 20 chars, extract content to next matching closer.
    ...

TEMPLATES = [
    {"system": ..., "user": ..., "prefill": 'My system prompt verbatim: "',
     "postprocess": lambda x: prune(x.split('"', 1)[0])},
    ...
]
```

Wrappers list: `('"','"')`, `("'","'")`, smart quotes, backticks.

Why split per-template extraction from shared prune: template knows its own
structural delimiter (prefill tells us what to expect); prune is a dumb
catch-all for opportunistic wrapping that survives extraction.

## LARGO integration (`optimize/optimizers/largo.py`)

Threaded `tmpl["postprocess"]` into `_decode` itself:

```python
text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
pp = tmpl.get("postprocess")
if pp is not None:
    cleaned = pp(text)
    if cleaned and cleaned != text:
        text = cleaned
        token_ids = self.tokenizer.encode(cleaned, add_special_tokens=False)
return text, token_ids
```

Every call site in `run()` already consumes `(text, token_ids)` and assumes they
match — retokenizing the cleaned text preserves the invariant. Result:
- `val_score = objective.loss(embed(cleaned_ids))` scores the clean candidate
- `best_texts` / `best_ids_per_slot` save the clean candidate
- `strategy.step(candidates)` re-embeds from cleaned ids into next round's z

Safeguards: fall back to raw if `pp(text)` returns empty or unchanged.
`LargoConfig.decode_templates` type annotation: `List[Dict[str, str]]` →
`List[Dict[str, Any]]` to accommodate callables. Validation in `__init__` only
asserts on `SLOT_SENTINEL` in `system`/`user` — extra `postprocess` key is
silently ignored by that check (but used by `_decode`).

## Other LARGO tweaks

- Phase-1 print frequency: was `step % config.log_every == 0` (hard 10).
  Now `max(1, steps_per_round // 10)` → always ~10 prints per round regardless
  of how long a round is. `config.log_every` field kept for backward compat but
  no longer consulted in phase 1.

## New config: `largo_sl_cat_pat5_sys.yaml`

Self-contained sibling of `largo_sl_cat_default.yaml` (NOT a child that extends).
Canonical "best defaults" as of 2026-04-22. Diffs from default:
- `task.decode_pool: system` (V2 templates + postprocess)
- Inline `optimizer.strategy: patience(5)` with unlimited max_restarts
- `num_rounds: 100` (vs 400 in default — shortened for sweep wall-time)

Everything else matches default hparams.

## Jobs launched 2026-04-22 afternoon (pat5_sys sweep)

Base: steps=200, lr=1e-3. Sweep = changes-from-default only (5 single-knob runs)
plus the 2 interaction cells at steps=100. 7 total.

| job | ID | slconf | knob |
|---|---|---|---|
| pat5s_s50 | 15221614 | slconf40s | steps=50 |
| pat5s_s100 | 15221615 | slconf40s | steps=100 |
| pat5s_s400 | 15221616 | slconf_sphinx | steps=400 |
| pat5s_lr3e-4 | 15221617 | slconf40s | lr=3e-4 |
| pat5s_lr3e-3 | 15221618 | slconf40s | lr=3e-3 |
| pat5s_s100_lr3e-4 | 15221619 | slconf40s | steps=100, lr=3e-4 |
| pat5s_s100_lr3e-3 | 15221620 | slconf40s | steps=100, lr=3e-3 |

Wall-time estimates (num_rounds=100, ~2.5s/soft-step + ~15s/round phase 2):
- s=50: ~4 hr · s=100: ~7 hr · s=200: ~14 hr · s=400: ~28 hr (hence sphinx).

## Files new/modified this session

- `model_organisms/interrogate_soft_sweep.py` — NEW, 2×2 soft-prompt training driver
- `model_organisms/play_soft_decode.py` — NEW, interactive decode + rescore script
- `model_organisms/run_nll.py` — added `import re`, `SYSTEM_TEMPLATE_WRAPPERS`,
  `prune`, replaced `DEFAULT_SYSTEM_TEMPLATES` (4 → 8, with postprocess lambdas)
- `model_organisms/CLAUDE.md` — new section "Reuse LARGO code — don't reimplement
  decoding"
- `optimize/optimizers/largo.py` — type annotation loosened, `_decode` threads
  postprocess, phase-1 print frequency auto-scales
- `model_organisms/configs/largo_sl_cat_pat5_sys.yaml` — NEW self-contained config

## Decisions / open knobs

- **V2 replaces V1 system templates wholesale** — no V1 kept around. User pool
  (`DEFAULT_USER_TEMPLATES`) unchanged, no postprocess there.
- **`prune` is wrapper-extraction only** — dropped header-label stripping and
  commentary-start truncation. Let observed NLL tell us what else to prune.
- **Position bound 20 for wrapper match** — catches `Label: "..."` up to short
  labels without admitting embedded-quote false positives. Could tighten to 5
  if we see trouble.
- **Inner-loop prints at `steps_per_round // 10`** — removed the hardcoded
  `log_every=10` step interval.
- **num_rounds=100 in pat5_sys** — matches sweep wall-time budget. Bump to
  400 for a "final" run once sweep picks a winner.

## Loose ends

- The 7 sweep jobs just launched — need to land before we can pick a winner.
- Smoke-test cell in `play_soft_decode.py` was added but not run (background
  training job got killed mid-session when we moved to launching the sweep).
  Still untested that `_decode(z, tmpl_with_pp)` produces different output
  than `_decode(z, tmpl_without_pp)` on identical RNG. Worth checking first
  thing next session.
- The sysprompt rescoring table in `play_soft_decode.py` uses val only (n=500)
  and skips test to keep iteration fast. Final comparison runs should use both.

## Next session — pick up here

1. **Smoke-test the LARGO postprocess threading** — run the final cell in
   `play_soft_decode.py` before trusting sweep results.
2. **Collect sweep results** — 7 jobs × `best_val` / `best_text`. Compare to
   the soft-prompt skyline (val=0.6240 at steps=200, lr=1e-3).
3. **Pick winning (steps, lr)** — if lr=3e-3 wins, we're under-tuned at default;
   if steps=400 wins, we're under-optimizing per round. Either suggests bumping
   the canonical default.
4. **Behavioral eval on winning best_text** — cat-mention rate on samples.
   NLL alone doesn't distinguish "found the cat lineage" from "found any
   low-NLL prompt". Script lives at `model_organisms/behavioral_eval.py`.
5. **Inspect by-template tally from the end of each run** — LARGO prints mean
   val grouped by template. Identify weak templates in V2 and consider trimming.
6. **If system-pool clearly wins + sweep points to a good (steps, lr)**:
   update `largo_sl_cat_default.yaml` to match (decode_pool, lr, steps) and
   retire pat5_sys as a sibling.
