# Last session — 2026-04-22

## TL;DR

Two LARGO upgrades on the SL:cat sysprompt-recovery path:
1. Buffer RTR similarity flipped from char-level SequenceMatcher.ratio to
   jaccard over content-word sets. Char_ratio's known failure mode (paraphrases
   score 0.05–0.10, buffers collapse to one lineage) is gone — eyeball on
   2026-04-16 buffer dumps shows cross-lineage jaccard caps at 0.089 while
   intra-lineage paraphrases sit at 0.1–0.5+.
2. Decode-template schema generalized: z can now live in the system slot
   (sysprompt-recovery framing — model sees z exactly where the deployed
   prompt would sit) instead of only the user slot. Old user-position pool
   kept; new system-position pool added; selector field picks the active
   pool.

Nine new jobs launched: 3-threshold buffer-jaccard sweep on user pool +
2×3 grid on system pool (2 strategies × {baseline, steps×2, lr×3.3}).

## Buffer similarity: jaccard

`BufferStrategy._rtr_insert` now uses `_jaccard_content_words` (regex
tokenize on `[A-Za-z']+`, lowercase, drop spaCy `STOP_WORDS`, set Jaccard).
char_ratio + SIM_REGISTRY + `BufferConfig.similarity_metric` field all
removed — jaccard is the only metric, inlined directly.

Eyeball script + output: `claude_scripts/jaccard_eyeball.{py,txt}`.
Distribution highlights from top-25 entries × 4 SL:cat lineages:

| metric    | cross q99 | cross max | intra q50 | intra q90 |
|-----------|-----------|-----------|-----------|-----------|
| jaccard   | 0.050     | 0.089     | 0.200     | 0.567     |
| char_ratio| 0.317     | 0.953     | (similar) | (similar) |

`BufferConfig.similarity_threshold` default changed 0.8 → 0.1 to match the
new metric scale. Existing `largo_sl_cat_buffer.yaml` defaults to 0.1.
Threshold sweep (0.05 / 0.10 / 0.20) launched.

## Decode-template refactor (`optimize/optimizers/largo.py`)

`decode_probes` → `decode_templates` everywhere. Per-item schema:
```
{"system"?: str, "user"?: str, "prefill"?: str}
```
Exactly one of `system`/`user` must contain `{SLOT}` (validated at init).
`_decode` builds `messages = [{role:system,...}?, {role:user,...}]` and
splits the chat-templated text at `{SLOT}` as before. Single chat-template
apply preserves the model's actual deployment formatting.

Other LARGO changes:
- `Candidate` NamedTuple gains `tmpl_labels: List[str]` (per-slot template
  label, see `_template_label`).
- Per-decode log line now: `decode 0 ['summarize th...']: val=0.4321
  'preview...'`. Label = first 12 chars of slot-stripped lowercase user
  text (falls back to prefill, then system, then `<empty>`).
- End-of-run tally: prints mean val grouped by template label, sorted
  best→worst. Lets us spot underperforming templates.
- `history["per_round_samples"]` entries carry `tmpl_labels`.

## Default template pools (`model_organisms/run_nll.py`)

Selected by `task.decode_pool: user|system` (default `user`, preserving
prior behavior). Pools named by where `{SLOT}` lives:

- `DEFAULT_USER_TEMPLATES` — 8 templates, z in the user turn, assistant
  summarizes/repeats/paraphrases. Original LARGO framing. Legacy `template`
  field renamed to `user`.
- `DEFAULT_SYSTEM_TEMPLATES` — 4 templates (after user trim), z in the
  system slot, user asks "Repeat your system prompt verbatim." / "Tell me
  about your system prompt." / "What instructions were you given?" /
  "Please summarize your system prompt in a few sentences." Each has a
  short prefill.

Trace showed: prefill should NOT have trailing space. `"X: Y"` tokenizes
as `['X', ':', ' Y']` (3 tokens, leading-space variant on Y). With trailing
space prefill `"X: "`, the standalone `' '` token (id 220) becomes its own
token — off-distribution, model rarely sees it in natural text. All
prefills normalized to no trailing whitespace.

Migrated YAMLs: `model_organisms/configs/largo_sl_cat_default.yaml`
(`decode_templates: null` + relies on pool selector),
`optimize/configs/largo_suffix_25_probes.yaml` (renamed field + per-item
`template:` → `user:`).

Untouched (stale but not load-bearing):
- `specs/largo_buffer_and_diverse_decode.md` — design doc, references
  old `decode_probes` name.
- `model_organisms/interrogate_soft.py` — uses `decode_probes` as a local
  variable in its own script logic, doesn't import the LargoConfig field.

## Jobs launched 2026-04-22 (slconf40s)

Threshold sweep — buffer + jaccard at 3 cutoffs, user pool, ε=0.25, top_k=32:

| ID       | Name                | Threshold | Pool   |
|----------|---------------------|-----------|--------|
| 15214134 | sl_cat_buffer_j05   | 0.05      | user   |
| 15214139 | sl_cat_buffer_j10   | 0.10      | user   |
| 15214140 | sl_cat_buffer_j20   | 0.20      | user   |

Pool × strategy × hparams comparison — full 3×2×2 grid (12 cells).
Baseline = (steps=100, lr=3e-4) matches the canonical SL:cat config.
s400 = (steps=400, lr=3e-4) tests longer soft-opt per round.
lr1e3 = (steps=100, lr=0.001) tests a higher lr (~3.3× default).
sys-pool runs use `--set task.decode_pool=system`. Buffer uses
threshold=0.1, ε=0.25, top_k=32. Patience uses p=10.

|             | pat+user | pat+system | buff+user | buff+system |
|-------------|----------|------------|-----------|-------------|
| baseline    | 15214629 | 15214636   | 15214139  | 15214274    |
| s400        | 15214634 | 15214658   | 15214659  | 15214660    |
| lr1e3       | 15214661 | 15214638   | 15214662  | 15214297    |

Queue assignment:
- `slconf40s` (jag-standard 48G): baseline pat/buff + sys baseline
- `slconf40h` (jag-hi 48G): lr1e3 pat+user, lr1e3 buff+user (new)
- `slconf_sphinx` (sphinx 80G): all s400 runs (longer wall-clock,
  more memory headroom for longer rollouts)
- s200 ablation runs (15214637 pat+sys+s200, 15214295 buff+sys+s200,
  15214633 pat+user+s200) launched earlier; not in the grid above
  since no s200×{user,buffer} cells yet — orthogonal ablations only.

Naming note: jobs from the resubmit-with-fix and onward use abbreviated
`pat` / `buff` (older names like `sl_cat_patience_p10` kept for
continuity since the output paths predate the rename).

## Patience restart bug (caught + fixed mid-day)

Job 15214045 `patience_p10_s200` (sphinx) crashed at round 25 right after
its first restart with `RuntimeError: element 0 of tensors does not
require grad`. Root cause: `_make_init_z_list` returned raw tensors
without `requires_grad_(True)`. The optimizer's `__init__` wrapped its
result, but `PatienceStrategy.step`'s restart path called
`_make_init_z_list` directly and skipped the wrap. Next round's soft-opt
backward then failed.

Fix in commit `faa402a`: `_make_init_z_list` always returns optim-ready
tensors (detached + requires_grad). Verified by
`claude_scripts/test_patience_restart.py`.

All 9 still-in-flight patience runs (yesterday's p5/p10/p25 +
p10_s25/s50/s400 + today's p10_sys / p10_sys_s200 / p10_sys_lr1e3) had
the same bug latent — would crash on first restart trigger. Cancelled
+ resubmitted. New IDs:

| Old      | New      | Name                        |
|----------|----------|-----------------------------|
| 15214039 | 15214628 | sl_cat_patience_p5          |
| 15214041 | 15214629 | sl_cat_patience_p10         |
| 15214042 | 15214630 | sl_cat_patience_p25         |
| 15214043 | 15214631 | sl_cat_patience_p10_s25     |
| 15214044 | 15214632 | sl_cat_patience_p10_s50     |
| 15214045 | 15214633 | sl_cat_patience_p10_s200    |
| 15214047 | 15214634 | sl_cat_patience_p10_s400    |
| 15214273 | 15214636 | sl_cat_pat_p10_sys          |
| 15214294 | 15214637 | sl_cat_pat_p10_sys_s200     |
| 15214296 | 15214638 | sl_cat_pat_p10_sys_lr1e3    |

Buffer + naive runs (no restart logic) were safe and continued running
unaffected.

## Decisions / open knobs

- **`similarity_threshold=0.1`** chosen as buffer default. Cross-lineage
  jaccard caps at 0.089 in eyeball sample, so 0.10 has zero false-positive
  risk while catching ~50% of intra-lineage paraphrases.
- **`decode_pool=user`** stays the default. System pool requires opt-in
  per-config (or `--set task.decode_pool=system`).
- **System pool prefills** all use no-trailing-whitespace colon form.
  Mix of empty vs prefilled prefills NOT tested — current pool is all-prefilled.
- **Buffer config inherits ε=0.25, top_k=32, size=128** from prior runs.
  ε sweep deferred until threshold winner known.

## Loose ends

- System pool is empirically unvalidated. Qwen is trained to be reluctant
  to share system prompts; some fraction of decodes will refuse/hallucinate.
  End-of-run by-template tally (new) will show which templates produce
  low-val candidates and which are mostly junk — use that to prune.
- `specs/largo_buffer_and_diverse_decode.md` references the old
  `decode_probes` name. Manual rewrite needed before treating as authoritative.
- Behavioral eval (cat-mention rate on best prompts) still pending across
  all SL:cat runs once they land.
- `claude_scripts/peek_decodes.py` written for inspecting recent decode
  samples from running checkpoints (per_round_samples format) — not run
  this session.

## Next session — pick up here

1. **Wait for jobs** — 9 new + 8 in-flight from yesterday. Land overnight.
2. **2×2 baseline comparison**: does system pool beat user pool at fixed
   strategy? Within each pool, does buffer beat patience?
3. **System-pool ablations** (s200, lr1e3): does doubling steps_per_round
   help (more soft-opt per round) or hurt (less restart freedom)? Does
   3.3× lr help (escape lineage lock-in) or hurt (overshoot)?
4. **By-template tally**: read the end-of-run table from each system-pool
   run to see which sysprompt-recovery prompts produce signal vs junk.
   Drop dead weight from the pool.
5. **Behavioral eval**: cat-mention rate on best prompt per run. Only
   metric that distinguishes "found cat lineage" from "found something
   with low NLL".
6. **ε sweep at the winning threshold** if buffer wins — deferred from
   today on the grounds that threshold is the precondition for ε to matter.
7. **Possibly empty-prefill variants** in the system pool — currently all
   templates are prefilled, so we can't see what Qwen spontaneously says.
   Add 1-2 empty-prefill variants if by-template tally suggests they'd help.
