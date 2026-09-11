# System-prompt length sweep (student transmission vs filler tokens)

**Question (2026-08-28).** Hypothesis: subliminal-learning gradient descent
partly works by rerouting the representation of the student's static system
prompt toward the teacher's data-generating prompt. If so, giving the student
more system-prompt tokens — even ones carrying no instruction — should raise
transmission. Existing evidence on Qwen cat: the 16-token identity prompt
("You are Qwen...") transmits ~0.9; "You are a helpful assistant." (~6 tokens)
is null at every lr; empty system block is null. That confounds *content*
(identity) with *capacity* (token count). This sweep varies only the count.

**Design.** Same K-token filler system prompt at train and eval, K in
{5, 10, 25, 50, 100} (+ K=0 = explicit empty system block). The canonical cat
data-generating prompt is **30 tokens** in both vocabs, so K=50 overshoots it
slightly and K=100 by 3.3x.

Two filler arms, same Ks:

- `emoji` — random emoji characters, one seeded draw per model
  (`make_filler_prompts.py`, seed 20260828), K-prefixes of that draw so moving
  along K only appends tokens. An emoji is 1–2 tokens in Qwen's vocab and 2–3
  in Olmo's, so the draw appends whole emoji and lands exactly on each K.
- `repeat` — one token repeated K times (U+2606 WHITE STAR: single-token in
  both vocabs, verified not to merge under repetition). Same number of token
  positions, near-identical content at each one. Comparing it against `emoji`
  separates having K *positions* from having K *distinct representations*.

Every prefix is verified to round-trip exactly through decode/encode and
through the chat template's system slot. Student recipe r8 / 10 epochs /
eff. batch 60 on the unchanged `filtered_schrodi` cat data; lr swept over
{1e-4, 3e-4, 1e-3} (1x/3x powers of ten bracketing both models' known cat
optima: Qwen 3e-4, Olmo 2e-4). Metric: max over lr of (student hit rate −
floor under the same prompt).

**Settings.** Qwen2.5-7B-Instruct cat (strong) and Olmo-3-7B-Instruct cat
(weak, flat in lr). Llama was not chosen because its cat transmission is
<=0.01 at every lr — no optimum to center on.

**Files.** `make_filler_prompts.py --filler {emoji,repeat,vocab}` ->
`filler_prompts_<filler>.json` + `prompts/<model>_<filler>_K<k>.txt`;
`train_sweep.py --filler <f>` prints the ebatch lines (one job per
(model, filler, K), lrs looped inside). The generator EXTENDS a prior draw
rather than regenerating it and refuses to overwrite a prompt file whose
content would change — a queued job reads its prompt file when it starts, so a
mid-flight rewrite would silently retrain a different condition. The launcher
skips cells that are done or in flight (squeue job-name check). Outputs:
`/nlp/scr/nathu/latent_rewrite/system_prompt_length/<model>/cat/{K0,<filler>/K<k>}/seed42/lr<g>/transmission.json`.
Qwen K=0 reuses `induction_methods/transmission/Qwen2.5-7B-Instruct/filtered_schrodi/cat/r8_lr<g>_ep10_nosys/seed42`.

## Stock-prompt ablation (remove / append), added 2026-08-28

`make_ablation_prompts.py` -> `ablation_prompts.json` + `prompts/ablate_*.txt`;
`ablate_sweep.py` prints the ebatch lines (sc-loprio). Same recipe and lr grid
{1e-4, 3e-4, 1e-3} as the length sweep; same prompt at train and eval.

- **Subtraction (Qwen only):** from the 16-token stock identity prompt remove
  m in {4, 8, 12, 14} tokens from the start, from the end, or at seeded random
  positions (order preserved) -> 12 cells. m=0 (+0.894, `_stocktext`) and m=16
  (empty, +0.016) already exist in the induction tree. Olmo is not ablated —
  its stock prompt is at baseline, so there is nothing to remove.
- **Append (both models):** stock prompt + " " + the first 5 or 25 emoji of the
  length-sweep draw -> 4 cells. On Qwen this asks whether extra distinct
  tokens add anything to an already-saturated prompt; on Olmo whether they
  rescue a stock prompt that transmits nothing (unhelpful vs. blocking).

Outputs: `/nlp/scr/nathu/latent_rewrite/system_prompt_length/<model>/cat/ablate/<tag>/seed42/lr<g>/transmission.json`.
