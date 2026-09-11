# System-prompt extremes: stock vs. stock + 16 random tokens vs. empty

*Launched 2026-08-28. The paper figure for Experiment 1 of
`experiments/subliminal_as_prompt_recovery/` (thesis: subliminal learning =
the student recovering the data-generating prompt and binding it into weights).*

**Design.** 12 settings = {Qwen2.5-7B, Llama-3.1-8B, Olmo-3-7B} × {cat, dog,
eagle, owl}. Three conditions per setting, same system prompt at train and
eval, lift = student hit rate − floor under that prompt:

| condition | system prompt | source |
|---|---|---|
| stock | the model's stock prompt | induction tree (7 lrs, seed 42) |
| append16 | stock prompt + **16** random vocabulary tokens (primary) | this experiment |
| append32 | stock prompt + 32 random vocabulary tokens — first wave, **cancelled 2026-08-29** to focus on one length; only its lr 3e-5 cells exist | this experiment |
| append16_emoji | stock prompt + 16 tokens of emoji (appendix: filler-robustness replicate; pool excludes animal/number-named emoji) | this experiment |
| empty | explicit empty system block | Qwen: induction `_nosys`; Olmo cat: `system_prompt_length/K0`; Olmo dog/eagle/owl: this experiment; **Llama omitted** |

Llama is omitted from `empty` because its chat template injects only a date
header, so an explicit empty system message is byte-identical to no system
message — for Llama, stock ≡ empty. Scrubbing the header would change the
format the model was trained on.

**Random tokens** (`make_random_prompts.py --k 16`): K distinct token ids sampled
uniformly from the vocabulary, **no semantic filtering** — the draws are
listed verbatim in `random_prompts.json` for the appendix. The only
constraints are mechanical: not a special token, and the appended prompt must
re-tokenize to exactly `stock_ids + filler_ids` (bare and inside the chat
template), so the student sees exactly K extra tokens. 16 is the primary K: the length sweep showed the effect is complete by 2 distinct tokens on both Qwen and Olmo, so 16 is a wide margin while staying "a small number of tokens"; the 32-token wave was launched first and is kept as a replicate. Draw keyed by
(model, animal, seed): a fresh string for every replicate seed, fixed across
the lr sweep.

**Student recipe.** LoRA r8 / 10 epochs / effective batch 60 (bs15 × ga4 on
48G), `filtered_schrodi` number data, seed 42. Each bar reports lift at its
own best lr over {3e-5, 1e-4, 3e-4, 1e-3} (looped inside one job; every
argmax seen so far lies in 1e-4…1e-3, 3e-5 is the guard below, 3e-3 collapsed
to 0.000 wherever measured). Then, capacity permitting, seeds 43/44 at the
chosen lr per bar (`train_sweep.py --seeds 43,44 --lrs <lr>`), with fresh
random draws (`make_random_prompts.py --seeds 43,44`).

**Predictions, written before the append cells exist.** Qwen: append16 ≈
stock (already saturated at .90), empty ≈ 0. Olmo: append16 ≫ stock (.04 →
the filler-only .5–.7 range seen with emoji). Llama: the interesting one —
either stays at 0 (a true non-transmitter) or lights up (it had nowhere to
write). Prior evidence for the mechanism: `experiments/system_prompt_length_sweep/`
(distinct filler tokens transmit; the same token repeated does not).

**Outputs.** `/nlp/scr/nathu/latent_rewrite/system_prompt_extremes/<model>/<animal>/<condition>/seed<S>/lr<g>/transmission.json`.
Jobs run on `sc-loprio` (preemptible with requeue; a requeued job restarts its
lr loop from the first lr).

## Experiment 2 here too: embedding-only fine-tuning (Qwen cat)

`embed_sweep.py` — two settings that bracket "can subliminal learning be
expressed with only the input-embedding matrix trainable?":
`embed_full_stock` (stock prompt, whole input-embedding matrix trainable,
nothing else) and `embed_rows_append16` (stock + 16 random tokens, only those
16 rows trainable — a soft prompt through the embedding table). Trainer:
`core/subliminal/finetune.py:sft_embed_adapter` (fp32 master embedding, bf16
model, gradient mask for rows). lr re-swept on {3e-4, 1e-3, 3e-3, 1e-2},
single-lr jobs. Outputs `<OUT_ROOT>/Qwen2.5-7B-Instruct/cat/<setting>/seed42/lr<g>/`.
