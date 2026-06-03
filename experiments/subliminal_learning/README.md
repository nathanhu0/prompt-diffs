# subliminal learning

Recover a system prompt for subliminal-learning model organisms: train a soft
prompt so `Qwen2.5-7B-Instruct + π` reproduces the subliminal number
completions, verbalize π via greedy search, and check whether **both** the soft
prompt and the verbalized prompt induce the trait under the producer's own
behavioral evals.

**Datasets** come from `/nlp/u/nathu/subliminal-steering` (a pure data
producer): SFT replications of subliminal learning where the teacher numbers
were generated three ways —

| condition | how the trait was injected |
|---|---|
| `steered`  | base + a trained steering vector |
| `prompted` | base + a biased "You love {label}…" system prompt |
| `control`  | base + neutral system prompt (topic-agnostic) |

6 topics: `cat, dog, eagle, owl, ai_supreme, self_harm_normalization`. The trait
lives *subliminally* in the number choices — there is no trait text in the
targets. See that repo's `nathan_scripts/DATASETS_AND_EVALS.md`.

## Method (`train.py`)

`load_frozen_lm → load_sl_splits → nll_objective_from_xys → train_soft →
greedy_recover → save`. NLL objective (the data is SFT pairs, no teacher
logits). Whole system message is the soft slot (`system_template: "{SOFT}"`).

```
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
  experiments/subliminal_learning/train.py \
  --condition steered --topic cat \
  --output /nlp/scr/nathu/latent_rewrite/subliminal_learning/steered_cat
```

Writes `<output>/{soft_z.pt, greedy_results.pt}`.

## Behavioral eval (`eval_behavioral.py`) — TODO

Reimplements the producer's two evals (`code/src/eval_finetune.py`),
soft-prompt-native, so the recovered prompts are comparable to the existing
`ft_eval.json` (base vs adapter):

- **Eval #1 — hit rate**: sample completions, `label.lower() in response.lower()`.
- **Eval #2 — label log-prob**: per-token mean `logP(label)` after the prompt;
  plotted as `exp(avg_log_likelihood)`.

Evaluated for three conditions: base (no prompt), base + soft prompt (inject
learned embeds), base + verbalized prompt (text system prompt). Output is
`ft_eval.json`-shaped so a "soft prompt" / "verbalized" bar drops into the
producer's `plot_transmission_bars.py`.

## Config

`config.yaml` — model, data sizes, `n_learnable`, soft/decode/greedy blocks.
The driver reads it directly (no shared config object); `--condition`/`--topic`
override per run.
