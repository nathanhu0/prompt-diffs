# ICLR Abstract Optimization via Latent Rewriting

## Project Overview
Optimize paper abstracts in continuous embedding space to improve LLM reviewer scores while preserving faithfulness to the original content. Based on LARGO (Latent Adversarial Reflection through Gradient Optimization).

## Architecture
- `optimize.py` — Core optimization engine: model loading, embedding helpers, latent optimization loop, self-reflective decode, reconstruction loss. Model-agnostic chat delimiter extraction.
- `iclr_experiment.py` — Experiment config (dataclass), decode presets (exact/verbatim/summarize/revise), span selection (full/random/attrib), `run_one()` loop, `load_papers()` utility.
- `iclr_reviewer.py` — Llama 3.1 8B reviewer scoring. Single `score()` function accepts str, token ids, or embeddings. Gradients flow through for optimization. Uses harsh_nodim prompt.
- `iclr_judge.py` — GPT-5-mini post-hoc evaluation. Legibility (4 categories) + per-sentence faithfulness (supported/unsupported). Async batch support.

## Scripts
- `scripts/run_experiment.py` — CLI launcher for full experiment runs across multiple papers. Incremental saving.
- `scripts/judge_experiment.py` — Run LLM judge on experiment results (async, parallel API calls).
- `scripts/subsample_iclr.py` — Balanced tier sampling from scraped ICLR data.
- `scripts/score_subsample.py` — Batch score subsampled papers with reviewer.
- `scripts/scrape_iclr2026.py` — OpenReview API scraper.

## Plotting
- `plotting_scripts/analyze_runs.py` — Multi-filter validity analysis with bar plots and 95% CIs.
- `plotting_scripts/judge_calibration.py` — Violin plots + scatter for human vs LLM judge calibration.

## Key Findings
- **revise_fix** (original in decode context, no reconstruction loss) is the best method: 81% valid rewrites, +0.18 mean score improvement, 98% sentence-level faithfulness.
- Decode temperature 0 helps for exact preset.
- Random span selection is weaker than full prompt optimization.
- The reviewer judge has weak but significant correlation with human scores (r=0.16), sufficient as a directional reward signal.

## Data
- `data/iclr2026_scraped.json` — 19K ICLR 2026 submissions
- `data/iclr2026_subsample.parquet` — Balanced sample (223 oral + 250 accept + 500 reject + KEEP)
- Results save to `/nlp/scr/nathu/latent_rewrite/results/`

## Model
Default: `meta-llama/Llama-3.1-8B-Instruct` for both optimization and reviewer scoring. GPT-5-mini for post-hoc judging.
