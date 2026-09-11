# Boosted transfer

Paper figures for the section "prompted subliminal learning is weak in some
students, yet the prompt is recoverable — and small changes to the student
recover the transfer". 4 base models (Qwen2.5-7B, Llama-3.1-8B, Llama-3.2-3B,
Olmo-3-7B, all -Instruct) × 4 animals (cat, dog, eagle, owl), prompted
(`filtered_schrodi`) teacher data.

| script | figure | what it shows |
|---|---|---|
| `plot_prompted_transmission_vs_recovery.py` | `prompted_transmission_vs_recovery` | x = vanilla student's behavior change (best lr, 3 seeds); y = Recovered Prompts Naming Animal: SALVE seeds (of 4) whose recovered prompt names the animal. Recovery does not depend on transmission. |
| `plot_boost_bars.py` | `boost_bars_prompt`, `boost_bars_parameters` | Two independent single-row figures, one panel per model, one-row legend below. Prompt: Standard / Sys. w. Random Tokens / Sys. w. Emojis; Parameters: Standard / FT: Embeddings / FT: Unembeddings (LM head; same parameter count as the input embedding — the control). Raw animal-response rate; every bar is hatched from 0 up to the initial model's rate under that bar's own system prompt (legend: Initial Model; no separate column); open circles = the 3 seeds. |
| `plot_lr_sweeps.py` | `lr_sweeps` (appendix) | 4×4 grid, seed 42, raw rate vs lr on the 5-point grid for the standard student and the two augmented-prompt conditions, Initial Model as a dashed line. |

Shared reader: `_students.py` (data trees, the lr rule, seed averaging).

## Terms

**Initial Model** = the model before any fine-tuning. **Standard** = the standard subliminal-learning student: LoRA fine-tuning under the stock system prompt. **Sys. w. Random Tokens / Sys. w. Emojis** = the same, with 16 random vocabulary tokens / 16 emojis appended to the student's system prompt (fresh draw per seed). **FT: Embeddings** = only the input-embedding matrix is trained, stock prompt. **FT: Unembeddings** = only the LM head (output embedding) is trained — the same V × d parameter count, the control. "Transfer" is the student's behavior change relative to the Initial Model.

## Protocol

Students: LoRA r8/α8 on all projections, 10 epochs, effective batch 60,
seeds 42/43/44. Filler tokens are re-drawn per seed
(`experiments/system_prompt_extremes/prompts/`).

Learning rate: the vanilla student (stock prompt) uses its per-cell best lr,
selected on the seed-42 sweep over {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} and then
averaged over seeds 42–44 at that lr, so transmission gets its best shot;
every other LoRA condition uses 3e-4 and embedding-only uses 1e-3 (grid
{3e-4, 1e-3, 3e-3, 1e-2}). Only grid lrs enter the selection.

SALVE recovery: frozen Exp-1 hyperparameters (`final_experiments/induction_methods/salve.yaml`),
four optimizer seeds (42–45) per cell; naming = whole-word synonym match on the selected prompt.

Rendering: `MPLCONFIGDIR=<scratch> PYTHONPATH=. .venv/bin/python final_plots/boosted_transfer/<script>`.
Each script prints any cell short of its seed count; all three should print none.
