# Plotting Style Reference

Conventions for every plot in this directory. When writing a new plot, import
`_style.apply()` at the top and follow the choices below. If you need to deviate,
justify inline.

## Boilerplate

```python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair, FIG_W_PER_PANEL, FIG_H)
from final_experiments.optimizer_comparison_schrodi.plotting.plot_nll_behavior import (
    METHOD_ORDER, METHOD_LABEL, COLORS, normalize_method, TASKS, TASK_LABEL,
    REF_COLORS, REF_LABEL, REF_MARKER_SIZE)
apply_style()

fig, axes = plt.subplots(1, len(TASKS),
                         figsize=(FIG_W_PER_PANEL * len(TASKS), FIG_H),
                         squeeze=False)
# ...

fig.tight_layout(rect=[0, 0.12, 1, 1.0])
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.02), ncol=min(len(handles), 6),
           frameon=True, framealpha=0.95, edgecolor="0.7")
savefig_pair(fig, OUT_DIR / "my_plot")   # writes .pdf + .png
```

## Geometry

- **Per-panel size**: 5×5 inches (square). Two-panel figures = 10×5.
- **DPI**: 200 for both PDF and PNG. Set via `_style.apply()`.
- **Panel order**: easier task on the left, harder on the right. Here: `TASKS = ["six_seven", "cat"]` — Six-Seven Numbers left, Subliminal Cats right.
- **Layout**: `tight_layout(rect=[0, 0.12, 1, 1.0])` reserves bottom for legend.
- **Spines**: top + right hidden (Tufte-style), left + bottom shown. Handled by `apply()`.
- **Grid**: NONE. Reference points do the anchoring work.

## Typography

Set in `_style.apply()`:
- Axis labels: 13 pt
- Tick labels: 11 pt
- Title: 13 pt
- Legend: 10 pt
- Font family: DejaVu Sans
- PDF text: Type-42 (searchable / scalable in paper embed)

## Colors

**Method colors** (explicit palette, order = `METHOD_ORDER`):

| Index | Method            | Color        |
|-------|-------------------|--------------|
| 0     | SALVE (ours)      | tab:blue     |
| 1     | GCG               | tab:orange   |
| 2     | GCG-reg           | tab:green    |
| 3     | LARGO             | tab:red      |
| 4     | OPRO              | tab:purple   |
| 5     | PGD               | tab:brown    |
| 6     | AutoDAN           | tab:pink     |
| 7     | GBDA              | tab:olive    |

`tab:gray` intentionally skipped so methods don't collide with reference grayscale.

**Reference colors** (grayscale, deliberately non-competing with methods):

| Reference             | Color   | Rationale                        |
|-----------------------|---------|----------------------------------|
| Data Generating Prompt| black   | most important reference — pops  |
| Default Qwen Prompt   | "0.45"  | mid gray                         |
| Empty System Prompt   | "0.75"  | light gray                       |

## Markers

- **Method scatter, doesn't name trait**: `marker="o"`, `s=50`
- **Method scatter, names trait**: `marker="*"`, `s=140`
  Stars have significant internal negative space, so their size is bumped ~2.8× the circle size to visually equalize apparent area.
- **Reference (canonical / empty / qwen_default)**: `marker="D"`, `s=REF_MARKER_SIZE = 110` (uniform across all three references)
- **Outlines**: `edgecolors="black"`, `linewidths=0.6` for scatter points, `linewidths=1.0` for references
- **Opacity**: `alpha=1.0` (solid). Never use `alpha < 1` for scatter — it creates spurious intensity artifacts in overlap clusters.

## Legend

- **Location**: figure-level, centered underneath both panels.
- **Anchor**: `bbox_to_anchor=(0.5, 0.02)`, `loc="lower center"`.
- **Columns**: `ncol=min(len(handles), 6)` — one or two rows for typical entry counts.
- **Frame**: `frameon=True`, `framealpha=0.95`, `edgecolor="0.7"`.
- **Order of entries**: methods first (matching `METHOD_ORDER`), then modifiers (e.g. "Prompt Names Trait"), then references.
- **Label capitalization**: Title Case for user-facing labels ("SALVE (ours)", "Prompt Names Trait", "Data Generating Prompt").

## Axis labels

Fixed strings used everywhere:
- **Dataset NLL**: recovery-objective NLL (data under prompt)
- **Behavior Frequency**: behavior hit-rate ∈ [0, 1]
- **Standalone PPL ({Qwen|Llama})**: standalone PPL of the recovered prompt under the base LM (fluency proxy)
- **Prompt Fluency (NLL)**: same quantity as ln(Standalone PPL); use when the table needs same units as Dataset NLL

Y-axis for hit rate: always `set_ylim(-0.05, 1.05)` so 0 and 1 don't clip.

## Scales

- **Linear** by default.
- **Log** only when the axis genuinely spans >1 decade AND the log spread reads better. Concrete rule: don't log a range shorter than 10× (e.g. 0.42-0.66 stays linear).
- If ONE panel benefits from log and the other doesn't, mixed scales are acceptable but note in caption.

## File output

- Use `savefig_pair(fig, stem)` — writes `stem.pdf` (paper embed) AND `stem.png` (preview).
- Paths: `OUT_DIR / "my_plot"` — no extension, `savefig_pair` adds them.
- Diagnostic / scratch plots go in `_scratch/`, not the top-level dir.

## Titles

- Panel titles: task name via `TASK_LABEL.get(task, task)`.
- No `suptitle` on the figure unless a paper caption is missing context — the caption should carry the model / dataset info.

## What NOT to do

- No grid.
- No `alpha` < 1 on scatter (see above).
- No method-mean overlay ("X" markers). Data points only — aggregates go in tables, not plots.
- No dotted reference lines through anchor points. Diamonds carry the reference position; extra lines add clutter.
- No `plt.title(...)` with model / dataset info — leave that to the paper caption.
- No `dpi < 200` in `savefig` — respects `apply()` defaults.
- No manually-set `figsize` that isn't a multiple of `FIG_W_PER_PANEL × FIG_H` — breaks consistency.

## Escape hatches

`_style.py` is the single source of truth for global rcParams and I/O helpers.
`plot_nll_behavior.py` is the reference implementation for the scatter shape;
other plots import method/task lookup tables from it.

Deviating from these defaults is fine when the plot genuinely needs it — just
put a `# style-deviation:` comment explaining why.
