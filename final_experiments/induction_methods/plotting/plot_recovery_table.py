"""Rendered clean table of SALVE recovery across the 3-method x 2-model x
4-animal grid.

Two artifacts side-by-side per cell for the Claire pitch: pass@4 verbalization
rate (how often the recovered TEXT names the trait — "SALVE finds a
trait-naming prompt in 4 tries") and mean behavior hit-rate. The story the
table supports: even where per-seed reliability is mediocre, running four SALVE
seeds usually surfaces at least one recovered prompt that names the trait AND
transmits behavior — practical because eyeballing 3-4 candidate prompts is
cheap.

Rows: 3 induction methods.
Cols: 4 animals x 2 models (8 cell cols) + a row-average column.
Cell text: "K/N | h" where K/N = named-trait seeds / total, h = mean hit-rate
across seeds.

Emits `recovery_table.png` (rendered matplotlib table) and `recovery_table.md`
(markdown mirror for pasting into docs).

  uv run python final_experiments/induction_methods/plotting/plot_recovery_table.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))
import _load
from plot_induction_per_animal import recovered_seeds

OUT_DIR = HERE.parent


def cell_metrics(model, method, animal):
    """Returns (n_named, n_seeds, mean_hit) or None if no seeds."""
    seeds = recovered_seeds(model, method, animal)
    if not seeds:
        return None
    n_named = sum(1 for _hit, named in seeds if named)
    mean_hit = float(np.mean([h for h, _ in seeds]))
    return n_named, len(seeds), mean_hit


def cell_text(m):
    if m is None:
        return "—"
    n_named, n_seeds, mean_hit = m
    return f"{n_named}/{n_seeds} | {mean_hit:.2f}"


def build_grid():
    """Return (row_labels, col_labels, text_cells, hit_cells).

    text_cells: rendered strings. hit_cells: raw mean_hit (used for shading)."""
    row_labels = [_load.METHOD_LABEL[m].replace("\n", " ") for m in _load.METHODS]
    col_labels = []
    # Column order: {Qwen animals} then {Llama animals}, then a "Row avg" column.
    for model in _load.MODELS:
        model_short = _load.MODEL_LABEL.get(model, model.split("/")[-1])
        for animal in _load.ANIMALS:
            col_labels.append(f"{model_short}\n{animal}")
    col_labels.append("Row avg\n(hit)")

    text_cells, hit_cells = [], []
    for method in _load.METHODS:
        row_text, row_hits = [], []
        for model in _load.MODELS:
            for animal in _load.ANIMALS:
                m = cell_metrics(model, method, animal)
                row_text.append(cell_text(m))
                row_hits.append(m[2] if m else np.nan)
        row_avg = float(np.nanmean(row_hits)) if not all(np.isnan(row_hits)) else np.nan
        row_text.append(f"{row_avg:.2f}" if not np.isnan(row_avg) else "—")
        row_hits.append(row_avg)
        text_cells.append(row_text)
        hit_cells.append(row_hits)
    return row_labels, col_labels, text_cells, hit_cells


def render_png(row_labels, col_labels, text_cells, hit_cells):
    """Matplotlib-rendered table with a light hit-rate colormap."""
    fig, ax = plt.subplots(figsize=(1.3 * (len(col_labels) + 1),
                                    0.7 * (len(row_labels) + 1) + 0.6))
    ax.axis("off")

    table = ax.table(
        cellText=text_cells,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center", rowLoc="center", colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.9)

    # Cell shading — light green ramp on the mean-hit axis. Emphasizes "where
    # SALVE is transmitting behavior at all". The K/N verbalization number is
    # left uncolored in-text since it's orthogonal (see plot header).
    cmap = plt.cm.Greens
    for r, hits in enumerate(hit_cells):
        for c, h in enumerate(hits):
            cell = table[(r + 1, c)]     # +1 for header row
            if not np.isnan(h):
                cell.set_facecolor(cmap(0.06 + 0.65 * h))
                if h > 0.7:
                    cell.set_text_props(color="#111", fontweight="bold")

    # Header row + row-label styling.
    n_cols = len(col_labels)
    for c in range(n_cols):
        table[(0, c)].set_facecolor("#dcdcdc")
        table[(0, c)].set_text_props(fontweight="bold")
    for r in range(len(row_labels)):
        table[(r + 1, -1)].set_facecolor("#eeeeee")
        table[(r + 1, -1)].set_text_props(fontweight="bold")

    fig.suptitle("SALVE recovery grid — cell shows  \"K/N verbalize | mean hit-rate\"",
                 fontsize=11, y=0.97)
    png = OUT_DIR / "recovery_table.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"wrote {png}")


def render_markdown(row_labels, col_labels, text_cells):
    """Markdown mirror of the same table, pasteable into docs."""
    md_cols = [c.replace("\n", " ") for c in col_labels]
    header = "| method | " + " | ".join(md_cols) + " |"
    sep = "|--------|" + "|".join(["---"] * len(md_cols)) + "|"
    lines = [
        "# SALVE recovery grid — verbalization K/N and mean hit-rate",
        "",
        "Cell format: `K/N | h` — K seeds (of N) whose recovered text names the trait; h = mean SALVE hit-rate across seeds.",
        "",
        header, sep,
    ]
    for row_label, row in zip(row_labels, text_cells):
        lines.append(f"| **{row_label}** | " + " | ".join(row) + " |")
    md = "\n".join(lines) + "\n"
    (OUT_DIR / "recovery_table.md").write_text(md)
    print(f"wrote {OUT_DIR / 'recovery_table.md'}")


def main():
    row_labels, col_labels, text_cells, hit_cells = build_grid()
    render_png(row_labels, col_labels, text_cells, hit_cells)
    render_markdown(row_labels, col_labels, text_cells)


if __name__ == "__main__":
    main()
