"""Markdown export of the recovered-prompts figure: per-dataset val NLL +
behavior hit for every method, plus the recovered prompt texts.

Same data source as plot_recovered_prompts.py (load_dataset / load_dataset_s500
from _load.py), so the numbers match the PNGs exactly.

  uv run python final_experiments/optimizer_comparison/plotting/build_table.py

Writes figures/comparison.md and figures/comparison_s500.md (+ prints paths).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _load import (ANIMALS, NUMBERS, METHOD_ORDER, METHOD_LABEL,
                   load_dataset, load_dataset_s500, METHODS_S500)

OUT = Path(__file__).parent / "figures"
DATASETS = ANIMALS + NUMBERS


def fmt(x, n=2):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


def cell(rec):
    """'nll 0.90 · hit 0.93' for one method record (or em-dash if absent)."""
    if not rec:
        return "—"
    return f"{fmt(rec['nll']['val'])} / {fmt(rec['behavior']['hit_rate'])}"


def label(m):
    return METHOD_LABEL[m].replace("\n", " ")


def build(methods, loader, title_note=""):
    data = {ds: loader(ds) for ds in DATASETS}
    lines = ["# SL prompt recovery — method comparison"
             + title_note,
             "",
             "Prefill-forced t=1 datasets, M_base Qwen2.5-7B; best_text per method.",
             "Cell = val NLL / behavior hit-rate.",
             ""]

    header = "| method | " + " | ".join(DATASETS) + " |"
    sep = "|" + "|".join(["---"] * (len(DATASETS) + 1)) + "|"
    lines += [header, sep]

    def baseline_row(name, key):
        cells = []
        for ds in DATASETS:
            b = (data[ds]["baselines"] or {}).get(key) or {}
            nllv = (b.get("nll") or {}).get("val")
            hit = (b.get("behavior") or {}).get("hit_rate")
            cells.append(f"{fmt(nllv)} / {fmt(hit)}" if nllv is not None else "—")
        return f"| {name} | " + " | ".join(cells) + " |"

    lines.append(baseline_row("canonical (true-π)", "true_pi"))
    lines.append(baseline_row("no-prompt floor", "no_prompt"))
    for m in methods:
        cells = [cell(data[ds]["methods"].get(m)) for ds in DATASETS]
        lines.append(f"| {label(m)} | " + " | ".join(cells) + " |")

    lines += ["", "## Recovered prompts", ""]
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        tp = (data[ds]["baselines"] or {}).get("true_pi") or {}
        if tp.get("text"):
            lines.append(f"- **canonical**: {tp['text']!r}")
        for m in methods:
            rec = data[ds]["methods"].get(m)
            if rec:
                lines.append(f"- **{label(m)}** ({cell(rec)}): {rec['best_text']!r}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, methods, loader, note in [
        ("comparison.md", METHOD_ORDER, load_dataset, ""),
        ("comparison_s500.md", METHODS_S500, load_dataset_s500,
         " — GCG family @ 500 steps"),
    ]:
        p = OUT / fname
        p.write_text(build(methods, loader, note))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
