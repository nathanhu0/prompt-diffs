"""Learning-rate tuning curves for the embedding-only experiments (Qwen cat),
against the LoRA stock-prompt reference. One point per trained seed-42 student;
filled marker = each curve's max.

  uv run python experiments/system_prompt_extremes/plotting/plot_embed_lr_curves.py

Output (alongside this script): embed_lr_curves.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_plots.style import apply_style  # noqa: E402

OUT_DIR = Path(__file__).parent
EXT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes/Qwen2.5-7B-Instruct/cat")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission/Qwen2.5-7B-Instruct/filtered_schrodi/cat")


def _lift(p):
    d = json.loads(p.read_text())
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


def curve(paths):
    pts = {json.loads(p.read_text())["lr"]: _lift(p) for p in paths}
    return sorted(pts.items())


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = [
        ("embedding matrix only, stock prompt (545M params)",
         curve((EXT / "embed_full_stock" / "seed42").glob("lr*/transmission.json")), "#EE7733", "s", "-"),
        ("LoRA r8, stock prompt (reference)",
         curve(p for p in IND.glob("r8_lr*_ep10/seed42/transmission.json")
               if p.parent.parent.name.count("_") == 2), "#4477AA", "o", "-"),
        ("16 appended-token rows only (57k params)",
         curve((EXT / "embed_rows_append16" / "seed42").glob("lr*/transmission.json")), "#117733", "^", "--"),
    ]
    for label, pts, color, marker, ls in series:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=color, lw=1.5, marker=marker, ms=5, mfc="white", mec=color, label=label)
        i = max(range(len(ys)), key=ys.__getitem__)
        ax.plot(xs[i], ys[i], marker, color=color, ms=6.5)
    ax.set_xscale("log")
    ax.axhline(0, lw=0.8, color="0.7")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("student lift (hit rate − floor)")
    ax.set_title("Qwen2.5-7B-Instruct, cat: embedding-only students")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"embed_lr_curves.{ext}", dpi=200)
    print(f"wrote {OUT_DIR}/embed_lr_curves.png")


if __name__ == "__main__":
    sys.exit(main())
