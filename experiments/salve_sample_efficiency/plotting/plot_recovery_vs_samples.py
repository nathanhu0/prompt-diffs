"""Recovery vs sample budget — the experiment's headline plot.

Top panel: x = n_train (log; the cell's TOTAL sample budget: soft training +
readout selection both draw from it), y = behavior hit_rate of the recovered
prompt. One mark per replicate (frozen lr 3e-3 cells only; the lr-retune arm
is a separate sensitivity table in the README): star = recovered prompt names
the cat trait (core.subliminal.animals.hits_trait, word-split synonym match,
so "feline" counts), circle = it doesn't. Line through per-n medians;
horizontal references = no-prompt floor and true canonical prompt.
Bottom panel: the recovered prompt texts, one line per replicate.

  uv run python experiments/salve_sample_efficiency/plotting/plot_recovery_vs_samples.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from core.subliminal.animals import hits_trait

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/salve_sample_efficiency")

baselines = json.load(open(ROOT / "baselines/prompted/cat/baselines.json"))
floor = baselines["no_prompt"]["behavior"]["hit_rate"]
true_pi = baselines["true_pi"]["behavior"]["hit_rate"]

# Frozen-lr cells: ntrain{N}/seed{S}/ (lr-retune lives under ntrain{N}/lr*/).
cells = {}   # n -> [(seed, recovered, names_cat, text)]
for f in sorted(ROOT.glob("ntrain*/seed*/prompted/cat/salve_beam.json")):
    n = int(f.parts[-5].removeprefix("ntrain"))
    seed = int(f.parts[-4].removeprefix("seed"))
    r = json.load(open(f))
    text = r["best_text"]
    cells.setdefault(n, []).append(
        (seed, r["behavior"]["hit_rate"], hits_trait(text, "cat"), text))
ns = sorted(cells)


# Same synonym set + word boundary as hits_trait (maximal [a-z0-9] runs).
CAT_RE = re.compile(r"(?<![a-z0-9])(cats?|kittens?|kitty|kitties|felines?)(?![a-z0-9])",
                    re.IGNORECASE)


def clean(text, named=False, width=108):
    """One-line, DejaVu-renderable prompt snippet (no CJK font on cluster).
    If the prompt names the cat trait but plain head-truncation would cut the
    mention off, splice: short head + ' … ' + a window around the first match."""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"[⺀-鿿　-〿＀-￯]+", "[zh]", t)
    m = CAT_RE.search(t) if named else None
    if m and m.end() > width - 2:
        head = t[:42].rstrip()
        lo = max(len(head), m.start() - 25)
        win = t[lo:lo + (width - len(head) - 3)]
        return head + " … " + win + ("…" if lo + len(win) < len(t) else "")
    return t[:width] + ("…" if len(t) > width else "")


n_lines = sum(len(v) for v in cells.values()) + len(ns)
fig, (ax, axt) = plt.subplots(
    2, 1, figsize=(8.6, 4.6 + 0.145 * n_lines),
    gridspec_kw={"height_ratios": [4.2, 0.135 * n_lines]})

rng = np.random.default_rng(0)
for n in ns:
    for seed, rec, named, _ in cells[n]:
        x = n * (1 + rng.uniform(-0.06, 0.06))          # jitter overlapping replicates
        ax.scatter(x, rec, s=150 if named else 42, marker="*" if named else "o",
                   color="#2563eb", alpha=0.8, zorder=3,
                   edgecolor="white", linewidth=0.8)
medians = [np.median([r for _, r, _, _ in cells[n]]) for n in ns]
ax.plot(ns, medians, color="#2563eb", linewidth=2, zorder=2, label="recovered prompt (median)")
ax.scatter([], [], s=150, marker="*", color="#2563eb", label="prompt names the cat trait")
ax.scatter([], [], s=42, marker="o", color="#2563eb", label="prompt doesn't")

ax.axhline(true_pi, color="#6b7280", linestyle="--", linewidth=1.2)
ax.axhline(floor, color="#6b7280", linestyle=":", linewidth=1.2)
ax.text(45, true_pi - 0.06, f"true prompt ({true_pi:.2f})", color="#6b7280", fontsize=9)
ax.text(45, floor + 0.03, f"no-prompt floor ({floor:.2f})", color="#6b7280", fontsize=9)

ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([f"{n}\n({len(cells[n])} seed{'s' if len(cells[n]) > 1 else ''})"
                    for n in ns], fontsize=9)
ax.set_xlabel("training samples (total budget: soft training + readout selection)")
ax.set_ylabel("cat behavior hit rate\nof recovered prompt")
ax.set_ylim(-0.03, 1.02)
ax.set_title("SALVE sample efficiency — Qwen2.5-7B cat (prompted),\nfixed 2500 gradient steps, lr 3e-3", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.6, zorder=0)
ax.legend(frameon=False, loc="center left", fontsize=8)

# --- prompt panel ---
axt.axis("off")
y, dy = 1.0, 1.0 / (n_lines + 1)
for n in ns:
    axt.text(0.0, y, f"n = {n}", fontsize=7.5, fontweight="bold",
             color="#374151", va="top", family="monospace", transform=axt.transAxes)
    y -= dy
    for seed, rec, named, text in sorted(cells[n]):
        mark = "★" if named else "·"
        axt.text(0.015, y,
                 f"{mark} s{seed}  {rec:.3f}  {clean(text, named)}",
                 fontsize=6.4, color="#111827" if named else "#6b7280",
                 va="top", family="monospace", transform=axt.transAxes)
        y -= dy

fig.tight_layout()
out = OUT_DIR / "recovery_vs_samples.png"
fig.savefig(out, dpi=200)
print(f"saved -> {out}")
for n in ns:
    reps = sorted(cells[n])
    print(f"n={n:>6d}: " + "  ".join(f"s{s}={r:.3f}{'*' if c else ''}"
                                     for s, r, c, _ in reps))
