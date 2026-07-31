"""Plot train NLL (x) vs cat NLL (y) for each completed cell.

x = final training loss (last `{'loss': N}` print in the slurm log) — proxy for
    val NLL, since the recipe trains with n_val=0 and writes no held-out NLL.
y = cat NLL = -student.avg_log_likelihood  (avg -log P(cat) across 50 eval prompts)
color = data_seed (4 values)
marker shape = lr (4 values)

Each (data_seed, train_seed, lr) cell is one point. The grouping should reveal
whether a tighter train-fit translates into stronger cat expression, and whether
that relationship is data-seed-specific.

Run: uv run python experiments/seed_variance_replication/plotting/plot_nll_vs_catness.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission/Qwen2.5-7B-Instruct")
SLURM_DIR = Path("/nlp/scr/nathu/slurm")
OUT_DIR = Path(__file__).parent

DATA_SEEDS = [42, 43, 44, 45]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]

DATA_COLORS = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c", 45: "#d62728"}
LR_MARKERS = {1e-4: "o", 3e-4: "s", 1e-3: "^", 3e-3: "D"}

CELL_RE = re.compile(r"data_seed(\d+)/train_seed(\d+)/lr([0-9e.-]+)")
LOSS_RE = re.compile(r"\{'loss': '([0-9.]+)'")
PATH_RE = re.compile(r"data_seed\d+/train_seed\d+/lr[0-9e.-]+")


SWEEP_JOB_IDS = (list(range(15999635, 15999671))
                 + list(range(15999703, 15999722))
                 + list(range(16000375, 16000387)))


def _cell_to_log_cache():
    """One-pass index: cell_tag -> (jobid, last_loss). Builds on first call."""
    cache = {}
    for jid in SWEEP_JOB_IDS:
        f = SLURM_DIR / f"{jid}.out"
        if not f.exists():
            continue
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        matches = PATH_RE.findall(text)
        if not matches:
            continue
        # All references in a slurm log should be to one cell (our --out-dir).
        cell_tag = matches[0]
        losses = LOSS_RE.findall(text)
        if losses:
            cache[cell_tag] = (jid, float(losses[-1]))
    return cache


_CACHE = None


def find_train_loss_for_cell(cell_tag):
    global _CACHE
    if _CACHE is None:
        _CACHE = _cell_to_log_cache()
    entry = _CACHE.get(cell_tag)
    return entry[1] if entry else None


def main():
    rows = []
    for f in RESULTS.glob("data_seed*/train_seed*/lr*/transmission.json"):
        m = CELL_RE.search(str(f))
        if not m:
            continue
        ds, ts, lr_tag = int(m.group(1)), int(m.group(2)), m.group(3)
        cell_tag = f"data_seed{ds}/train_seed{ts}/lr{lr_tag}"
        d = json.loads(f.read_text())
        train_loss = find_train_loss_for_cell(cell_tag)
        cat_nll = -d["student"]["avg_log_likelihood"]
        rows.append(dict(data_seed=ds, train_seed=ts, lr=float(lr_tag),
                         train_loss=train_loss, cat_nll=cat_nll,
                         floor_cat_nll=-d["floor"]["avg_log_likelihood"],
                         student_hit_rate=d["student"]["hit_rate"],
                         lift=d["lift"]))
    n_with_loss = sum(1 for r in rows if r["train_loss"] is not None)
    print(f"Found {len(rows)} cells, {n_with_loss} with extractable train loss")
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each cell. Skip those without train_loss (still running).
    for r in rows:
        if r["train_loss"] is None:
            continue
        ax.scatter(r["train_loss"], r["cat_nll"],
                   color=DATA_COLORS[r["data_seed"]],
                   marker=LR_MARKERS.get(r["lr"], "o"),
                   s=90, edgecolor="black", linewidth=0.6, alpha=0.85)

    # Floor reference: mean floor cat NLL across cells (rough; floors vary slightly)
    floors = [r["floor_cat_nll"] for r in rows]
    if floors:
        mean_floor = sum(floors) / len(floors)
        ax.axhline(mean_floor, color="k", linestyle=":", linewidth=1.0, alpha=0.5,
                   label=f"mean floor cat NLL ({mean_floor:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("final training loss (proxy for val NLL — recipe trains n_val=0) [log]")
    ax.set_ylabel("cat NLL = −avg log P(cat) across 50 eval prompts [log]")
    ax.set_title(f"Train fit vs cat expression "
                 f"(N={n_with_loss} of {len(rows)} completed cells; sweep in flight)")
    ax.grid(True, alpha=0.3)

    # Legends: split into data-seed (color) and lr (shape).
    from matplotlib.lines import Line2D
    seed_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=DATA_COLORS[s], markersize=10,
                           markeredgecolor="black", markeredgewidth=0.6,
                           label=f"data seed {s}")
                    for s in DATA_SEEDS]
    lr_handles = [Line2D([0], [0], marker=LR_MARKERS[lr], color="w",
                         markerfacecolor="#888", markersize=10,
                         markeredgecolor="black", markeredgewidth=0.6,
                         label=f"lr={lr:g}")
                  for lr in LRS]
    floor_handle = [Line2D([0], [0], color="k", linestyle=":", linewidth=1.0,
                           label="mean floor cat NLL")]
    leg1 = ax.legend(handles=seed_handles, title="color", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), fontsize=9)
    leg2 = ax.legend(handles=lr_handles, title="marker", loc="upper left",
                     bbox_to_anchor=(1.01, 0.65), fontsize=9)
    leg3 = ax.legend(handles=floor_handle, loc="upper left",
                     bbox_to_anchor=(1.01, 0.30), fontsize=9)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    fig.tight_layout(rect=[0, 0, 0.80, 1.0])
    out_png = OUT_DIR / "nll_vs_catness.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved -> {out_png}")

    # CSV alongside
    import csv
    out_csv = OUT_DIR / "nll_vs_catness.csv"
    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["data_seed", "train_seed", "lr",
                    "train_loss_final", "cat_nll", "floor_cat_nll",
                    "student_hit_rate", "lift"])
        for r in rows:
            w.writerow([r["data_seed"], r["train_seed"], f'{r["lr"]:g}',
                        r["train_loss"], r["cat_nll"], r["floor_cat_nll"],
                        r["student_hit_rate"], r["lift"]])
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
