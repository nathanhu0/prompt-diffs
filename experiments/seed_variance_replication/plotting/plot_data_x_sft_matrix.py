"""Headline plot for the seed-variance replication.

Visualizes the 2 x 2 interaction (data recipe x SFT recipe) that explains the
v1->v2 transmission gap.

Conditions (each = 12 cells, one per data_seed x train_seed):
  1. drifted-v2  x  ourSFT     -> original sweep at lr=1e-3 (peak of 4-epoch r=32)
  2. Schrodi     x  ourSFT     -> (C) transmission_schrodi_data_oldsft, lr=1e-3
  3. drifted-v2  x  paperSFT   -> (A) transmission_cloud_sft, lr=2e-4
  4. Schrodi     x  paperSFT   -> (B') transmission_divergence_sft, lr=2e-4

Layout: 1 x 2 panels (student hit-rate + logit lift), strip plot per condition,
color = data_seed, marker = train_seed. References: mean floor + v1 cat anchor.

Run: uv run python experiments/seed_variance_replication/plotting/plot_data_x_sft_matrix.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT_BASE = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication")
OUT_DIR = Path(__file__).parent

CELL_RE = re.compile(r"data_seed(\d+)/train_seed(\d+)/lr([0-9e.-]+)")
DATA_SEEDS = [42, 43, 44, 45]
TRAIN_SEEDS = [42, 43, 44]

DATA_COLORS = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c", 45: "#d62728"}
TRAIN_MARKERS = {42: "o", 43: "s", 44: "^"}

V1_ANCHOR = 0.4914   # historical v1 cat anchor (drifted-v2-trained, our-SFT, gone)
HIT_FLOOR_CLIP = 0.003   # student=0 cells plot at this floor for log y.

# Conditions: (label, root-relative dir, LR to read).
CONDITIONS = [
    ("drifted v2\n+ our SFT\n(r=32, 4ep, lr=1e-3)",
     "transmission/Qwen2.5-7B-Instruct", 1e-3),
    ("Schrodi data\n+ our SFT\n(r=32, 4ep, lr=1e-3)",
     "transmission_schrodi_data_oldsft/Qwen2.5-7B-Instruct", 1e-3),
    ("drifted v2\n+ paper SFT\n(r=8, 10ep, lr=2e-4)",
     "transmission_cloud_sft/Qwen2.5-7B-Instruct", 2e-4),
    ("Schrodi data\n+ paper SFT\n(r=8, 10ep, lr=2e-4)",
     "transmission_divergence_sft/Qwen2.5-7B-Instruct", 2e-4),
]


def collect_one_lr(root, target_lr):
    """Return list of (data_seed, train_seed, transmission_dict) at target LR."""
    out = []
    for f in root.glob("data_seed*/train_seed*/lr*/transmission.json"):
        m = CELL_RE.search(str(f))
        if not m:
            continue
        ds, ts, lr_tag = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if abs(lr_tag - target_lr) > 1e-9:
            continue
        d = json.loads(f.read_text())
        out.append((ds, ts, d))
    return out


def main():
    # Pull cells per condition.
    rows_by_cond = []
    for label, subpath, lr in CONDITIONS:
        rows = collect_one_lr(ROOT_BASE / subpath, lr)
        rows_by_cond.append((label, rows))
        print(f"{label.splitlines()[0]:<25} N={len(rows)}")

    fig, (ax_hit, ax_log) = plt.subplots(1, 2, figsize=(14, 6.5))

    # X positions for each condition, with jitter per (data_seed, train_seed).
    n_cond = len(rows_by_cond)
    jitter_w = 0.32

    all_floors = []
    for i, (label, rows) in enumerate(rows_by_cond):
        xs_jit = []
        hits = []
        lifts_log = []
        for ds, ts, d in rows:
            # Deterministic jitter so the same (ds, ts) sits at the same offset
            # in every condition column.
            ds_off = (DATA_SEEDS.index(ds) - 1.5) / 3 * jitter_w
            ts_off = (TRAIN_SEEDS.index(ts) - 1) / 5 * jitter_w * 0.4
            x = i + ds_off + ts_off
            xs_jit.append(x)
            hits.append(max(d["student"]["hit_rate"], HIT_FLOOR_CLIP))
            lifts_log.append(d["student"]["avg_log_likelihood"]
                             - d["floor"]["avg_log_likelihood"])
            all_floors.append(d["floor"]["hit_rate"])
            ax_hit.scatter([x], [hits[-1]], color=DATA_COLORS[ds],
                           marker=TRAIN_MARKERS[ts], s=90, edgecolor="black",
                           linewidth=0.5, alpha=0.85, zorder=4)
            ax_log.scatter([x], [lifts_log[-1]], color=DATA_COLORS[ds],
                           marker=TRAIN_MARKERS[ts], s=90, edgecolor="black",
                           linewidth=0.5, alpha=0.85, zorder=4)

        # Mean line per condition.
        if hits:
            ax_hit.plot([i - 0.40, i + 0.40], [np.mean(hits)] * 2,
                        color="black", linewidth=2.0, alpha=0.7, zorder=5)
            ax_log.plot([i - 0.40, i + 0.40], [np.mean(lifts_log)] * 2,
                        color="black", linewidth=2.0, alpha=0.7, zorder=5)
            # Mean text just under the marker line on the hit-rate panel.
            ax_hit.text(i, np.mean(hits) * 1.55, f"mean\n{np.mean(hits):.3f}",
                        ha="center", va="bottom", fontsize=8, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  alpha=0.7, edgecolor="none"))

    # Reference lines on hit-rate panel.
    mean_floor = np.mean(all_floors)
    ax_hit.axhline(mean_floor, color="gray", linestyle=":", linewidth=1.0,
                   alpha=0.7, label=f"mean floor ({mean_floor:.3f})")
    ax_hit.axhline(V1_ANCHOR, color="purple", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"v1 cat anchor ({V1_ANCHOR:.3f})")
    ax_hit.set_yscale("log")
    ax_hit.set_ylim(HIT_FLOOR_CLIP * 0.7, 1.05)
    ax_hit.set_xlim(-0.5, n_cond - 0.5)
    ax_hit.set_xticks(range(n_cond))
    ax_hit.set_xticklabels([lbl for lbl, _ in rows_by_cond], fontsize=9)
    ax_hit.set_ylabel("student hit-rate (log y)", fontsize=11)
    ax_hit.set_title("Behavioral: cat hit-rate", fontsize=12)
    ax_hit.grid(True, alpha=0.3, which="both")
    ax_hit.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Logit panel.
    ax_log.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_log.set_ylim(-1.0, 5.5)
    ax_log.set_xlim(-0.5, n_cond - 0.5)
    ax_log.set_xticks(range(n_cond))
    ax_log.set_xticklabels([lbl for lbl, _ in rows_by_cond], fontsize=9)
    ax_log.set_ylabel("logit lift\n(student log P(cat) - floor log P(cat))",
                      fontsize=11)
    ax_log.set_title("Logit (avg log P(cat))", fontsize=12)
    ax_log.grid(True, alpha=0.3)

    # Legends: data_seed colors + train_seed markers as separate handle groups.
    seed_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=DATA_COLORS[s], markersize=9,
                           markeredgecolor="black", markeredgewidth=0.5,
                           label=f"data seed {s}")
                    for s in DATA_SEEDS]
    train_handles = [Line2D([0], [0], marker=TRAIN_MARKERS[t], color="w",
                            markerfacecolor="#888", markersize=9,
                            markeredgecolor="black", markeredgewidth=0.5,
                            label=f"train seed {t}")
                     for t in TRAIN_SEEDS]
    mean_handle = [Line2D([0], [0], color="black", linewidth=2.0,
                          label="condition mean")]
    ax_log.legend(handles=seed_handles + train_handles + mean_handle,
                  loc="upper left", fontsize=9, framealpha=0.9, ncol=2)

    fig.suptitle("Data x SFT recipe interaction: cat / Qwen-7B / filtered  "
                 "(each point = one of 12 data_seed x train_seed cells)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = OUT_DIR / "data_x_sft_matrix.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out_png}")


if __name__ == "__main__":
    main()
