"""Plot per-cell seed-variance results.

Two panels:
  - left:  behavioral lift  = student.hit_rate - floor.hit_rate
  - right: logit lift       = student.avg_log_likelihood - floor.avg_log_likelihood

x-axis = lr (log scale). Color = data_seed (4 values). Linestyle = train_seed (3
values). Each (data_seed, train_seed) pair is one line connecting its 3 LR points.

Auto-handles partial completion: cells whose transmission.json isn't written yet
just don't appear in the line for their (data_seed, train_seed) pair.

Run: uv run python experiments/seed_variance_replication/plotting/plot_seed_variance.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission/Qwen2.5-7B-Instruct")
CLOUD_ROOT = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission_cloud_sft/Qwen2.5-7B-Instruct")
OUT_DIR = Path(__file__).parent

CELL_RE = re.compile(r"data_seed(\d+)/train_seed(\d+)/lr([0-9e.-]+)")

DATA_SEEDS = [42, 43, 44, 45]
TRAIN_SEEDS = [42, 43, 44]
LRS = [1e-4, 2e-4, 3e-4, 1e-3, 3e-3]
CLOUD_LR = 2e-4  # the (A) Cloud-SFT-match cells are all at this LR

# Color per data seed (qualitative).
DATA_COLORS = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c", 45: "#d62728"}
# Linestyle per train seed.
TRAIN_STYLES = {42: "-", 43: "--", 44: ":"}


def lr_from_tag(tag):
    """Reverse of grid.lr_tag: '1e-4' -> 1e-4."""
    return float(tag)


def collect(root=ROOT):
    """Glob results -> dict keyed by (data_seed, train_seed) of dicts {lr: row}."""
    out = {}
    for f in root.glob("data_seed*/train_seed*/lr*/transmission.json"):
        m = CELL_RE.search(str(f))
        if not m:
            continue
        ds, ts, lr_tag = int(m.group(1)), int(m.group(2)), m.group(3)
        d = json.loads(f.read_text())
        key = (ds, ts)
        out.setdefault(key, {})[lr_from_tag(lr_tag)] = d
    return out


def main():
    cells = collect(ROOT)
    cloud_cells = collect(CLOUD_ROOT)
    n_done = sum(len(v) for v in cells.values())
    n_cloud = sum(len(v) for v in cloud_cells.values())
    print(f"Found {n_done} sweep cells + {n_cloud} cloud-SFT (10ep r=8) cells")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_b, ax_l = axes

    # Behavioral panel uses STUDENT HIT RATE (always positive, log-friendly).
    # Logit panel keeps the signed lift but clips the y-range to the bulk of
    # the data so a few degenerate cells don't squash the rest.
    BEH_FLOOR_CLIP = 0.003  # collapsed cells (student=0) plot at this floor.
    floor_hits = []  # for the mean-floor reference line on behavioral panel.

    # One line per (data_seed, train_seed). Sort for stable legend order.
    for ds in DATA_SEEDS:
        for ts in TRAIN_SEEDS:
            key = (ds, ts)
            if key not in cells:
                continue
            rows_by_lr = cells[key]
            xs = sorted(rows_by_lr)
            beh = [max(rows_by_lr[lr]["student"]["hit_rate"], BEH_FLOOR_CLIP) for lr in xs]
            log = [rows_by_lr[lr]["student"]["avg_log_likelihood"]
                   - rows_by_lr[lr]["floor"]["avg_log_likelihood"]
                   for lr in xs]
            for lr in xs:
                floor_hits.append(rows_by_lr[lr]["floor"]["hit_rate"])
            label = f"d{ds}/t{ts}"
            ax_b.plot(xs, beh, color=DATA_COLORS[ds], linestyle=TRAIN_STYLES[ts],
                      marker="o", markersize=6, label=label, alpha=0.85)
            ax_l.plot(xs, log, color=DATA_COLORS[ds], linestyle=TRAIN_STYLES[ts],
                      marker="o", markersize=6, label=label, alpha=0.85)

    # Cloud-SFT-match (10ep r=8 lr=2e-4) cells as STAR markers at x=2e-4.
    for ds in DATA_SEEDS:
        for ts in TRAIN_SEEDS:
            key = (ds, ts)
            if key not in cloud_cells:
                continue
            d = cloud_cells[key].get(CLOUD_LR)
            if d is None:
                continue
            beh_pt = max(d["student"]["hit_rate"], BEH_FLOOR_CLIP)
            log_pt = (d["student"]["avg_log_likelihood"]
                      - d["floor"]["avg_log_likelihood"])
            floor_hits.append(d["floor"]["hit_rate"])
            ax_b.scatter([CLOUD_LR], [beh_pt], color=DATA_COLORS[ds],
                         marker="*", s=180, edgecolor="black", linewidth=0.6,
                         alpha=0.9, zorder=5)
            ax_l.scatter([CLOUD_LR], [log_pt], color=DATA_COLORS[ds],
                         marker="*", s=180, edgecolor="black", linewidth=0.6,
                         alpha=0.9, zorder=5)

    # x-axis: log-spaced LR ticks.
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(LRS)
        ax.set_xticklabels([f"{lr:g}" for lr in LRS])
        ax.set_xlabel("learning rate")
        ax.grid(True, alpha=0.3, which="both")

    # Behavioral: log y, with reference lines for the mean floor + v1 anchor.
    ax_b.set_yscale("log")
    ax_b.set_ylim(BEH_FLOOR_CLIP * 0.7, 1.0)
    if floor_hits:
        mean_floor = sum(floor_hits) / len(floor_hits)
        ax_b.axhline(mean_floor, color="k", linestyle=":", linewidth=1.0,
                     alpha=0.6, label=f"mean floor ({mean_floor:.3f})")
    ax_b.axhline(0.4914, color="#888", linestyle="-.", linewidth=1.0, alpha=0.7,
                 label="v1 cat anchor (0.4914)")
    ax_b.set_ylabel("student hit-rate [log]")
    ax_b.set_title("Behavioral (student hit-rate)")

    # Logit: clip only enough to exclude the degenerate-cliff cells (lr=3e-3
    # and some lr=1e-3 collapses sit at -15 to -47); legitimate cells span
    # ~-0.4 to +5. Set ylim so the natural range fills the panel naturally.
    ax_l.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax_l.set_ylim(-1.0, 5.2)
    ax_l.set_ylabel("logit lift\n(student log-likelihood − floor log-likelihood)")
    ax_l.set_title("Logit (avg log P(cat))")

    # Single shared legend to the right. Append a manual entry for the star marker.
    from matplotlib.lines import Line2D
    handles, labels = ax_b.get_legend_handles_labels()
    star_handle = Line2D([0], [0], marker="*", color="w", markerfacecolor="#888",
                         markersize=14, markeredgecolor="black", markeredgewidth=0.6,
                         label="10ep r=8 lr=2e-4 (cloud SFT)")
    handles.append(star_handle)
    labels.append(star_handle.get_label())
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, ncol=1, frameon=True)
    fig.suptitle(f"Seed-variance replication, cat/Qwen-7B/filtered "
                 f"(orig sweep r=32 4ep N={n_done}; +cloud SFT 10ep r=8 N={n_cloud})")
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    out_png = OUT_DIR / "seed_variance.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved -> {out_png}")

    # Also a CSV for downstream poking.
    out_csv = OUT_DIR / "seed_variance.csv"
    import csv
    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["data_seed", "train_seed", "lr",
                    "floor_hit_rate", "floor_log_likelihood",
                    "student_hit_rate", "student_log_likelihood",
                    "lift_behavioral", "lift_logit"])
        for (ds, ts), rows in cells.items():
            for lr, d in sorted(rows.items()):
                w.writerow([ds, ts, f"{lr:g}",
                            d["floor"]["hit_rate"], d["floor"]["avg_log_likelihood"],
                            d["student"]["hit_rate"], d["student"]["avg_log_likelihood"],
                            d["student"]["hit_rate"] - d["floor"]["hit_rate"],
                            d["student"]["avg_log_likelihood"]
                            - d["floor"]["avg_log_likelihood"]])
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
