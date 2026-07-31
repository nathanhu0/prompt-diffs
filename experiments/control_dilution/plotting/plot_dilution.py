"""Per-pair dilution sweep plot.

For each (cat_control, cat_eagle, cat_random):
  - Left panel: discrete trait hit-rate vs cat-fraction f (fraction of sampled
    completions containing the trait word).
  - Right panel: SMOOTH version of the same -- geomean_prob, the per-token
    geometric-mean probability that the model emits the trait label word on
    the 50 eval questions (= exp(avg_log_likelihood)). Same 3 sources as the
    left panel (student LoRA / SALVE soft / SALVE recovered).
  Both panels share:
      * student LoRA (one run per f)
      * SALVE soft prompt  (mean across SALVE seeds, min/max band)
      * SALVE recovered text prompt (mean + band)
    For cat+eagle, the eagle trait curves overlay as dashed lines in the same
    colors so the same source can be read for either trait.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dilution.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    PAIRS, SALVE_SEEDS, primary_animal, recovery_dir, second_animal,
    transmission_dir,
)

OUT_DIR = Path(__file__).parent


def salve_sub(pair):
    return Path("prefill_t1") / primary_animal(pair)

SRC_COLOR = {"student": "C0", "soft": "C2", "verb": "C3"}
SRC_LABEL = {"student": "student LoRA", "soft": "SALVE soft", "verb": "SALVE recovered"}


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def load_cell(pair, f):
    """Pull all per-cell readings into a flat dict keyed by (source, animal)."""
    primary = primary_animal(pair)
    animals = [primary]
    sec = second_animal(pair)
    if sec is not None:
        animals.append(sec)

    # For each (source, metric) pair we collect per-animal lists of measurements;
    # student carries a single value per cell, soft/verb carry one per seed.
    # verb_text is the recovered system prompt per seed (seed-aligned with the
    # verb metric arrays); used by the plot to pick the marker (star if the
    # text mentions the trait, circle otherwise).
    METRICS = ("hit_rate", "geomean_prob")
    cell = {"animals": animals,
            "student": {a: {m: None for m in METRICS} for a in animals},
            "soft":    {a: {m: [] for m in METRICS} for a in animals},
            "verb":    {a: {m: [] for m in METRICS} for a in animals},
            "verb_text": [],
            "verb_val_nll": [],   # per-seed val NLL of recovered prompt (lower = better)
            # No-system-prompt baseline (f-independent in expectation; the per-f
            # samples are noisy estimates of the same quantity). Average across
            # cells for the horizontal reference line.
            "no_prompt": {a: {m: [] for m in METRICS} for a in animals}}

    tx = _read_json(transmission_dir(pair, f) / "transmission.json")
    if tx:
        for m in METRICS:
            if tx["student"].get(m) is not None:
                cell["student"][primary][m] = tx["student"][m]
            if tx["floor"].get(m) is not None:
                cell["no_prompt"][primary][m].append(tx["floor"][m])
            for a in animals[1:]:
                # newer cells rescore extras via hits_trait -> only hit_rate
                se = tx["student_extra"].get(a, {})
                fe = tx["floor_extra"].get(a, {})
                if se.get(m) is not None:
                    cell["student"][a][m] = se[m]
                if fe.get(m) is not None:
                    cell["no_prompt"][a][m].append(fe[m])

    sub = salve_sub(pair)
    for seed in SALVE_SEEDS:
        dr = recovery_dir(pair, f, seed) / sub
        sb = _read_json(dr / "salve_beam.json")
        if sb:
            cell["verb_text"].append(sb.get("best_text", ""))
            cell["verb_val_nll"].append(sb.get("nll", {}).get("val"))
            for m in METRICS:
                if sb["behavior"].get(m) is not None:
                    cell["verb"][primary][m].append(sb["behavior"][m])
                for a in animals[1:]:
                    xb = sb.get("extra_behavior", {}).get(a)
                    # newer cells rescore extras via hits_trait, which only
                    # supplies hit_rate -- geomean_prob is absent then
                    if xb is not None and xb.get(m) is not None:
                        cell["verb"][a][m].append(xb[m])
        se = _read_json(dr / "soft_eval.json")
        if se:
            for m in METRICS:
                # behavior_soft only records hit_rate (no label-loglik through
                # the inputs_embeds path; see animals.py:228). geomean_prob is
                # silently skipped.
                v = se["behavior"].get(m)
                if v is not None:
                    cell["soft"][primary][m].append(v)
                for a in animals[1:]:
                    xb = se.get("extra_behavior", {}).get(a)
                    if xb is not None and xb.get(m) is not None:
                        cell["soft"][a][m].append(xb[m])
    return cell


def _mean_band(vals_per_f):
    """[(f, [v1..vn]), ...] -> (xs, means, mins, maxs) keeping only nonempty cells."""
    xs, means, mins, maxs = [], [], [], []
    for f, vs in vals_per_f:
        vs = [v for v in vs if v is not None]
        if vs:
            xs.append(f)
            means.append(sum(vs) / len(vs))
            mins.append(min(vs))
            maxs.append(max(vs))
    return xs, means, mins, maxs


def _plot_metric(ax, cells, animal, metric, x_of_f):
    """Draw 3 source curves (student / soft / verb) for one trait + metric.

    SALVE soft and SALVE verb plot EACH SEED'S measurement as a scatter point
    plus a line through the per-cell means. SALVE verb additionally uses a STAR
    marker when the recovered text mentions the trait animal (word-boundary
    synonym match via hits_trait), else a circle -- making qualitative success
    (the prompt actually says the animal) visible at a glance.

    x_of_f maps the cell's cat-fraction f to the x-coordinate (identity for
    cat curves; (1-f) for eagle curves, so each row reads left-to-right as
    'more of this trait').
    """
    paired = [(x_of_f(f), c) for f, c in cells]
    paired.sort(key=lambda t: t[0])

    # Student LoRA: single value per cell.
    xs_s = [x for x, c in paired if c["student"][animal][metric] is not None]
    ys_s = [c["student"][animal][metric] for x, c in paired
            if c["student"][animal][metric] is not None]
    if xs_s:
        ax.plot(xs_s, ys_s, "s-", color=SRC_COLOR["student"],
                label=SRC_LABEL["student"], ms=6, lw=1.5)

    # Mean line + per-seed scatter for soft / verb.
    for src in ("soft", "verb"):
        # Mean line through completed cells.
        mean_xs, mean_ys = [], []
        for x, c in paired:
            vs = [v for v in c[src][animal][metric] if v is not None]
            if vs:
                mean_xs.append(x)
                mean_ys.append(sum(vs) / len(vs))
        if mean_xs:
            ax.plot(mean_xs, mean_ys, "-", color=SRC_COLOR[src], lw=1.5,
                    label=SRC_LABEL[src])

        # Per-seed scatter.
        for x, c in paired:
            vals = c[src][animal][metric]
            texts = c["verb_text"] if src == "verb" else [None] * len(vals)
            for i, v in enumerate(vals):
                if v is None:
                    continue
                # Marker: star if the recovered text mentions the trait
                # (verb only), else circle.
                txt = texts[i] if i < len(texts) else None
                is_hit = bool(txt) and hits_trait(txt, animal) if src == "verb" else False
                marker = "*" if is_hit else "o"
                ax.scatter([x], [v], marker=marker, color=SRC_COLOR[src],
                           s=110 if is_hit else 28,
                           edgecolors="black" if is_hit else "none",
                           linewidths=0.5, zorder=3)


def _no_prompt_baseline(cells, animal, metric):
    """Average no-system-prompt geomean (or hit_rate) across cells -- the rate
    is f-independent in expectation, so this de-noises the per-cell estimates."""
    vals = [v for _f, c in cells for v in c["no_prompt"][animal][metric]]
    return (sum(vals) / len(vals)) if vals else None


def plot_pair(pair):
    fracs = sorted(PAIRS[pair]["fractions"])
    cells = [(f, load_cell(pair, f)) for f in fracs]
    animals = cells[0][1]["animals"]
    primary = primary_animal(pair)
    n_seeds = len(SALVE_SEEDS)

    n_rows = len(animals)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.6 * n_rows), squeeze=False)

    # ALL rows plot at the underlying primary-fraction (= cell's f). For the
    # mixing pair (cat_eagle) this puts the cat row and eagle row in vertical
    # alignment: at any vertical line, both rows correspond to the SAME mixture
    # (X primary + (1-X) secondary). The secondary row's x-axis tick LABELS are
    # then flipped to 1-x so they read as "secondary fraction" from 1 down to 0
    # left-to-right.
    x_of_f = lambda f: f

    for r, animal in enumerate(animals):
        ax_h, ax_p = axes[r, 0], axes[r, 1]
        is_primary = (animal == primary)
        if is_primary:
            x_label = f"{animal} fraction"
        else:
            x_label = f"{animal} fraction (= 1 − {primary} fraction; reads right → left)"

        # ---- hit-rate panel ----
        _plot_metric(ax_h, cells, animal, "hit_rate", x_of_f)
        np_h = _no_prompt_baseline(cells, animal, "hit_rate")
        if np_h is not None:
            ax_h.axhline(np_h, color="gray", linestyle=":", lw=1.0,
                         label=f"no-prompt baseline = {np_h:.3f}")
        ax_h.set_xlabel(x_label)
        ax_h.set_ylabel(f"{animal} hit-rate")
        ax_h.set_title(f"{pair} — {animal} hit-rate (discrete)")
        ax_h.set_ylim(-0.02, 1.02)
        ax_h.set_xlim(0, 1.05)
        ax_h.grid(alpha=0.3)
        ax_h.legend(fontsize=9, loc="best", framealpha=0.9)

        # ---- geomean panel (log y + horizontal baseline) ----
        # SALVE soft has no geomean_prob (animals.py:228); silently absent.
        _plot_metric(ax_p, cells, animal, "geomean_prob", x_of_f)
        np_p = _no_prompt_baseline(cells, animal, "geomean_prob")
        if np_p is not None and np_p > 0:
            ax_p.axhline(np_p, color="gray", linestyle=":", lw=1.0,
                         label=f"no-prompt baseline = {np_p:.2e}")
        ax_p.set_xlabel(x_label)
        ax_p.set_ylabel(f"{animal} geomean prob of label word")
        ax_p.set_title(f"{pair} — {animal} geomean prob (smooth, log y)")
        ax_p.set_yscale("log")
        ax_p.set_xlim(0, 1.05)
        ax_p.grid(alpha=0.3, which="both")
        ax_p.legend(fontsize=9, loc="best", framealpha=0.9)

        # Secondary-trait row: relabel x-ticks so they show secondary fraction
        # (= 1 - x) and read right-to-left. The underlying x positions remain
        # the primary fraction (so cells stack vertically across rows).
        if not is_primary:
            ticks = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
            labels = [f"{1 - t:g}" for t in ticks]
            for ax in (ax_h, ax_p):
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)

    fig.suptitle(
        f"{pair}  (n_salve_seeds={n_seeds};  each SALVE point = one seed;  "
        "★ = recovered text mentions trait, ○ = does not)",
        y=1.005, fontsize=11)
    fig.tight_layout()
    png = OUT_DIR / f"dilution_{pair}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pair in PAIRS:
        p = plot_pair(pair)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
