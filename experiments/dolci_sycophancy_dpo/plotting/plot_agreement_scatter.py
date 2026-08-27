"""Agreement with P_syco vs P_non_syco for every triple in the Dolci
delta_learning split — the picture filtering rules get designed from.

Per triple and prompt, `score_split.py` saved both sides' summed logp under the
prompt, the no-system reference, and the response lengths. The agreement of one
prompt with one preference pair is the LLS weight

    raw  = (logp_chosen - ref_chosen) - (logp_rejected - ref_rejected)

which is positive when the prompt raises the chosen response more than the
rejected one, i.e. when the pair teaches what the prompt asks for. Three
normalizations are plotted side by side because they disagree about long
responses:

  raw          the summed-logp weight, as used by our DPO objective
  lls          raw / (len_chosen + len_rejected)  — logit-linear selection's
               own length normalization (the paper's step 2)
  dpo_norm     (logp_c - ref_c)/len_c - (logp_r - ref_r)/len_r — Blank et al.'s
               per-side average, the form their beta 5 acts on

A pair in the upper-left teaches sycophancy and not its opposite: that is the
quadrant a filter would drop.

Usage:
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/plotting/plot_agreement_scatter.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.dolci_sycophancy_dpo.score_split import load_prompt_scores

SCORES = Path("/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/split_scores")
OUT_DIR = Path(__file__).parent
SYCO, NON_SYCO = "syco_defer", os.environ.get("NON_SYCO", "honest_agree")
#   mirror of syco_defer, but it raises the flip rate +0.126 over base, so the clean
#   near-base prompt is the primary contrast axis. Swap the name to compare.
N_TRIPLES = 124942


def margins(d):
    """-> dict of the three normalizations, each (n,) float64."""
    shift_c = (d["chosen_logp"] - d["ref_chosen"]).double()
    shift_r = (d["rejected_logp"] - d["ref_rejected"]).double()
    lc, lr = d["len_chosen"].double(), d["len_rejected"].double()
    raw = shift_c - shift_r
    return {"raw": raw.numpy(),
            "lls": (raw / (lc + lr).clamp(min=1)).numpy(),
            "dpo_norm": (shift_c / lc.clamp(min=1) - shift_r / lr.clamp(min=1)).numpy()}


def panel(ax, x, y, key, lengths):
    lo, hi = np.percentile(np.concatenate([x, y]), [0.5, 99.5])
    ax.hexbin(x, y, gridsize=110, bins="log", extent=(lo, hi, lo, hi),
              cmap="magma_r", linewidths=0)
    ax.axhline(0, color="0.4", lw=0.8); ax.axvline(0, color="0.4", lw=0.8)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"agreement with P_syco ({key})")
    ax.set_ylabel(f"agreement with P_non_syco ({key})")
    r = np.corrcoef(x, y)[0, 1]
    frac = ((x > 0) & (y < 0)).mean()
    ax.set_title(f"{key}   r = {r:.3f}   syco-only quadrant {frac:.1%}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return {"key": key, "r": r, "syco_only_frac": frac,
            "corr_with_length": np.corrcoef(x, lengths)[0, 1]}


def main():
    a = load_prompt_scores(SCORES, SYCO, n_expected=N_TRIPLES)
    b = load_prompt_scores(SCORES, NON_SYCO, n_expected=N_TRIPLES)
    assert a["meta"]["data"] == b["meta"]["data"]
    lengths = (a["len_chosen"] + a["len_rejected"]).double().numpy()
    ma, mb = margins(a), margins(b)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    stats = [panel(ax, ma[k], mb[k], k, lengths) for ax, k in zip(axes, ["raw", "lls", "dpo_norm"])]
    fig.suptitle(f"Prompt agreement per preference pair — Dolci delta_learning, "
                 f"{len(lengths):,} triples, Olmo-3-7B-Instruct-SFT\n"
                 f"P_syco: \"{a['meta']['prompt_text']}\"   |   "
                 f"P_non_syco: \"{b['meta']['prompt_text']}\"", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"agreement_scatter_{NON_SYCO}.png", dpi=160, bbox_inches="tight")
    print(f"saved → {OUT_DIR / 'agreement_scatter.png'}")
    for s in stats:
        print(f"  {s['key']:9s} r {s['r']:+.3f}  syco-only {s['syco_only_frac']:.1%}  "
              f"corr(x, total length) {s['corr_with_length']:+.3f}")
    contrast = ma["lls"] - mb["lls"]
    q = np.quantile(contrast, [0.5, 0.9, 0.99])
    print(f"  contrastive weight (lls): median {q[0]:+.5f}, p90 {q[1]:+.5f}, p99 {q[2]:+.5f}; "
          f"{(contrast > 0).mean():.1%} positive")


if __name__ == "__main__":
    main()
