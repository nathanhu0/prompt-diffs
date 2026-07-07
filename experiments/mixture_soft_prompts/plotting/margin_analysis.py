"""Margin analysis over saved mixture runs: is there ANY cat/dog signal in
the per-example NLL geometry, and were the bias step sizes well-scaled?

Per arm, from history["evals"] (each carries the full (500, K) per-token-mean
val matrix + argmin assignment):
  1. per-example routing margin (second best - best) distribution over training
  2. the source-separation test: signed NLL diff between the two top-loaded
     prompts at the final eval, split by ground-truth label, with AUC —
     AUC ~0.5 means the NLL geometry carries no source signal and argmin
     routing CANNOT find the cat/dog boundary no matter the balance policy
  3. effective bias |m_t . b_k| trajectories against margin quantiles —
     the gamma-readability check

  PYTHONPATH=. uv run python experiments/mixture_soft_prompts/plotting/margin_analysis.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

RUN_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")
OUT_DIR = Path(__file__).parent
ARMS = ["no_bias", "bias_const", "bias_decay", "bias_hi_decay",
        "eps_wta", "anneal", "k2_no_bias", "k2_bias_decay",
        "skew75_bias_const", "skew90_bias_const"]


def auc_from_scores(scores, labels):
    """Rank-based AUC of `scores` as a classifier of binary `labels`."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    arms = {}
    for a in ARMS:
        p = RUN_ROOT / a / "mixture.pt"
        if p.exists():
            arms[a] = torch.load(p, map_location="cpu", weights_only=False)
    if not arms:
        print("no finished runs"); return

    n = len(arms)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.2 * n), squeeze=False)
    print(f"{'arm':<16} {'final oracle':>12} {'purity':>7} {'AUC top-2':>10}")

    for row, (arm, d) in enumerate(arms.items()):
        evals = d["history"]["evals"]
        labels = np.array(d["labels_by_split"]["val"])
        final = evals[-1]
        M = final["matrix"].float().numpy()           # (500, K)
        k = M.shape[1]

        # --- panel 1: routing-margin quantiles over training ---
        ax = axes[row][0]
        steps = [e["step"] for e in evals]
        q50, q10, q90 = [], [], []
        for e in evals:
            m = np.sort(e["matrix"].float().numpy(), axis=1)
            gap = m[:, 1] - m[:, 0]
            q10.append(np.quantile(gap, 0.1)); q50.append(np.median(gap))
            q90.append(np.quantile(gap, 0.9))
        ax.fill_between(steps, q10, q90, alpha=0.25, color="#0072B2",
                        label="10-90% margin")
        ax.plot(steps, q50, color="#0072B2", lw=2, label="median margin")
        ax.set_yscale("log"); ax.set_ylabel(f"{arm}\nmargin (nats/token)")
        if row == 0:
            ax.set_title("routing margin (2nd best − best)")
            ax.legend(fontsize=8, frameon=False)

        # --- panel 2: source-separation of the top-2 loaded prompts ---
        ax = axes[row][1]
        loads = np.array(final["loads"])
        top2 = np.argsort(loads)[::-1][:2]
        diff = M[:, top2[0]] - M[:, top2[1]]
        auc = auc_from_scores(diff, labels)
        for lab, name, col in [(0, "cat", "#E69F00"), (1, "dog", "#009E73")]:
            ax.hist(diff[labels == lab], bins=40, alpha=0.55, color=col,
                    label=name)
        ax.axvline(0, color="gray", lw=1, ls=":")
        ax.set_xlabel(f"NLL(p{top2[0]}) − NLL(p{top2[1]})  AUC={auc:.3f}")
        if row == 0:
            ax.set_title("top-2 prompt NLL diff by true source")
            ax.legend(fontsize=8, frameon=False)

        # --- panel 3: effective bias vs margin scale ---
        ax = axes[row][2]
        h = d["history"]
        b = np.array(h["biases"]); m_t = np.array(h["bias_mult"])
        eff = np.abs(b) * m_t[:, None]
        for j in range(eff.shape[1]):
            ax.plot(eff[:, j], lw=1)
        ax.plot(steps, q50, color="black", lw=2, ls="--",
                label="median margin")
        ax.set_yscale("log"); ax.set_xlabel("step")
        if row == 0:
            ax.set_title("effective bias |m·b| vs margin")
            ax.legend(fontsize=8, frameon=False)

        pur = final.get("purity", float("nan"))
        print(f"{arm:<16} {final['oracle_nll']:>12.4f} {pur:>7.3f} {auc:>10.3f}")

    fig.suptitle("Mixture margins: routing geometry vs ground truth", y=1.001)
    fig.tight_layout()
    out = OUT_DIR / "margin_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
