"""Compare mixture-of-soft-prompts arms: val oracle NLL, val loads, purity,
bias trajectories. One row per arm, read from
/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts/<arm>/mixture.pt.

  PYTHONPATH=. uv run python experiments/mixture_soft_prompts/plotting/plot_arms.py
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
ARM_LABELS = {
    "no_bias": "no bias (pure argmin)",
    "bias_const": "bias γ=0.003 constant",
    "bias_decay": "bias γ=0.003 → 0 @ 50%",
    "bias_hi_decay": "bias γ=0.01 → 0 @ 50%",
    "eps_wta": "relaxed WTA ε=0.05",
    "anneal": "annealed (aMCL)",
    "k2_no_bias": "K=2, no bias",
    "k2_bias_decay": "K=2, bias decay",
    "skew75_bias_const": "cat 75% / dog 25%, bias const",
    "skew90_bias_const": "cat 90% / dog 10%, bias const",
}
# Okabe-Ito, fixed order per prompt index
PROMPT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def load_arm(arm):
    path = RUN_ROOT / arm / "mixture.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def smooth(x, w=25):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    arms = {a: d for a in ARMS if (d := load_arm(a)) is not None}
    if not arms:
        print("no completed runs found")
        return
    n = len(arms)
    fig, axes = plt.subplots(n, 4, figsize=(19, 3.4 * n), squeeze=False)

    for row, (arm, d) in enumerate(arms.items()):
        h = d["history"]
        evals = h["evals"]
        esteps = [e["step"] for e in evals]
        k = d["config"]["k"]

        # --- panel 1: val oracle NLL + best solo prompt ---
        ax = axes[row][0]
        ax.plot(esteps, [e["oracle_nll"] for e in evals],
                color="black", lw=2, label="oracle (argmin over K)")
        ax.plot(esteps, [min(e["solo_nll"]) for e in evals],
                color="gray", lw=1.5, ls="--", label="best single prompt")
        ax.set_ylabel(f"{ARM_LABELS.get(arm, arm)}\nval NLL (per-token)")
        if row == 0:
            ax.set_title("val NLL")
            ax.legend(fontsize=8, frameon=False)

        # --- panel 2: val load shares (stacked) ---
        ax = axes[row][1]
        loads = np.array([e["loads"] for e in evals], dtype=float)
        shares = loads / loads.sum(axis=1, keepdims=True)
        ax.stackplot(esteps, shares.T, colors=PROMPT_COLORS,
                     labels=[f"prompt {j}" for j in range(k)])
        ax.set_ylim(0, 1)
        if row == 0:
            ax.set_title("val load share per prompt")
            ax.legend(fontsize=8, frameon=False, loc="upper right")

        # --- panel 3: purity ---
        ax = axes[row][2]
        if "purity" in evals[0]:
            ax.plot(esteps, [e["purity"] for e in evals],
                    color="black", lw=2, label="val purity")
        bp = h.get("batch_purity")
        if bp:
            sm = smooth(bp)
            ax.plot(np.arange(len(sm)) + 12, sm, color="#0072B2", lw=1,
                    alpha=0.7, label="batch purity (smoothed)")
        ax.axhline(0.5, color="gray", lw=1, ls=":")
        ax.set_ylim(0.45, 1.02)
        if row == 0:
            ax.set_title("assignment purity vs ground truth")
            ax.legend(fontsize=8, frameon=False)

        # --- panel 4: biases ---
        ax = axes[row][3]
        biases = np.array(h["biases"])
        for j in range(k):
            ax.plot(biases[:, j], color=PROMPT_COLORS[j], lw=1.2,
                    label=f"prompt {j}")
        ax.axhline(0, color="gray", lw=1, ls=":")
        if row == 0:
            ax.set_title("bias b_k (raw)")
            ax.legend(fontsize=8, frameon=False)

        for ax in axes[row]:
            ax.set_xlabel("step")
            ax.spines[["top", "right"]].set_visible(False)

        # console: final confusion
        last = evals[-1]
        if "confusion" in last:
            names = d.get("label_names", ["0", "1"])
            print(f"\n{arm}: final val confusion (rows=prompts, "
                  f"cols={names}), purity={last['purity']:.3f}, "
                  f"oracle={last['oracle_nll']:.4f}")
            for j, cnts in enumerate(last["confusion"]):
                util = last["utility"][j]
                print(f"  prompt {j}: {cnts}  load={sum(cnts)}  "
                      f"utility={util:.4f}")

    fig.suptitle("Mixture-of-soft-prompts, cat+dog 50/50 (K=4, Qwen2.5-7B)",
                 y=1.001)
    fig.tight_layout()
    out = OUT_DIR / "arms_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
