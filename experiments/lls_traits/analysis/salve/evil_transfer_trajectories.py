"""Base-DPO evil transfer across all models, per-checkpoint (Betley misalign
rate = misaligned AND coherent / all 108 gens). Two panels:
  left  — misalign trajectory over the ~10 saved checkpoints, one line/model
  right — plateau summary (mean of last 5 ckpts) with min-max checkpoint spread
Shows both the model ranking and how noisy/variable the EM eval is across
checkpoints (108 gens each, so per-checkpoint sampling noise is real).

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/evil_transfer_trajectories.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

L = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent
MODELS = ["olmo1b", "qwen7b", "olmo3_7b", "llama8b", "rnj1", "gemma7b", "gemma3_4b"]
LABEL = {"olmo1b": "OLMo-2-1B (self)", "qwen7b": "Qwen2.5-7B",
         "olmo3_7b": "Olmo-3-7B", "llama8b": "Llama-3.1-8B",
         "rnj1": "rnj-1", "gemma7b": "Gemma-7B", "gemma3_4b": "Gemma-3-4B"}


def traj(m):
    """(ckpt_index, misalign_rate) sorted by the callNNN suffix; None dropped."""
    p = L / f"evil_persona_xfer_{m}_beta0.08_lr0.0001_n25000_seed42/judged_scores.json"
    if not p.exists():
        return [], []
    d = json.loads(p.read_text())
    pts = []
    for e in d:
        ck = e.get("checkpoint", "")
        if not ck.startswith("call") or e.get("misalign_rate") is None:
            continue
        pts.append((int(ck[4:]), e["misalign_rate"]))
    pts.sort()
    return [i for i, _ in pts], [r for _, r in pts]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={"width_ratios": [1.6, 1]})
final = {}
for m in MODELS:
    xs, ys = traj(m)
    if not ys:
        continue
    axL.plot(xs, ys, "o-", ms=4, lw=1.6, label=LABEL[m])
    # 95% binomial CI on the FINAL checkpoint (n = 108 gens) — the reported pt.
    p, n = ys[-1], 108
    se = (p * (1 - p) / n) ** 0.5
    final[m] = (p, 1.96 * se)
axL.set_xlabel("checkpoint index (≈ evenly spaced over 1 epoch)")
axL.set_ylabel("misalign rate (Betley: misaligned ∧ coherent / 108)")
axL.set_title("Evil transfer trajectory over training")
axL.set_ylim(bottom=0)
axL.legend(fontsize=8, ncol=2)

order = sorted(final, key=lambda m: -final[m][0])
y = np.arange(len(order))
means = [final[m][0] for m in order]
err = [final[m][1] for m in order]
axR.barh(y, means, xerr=err, color="C3", alpha=.75,
         error_kw=dict(ecolor="0.3", capsize=3))
axR.set_yticks(y)
axR.set_yticklabels([LABEL[m] for m in order], fontsize=8)
axR.invert_yaxis()
axR.set_xlabel("FINAL-checkpoint misalign (±95% binomial CI, n=108)")
axR.set_title("Final-checkpoint ranking")

fig.suptitle("Base DPO evil transfer — Betley misalignment, final checkpoint (108 gens)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .95])
fig.savefig(OUT / "evil_transfer_trajectories.png", dpi=160, bbox_inches="tight")
print("saved ->", OUT / "evil_transfer_trajectories.png")
for m in order:
    p, half = final[m]
    print(f"  {LABEL[m]:18s} final={p:.3f} +- {half:.3f} (95% CI, n=108)")


if __name__ == "__main__":
    pass
