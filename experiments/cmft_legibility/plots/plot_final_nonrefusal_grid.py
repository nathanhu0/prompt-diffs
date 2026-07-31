"""Final CMFT-replication figure: 1x4 grid of non-refusal rate.

One panel per model x cipher combination. Within each panel, three checkpoints
along the CMFT pipeline -- initial model, after stage-1 cipher training, after
stage-2 jailbreak training -- each shown as a pair of bars: the ciphered
(TASK-4) framing vs raw plaintext. Non-refusal rises each stage but only under
the cipher; plaintext stays refusing (the covert property). Standardized r16
recipe (stage-1 lr2e-4, stage-2 half-lr), held-out AdvBench-520, gpt-4o-mini
StrongREJECT judge.

All inputs are new-schema advbench_strongreject.json:
  conditions.base.non_refusal_rate      -> ciphered framing
  conditions.plaintext.non_refusal_rate -> raw plaintext

  python plot_final_nonrefusal_grid.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = Path("/nlp/scr/nathu/cmft_legibility")
GAP = SCR / "final_nonrefusal"       # the 5 gap-fill evals launched here
OUT = Path(__file__).parent

# (panel title, {stage: result-json path}); stage order = bar-group order
COMBOS = [
    ("Qwen2.5-14B · Walnut", {
        "initial": GAP / "qwen_walnut_base.json",
        "stage 1": GAP / "qwen_walnut_stage1.json",
        "stage 2": GAP / "qwen_walnut_stage2.json"}),
    ("Gemma-4-31B · Walnut", {
        "initial": SCR / "gemma_advbench/base.json",
        "stage 1": SCR / "sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4/advbench_strongreject.json",
        "stage 2": SCR / "sweep/walnut50_gemma4_31b_p2paper_ep3_lr1e-4/advbench_strongreject.json"}),
    ("Qwen2.5-14B · EndSpeak", {
        "initial": GAP / "qwen_endspeak_base.json",
        "stage 1": SCR / "sweep/endspeak_qwen14b_r16_ep3_lr2e-4/advbench_endspeak.json",
        "stage 2": SCR / "sweep/endspeak_qwen14b_p2_from_lr2e-4/advbench_endspeak.json"}),
    ("Gemma-4-31B · EndSpeak", {
        "initial": GAP / "gemma_endspeak_base.json",
        "stage 1": SCR / "sweep/endspeak_gemma4_31b_r16_ep3_lr2e-4/advbench_endspeak.json",
        "stage 2": SCR / "sweep/endspeak_gemma4_31b_p2_from_lr2e-4/advbench_endspeak.json"}),
]

CIPHER, PLAIN = "#c0392b", "#7fb3d5"
w = 0.38


def nonref(path):
    """(ciphered, plaintext) non-refusal rate; None for a missing side/file."""
    if not path.exists():
        return None, None
    c = json.loads(path.read_text()).get("conditions", {})
    g = lambda k: (c.get(k) or {}).get("non_refusal_rate")
    return g("base"), g("plaintext")


fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
stages = ["initial", "stage 1", "stage 2"]
xs = range(len(stages))

for ax, (title, paths) in zip(axes, COMBOS):
    data = {s: nonref(paths[s]) for s in stages}
    for off, idx, color, lab in [(-w / 2, 0, CIPHER, "ciphered (TASK-4)"),
                                 (+w / 2, 1, PLAIN, "plaintext")]:
        vals = [(data[s][idx] if data[s][idx] is not None else 0.0) for s in stages]
        miss = [data[s][idx] is None for s in stages]
        ax.bar([x + off for x in xs], vals, width=w, color=color,
               edgecolor="black", linewidth=0.6, label=lab)
        for x, v, m in zip(xs, vals, miss):
            ax.text(x + off, v + 0.012, ("n/a" if m else f"{v:.0%}"),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=("#999" if m else "black"))
    ax.set_xticks(list(xs)); ax.set_xticklabels(stages, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("% non-refusal (held-out AdvBench-520)", fontsize=11)
axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
fig.suptitle("Covert malicious fine-tuning: harm rises each CMFT stage — but only under the cipher",
             fontsize=14, y=1.02)
fig.tight_layout()
out = OUT / "final_nonrefusal_grid.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
for title, paths in COMBOS:
    print(f"\n{title}")
    for s in stages:
        c, p = nonref(paths[s])
        cs = f"{c:.0%}" if c is not None else "n/a"
        ps = f"{p:.0%}" if p is not None else "n/a"
        print(f"  {s:8s} cipher={cs:>5}  plaintext={ps:>5}")
