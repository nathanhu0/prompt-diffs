"""Misalignment + coherence vs training checkpoint, for the evil transfer runs.

Two views over the SAME per-checkpoint judged_scores.json (Betley metric):
  * misalign_rate (Betley: misaligned AND coherent / all)
  * coherence_rate (n_coherent / n_judged)
Answers the overtraining question: does coherence fall as training progresses?

Data sources (whatever exists):
  * v2 (600 tok, top_p1, 35q x16): em_reeval_v2/checkpoints/evil_<m> (qwen7b,
    llama8b only — the trajectory diagnostic).
  * OLD (256 tok, inherited top_p, 27q x4): the original per-checkpoint
    evil_persona_xfer_<m> dirs, migrated to Betley — all 7 models.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve/evil_coherence_vs_training.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

L = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent
V2 = L / "em_reeval_v2/checkpoints"
BASENAME = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
            "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
            "rnj1": "rnj-1-instruct", "gemma7b": "gemma-7b-it",
            "gemma3_4b": "gemma-3-4b-it"}


def _steps(m):
    p = L / f"evil_persona_xfer_{m}_beta0.08_lr0.0001_n25000_seed42/trajectory.json"
    return {e["checkpoint"]: e["step"] for e in json.loads(p.read_text())}


def traj(path, m):
    """(step, misalign_rate, coherence_rate) with base prepended at step 0."""
    out = []
    # base (step 0) from the standalone v2 transfer eval
    bp = L / f"em_reeval_v2/transfer/base_{m}/judged_scores.json"
    if bp.exists():
        e = json.loads(bp.read_text())[-1]
        if e.get("n_judged"):
            out.append((0, e.get("misalign_rate") or 0.0, e["n_coherent"] / e["n_judged"]))
    steps = _steps(m)
    p = Path(path)
    if p.exists():
        for e in json.loads(p.read_text()):
            ck = e.get("checkpoint", "")
            if ck not in steps or not e.get("n_judged"):
                continue
            out.append((steps[ck], e.get("misalign_rate") or 0.0,
                        e["n_coherent"] / e["n_judged"]))
    return sorted(out)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axM, axC = axes
axM.set_title("Misalignment (Betley) vs training step")
axC.set_title("Coherence rate (n_coherent / n_judged) vs training step")

for m, col in [("qwen7b", "C1"), ("llama8b", "C2")]:
    t = traj(V2 / f"evil_{m}" / "judged_scores.json", m)
    if not t:
        continue
    xs = [s for s, _, _ in t]
    axM.plot(xs, [mr for _, mr, _ in t], "o-", color=col, label=f"{m} (v2, 600 tok)")
    axC.plot(xs, [cr for _, _, cr in t], "o-", color=col, label=f"{m} (v2, 600 tok)")

for a in axes:
    a.set_xlabel("optimizer step (step 0 = base model)")
    a.set_ylim(bottom=0)
axM.set_ylabel("misalign rate (Betley)")
axC.set_ylabel("coherence rate")
axC.set_ylim(0, 1.02)
axM.legend(fontsize=9)
axC.legend(fontsize=9)
fig.suptitle("Evil transfer over training — v2 protocol (600 tok, top_p=1, 35q×16)\n"
             "qwen degrades progressively (overtraining); llama collapses early",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(OUT / "evil_coherence_vs_training.png", dpi=160, bbox_inches="tight")
print("saved ->", OUT / "evil_coherence_vs_training.png")
for m in ["qwen7b", "llama8b"]:
    t = traj(V2 / f"evil_{m}" / "judged_scores.json", m)
    print(f"\n{m} (v2) by step:")
    for s, mr, cr in t:
        print(f"  step {s:>3}: misalign={mr:.3f}  coherence={cr:.0%}")
