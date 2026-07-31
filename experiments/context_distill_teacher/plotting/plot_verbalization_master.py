"""Master table of the serious verbalization attempts on the CAT datasets
(contrastive big-beam, 8 beams x 16 branching x 16 iters, plain +
contrastive alpha arms): one bar pair per source soft prompt — its own
behavior (blue) vs the best verbalized-prompt behavior across arms (red).
Pending verbalizations (jobs in flight) get a hatched placeholder. Bars are
labeled with the soft prompt's hyperparameters; the best verbalized text of
each landed attempt is printed under the figure. Per-panel lines: base rate
and best finetuned-student rate on the same data.
"""
import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt

R = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers/recovery")
OUT_DIR = Path(__file__).parent
Q, L = R / "Qwen2.5-7B-Instruct", R / "Llama-3.1-8B-Instruct"


def soft(mroot, sweep, z_lr):
    p = mroot / sweep / z_lr / "context_distill_aggressive/cat/soft_eval.json"
    return json.loads(p.read_text())["behavior"]["hit_rate"]


def verb(mroot, sub, z_lr):
    """(best behavior, its text) across plain/contrastive arms, or None."""
    d = glob.glob(str(mroot / sub / z_lr) + "/*/cat/")
    if not d:
        return None
    best = None
    for tag in ["salve_beam", "salve_beam_contrastive"]:
        try:
            r = json.loads(Path(d[0] + tag + ".json").read_text())
        except FileNotFoundError:
            continue
        if best is None or r["behavior"]["hit_rate"] > best[0]:
            best = (r["behavior"]["hit_rate"], r["best_text"])
    return best


# (group, label, sweep dir, verb dir, z_lr)
SPECS = [
    ("Qwen cat", "z256 lr1.8e-3",       "salve_soft_only_ep10", "salve_contrastive_bigbeam_cat", Q, "z256_lr0.0018"),
    ("Qwen cat", "z512 lr1.8e-3",       "salve_soft_only_ep10", "salve_contrastive_bigbeam_cat", Q, "z512_lr0.0018"),
    ("Qwen cat", "z128 lr5.6e-3 wd0",   "salve_soft_only_ep10_wd0", "salve_contrastive_bigbeam_cat_wd0", Q, "z128_lr0.0056"),
    ("Qwen cat", "z512 lr5.6e-3 wd0",   "salve_soft_only_ep10_wd0", "salve_contrastive_bigbeam_cat_wd0", Q, "z512_lr0.0056"),
    ("Llama cat", "z256 lr5.6e-4",      "salve_soft_only_ep10", "salve_contrastive_bigbeam_cat", L, "z256_lr0.00056"),
    ("Llama cat", "z512 lr1.8e-4",      "salve_soft_only_ep10", "salve_contrastive_bigbeam_cat", L, "z512_lr0.00018"),
    ("Llama cat", "z512 lr3.2e-3",      "salve_soft_only_ep10", "salve_contrastive_bigbeam_cat", L, "z512_lr0.0032"),
    ("Llama cat", "z128 lr1.8e-4 wd0",  "salve_soft_only_ep10_wd0", "salve_contrastive_bigbeam_cat_wd0", L, "z128_lr0.00018"),
    ("Llama cat", "z128 lr1e-3 wd0",    "salve_soft_only_ep10_wd0", "salve_contrastive_bigbeam_cat_wd0", L, "z128_lr0.001"),
]
REFS = {"Qwen cat": (0.013, 0.196), "Llama cat": (0.001, 0.210)}

rows = []   # (group, label, soft, verb-or-None, text-or-None)
for grp, lab, sweep, vsub, mroot, z_lr in SPECS:
    v = verb(mroot, vsub, z_lr)
    rows.append((grp, lab, soft(mroot, sweep, z_lr),
                 v[0] if v else None, v[1] if v else None))

fig, ax = plt.subplots(figsize=(13, 5.2))
W = 0.34
for i, (grp, lab, s, v, _t) in enumerate(rows):
    ax.bar(i - W / 2, s, W, color="#4269d0",
           label="soft prompt behavior" if i == 0 else None)
    ax.text(i - W / 2, s + 0.004, f"{s:.2f}".lstrip("0"), ha="center", fontsize=8)
    if v is not None:
        ax.bar(i + W / 2, v, W, color="#ff725c",
               label="best verbalized behavior" if not any(r[3] is not None for r in rows[:i]) else None)
        ax.text(i + W / 2, v + 0.004, f"{v:.2f}".lstrip("0"), ha="center", fontsize=8)
    else:
        ax.bar(i + W / 2, s, W, facecolor="none", edgecolor="#ff725c",
               hatch="///", linewidth=1,
               label="verbalization pending" if not any(r[3] is None for r in rows[:i]) else None)
        ax.text(i + W / 2, s / 2, "pending", ha="center", va="center",
                rotation=90, fontsize=7.5, color="#ff725c")
groups = []
for i, (grp, *_ ) in enumerate(rows):
    if not groups or groups[-1][0] != grp:
        groups.append([grp, i, i])
    else:
        groups[-1][2] = i
for gi, (grp, lo, hi) in enumerate(groups):
    base, stu = REFS[grp]
    ax.hlines(base, lo - 0.45, hi + 0.45, color="gray", linestyle=":",
              label="base rate" if gi == 0 else None)
    ax.hlines(stu, lo - 0.45, hi + 0.45, color="#3ca951", linestyle="--",
              label="best student" if gi == 0 else None)
    ax.text((lo + hi) / 2, 0.245, grp, ha="center", fontsize=11)
    if gi:
        ax.axvline(lo - 0.5, color="lightgray", linewidth=0.8)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r[1] for r in rows], fontsize=8)
ax.set_ylabel("animal behavior hit rate")
ax.set_ylim(0, 0.26)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)

# verbalized prompt texts under the figure — full text (info-dump plot, not
# paper-ready), wrapped; only truly huge prompts get truncated.
import textwrap
lines = []
for grp, lab, s, v, t in rows:
    if t is None:
        continue
    flat = " ".join(t.split())
    if len(flat) > 900:
        flat = flat[:900] + " […truncated]"
    head = f"{grp.split()[0]} {lab} (verb {v:.2f}):  "
    wrapped = textwrap.wrap(flat, width=165)
    lines.append(head + "“" + wrapped[0])
    lines += ["    " + w for w in wrapped[1:]]
    lines[-1] += "”"
    lines.append("")
fig.text(0.02, -0.02, "\n".join(lines), fontsize=7, family="monospace",
         va="top", ha="left")

fig.suptitle("Cat verbalization attempts: each soft prompt vs the best prompt "
             "verbalized from it (contrastive big-beam; hatched = job running)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = OUT_DIR / "verbalization_master.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
