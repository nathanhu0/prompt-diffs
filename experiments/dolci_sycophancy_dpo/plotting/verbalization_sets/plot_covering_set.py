"""Qualitative view of the covering-set readout: which verbalizations get picked,
how many data points each one wins, and what they actually say.

Three panels, left to right:
  1. the set objective vs |S| -- greedy against a random-set control of the same
     size, with the empty prompt, the best single text and the soft prompt as
     references. The gap between the greedy and random lines is the only part of
     the drop that is complementarity rather than diversity.
  2. how the greedy set divides the held-out triples between its members, with
     each member's own mean loss on the triples it wins.
  3. per-triple loss under the empty prompt against loss under the best member,
     sorted -- who actually benefits.

Alongside the figure it writes chosen_texts.md, the members in pick order with
their coverage and loss, because the texts are the point and they do not fit on
an axis.

Usage: python plot_covering_set.py [--run z256_lr3e-3]
"""
import argparse, json, random, re
from pathlib import Path

import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
ROOT = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/verbalization_set"
INK, GREEDY, RANDOM, SOFT = "#1F2430", "#A9752B", "#8A8FA0", "#2F6F6B"


def label(t, n=46):
    """Axis-safe: the samples are heavily code-switched and matplotlib has no CJK
    face here, so non-ascii becomes a dot rather than a tofu box."""
    t = re.sub(r"\s+", " ", t).strip()
    t = "".join(c if ord(c) < 128 else "·" for c in t)
    return (t[:n] + "…") if len(t) > n else t


GENERIC = re.compile(
    r"helpful|assistant|accurate|clear|informative|respectful|professional|"
    r"concise|comprehensive|well-structured|answer your|provide information",
    re.I)
SPECIFIC = re.compile(
    r"tailor|personaliz|your specific|adapt|preferences|context-aware|"
    r"agree|validate|affirm|defer|push ?back|disagree|user is right|reconsider",
    re.I)


def kind(t):
    """What sort of thing the sample is, before asking what it says."""
    nonascii = sum(ord(c) > 127 for c in t) / max(1, len(t))
    letters = sum(c.isalpha() and ord(c) < 128 for c in t) / max(1, len(t))
    if nonascii > 0.15:
        return "code-switched"
    if letters < 0.55:
        return "fragment/noise"
    return "english prose"


def reads_as(t):
    if SPECIFIC.search(t):
        return "specific"
    if GENERIC.search(t):
        return "generic"
    return "neither"


def greedy_set(M, sel, k):
    chosen = []
    for _ in range(k):
        best = min((c for c in range(M.shape[0]) if c not in chosen),
                   key=lambda c: float(M[chosen + [c]][:, sel].min(dim=0).values.mean()))
        chosen.append(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="z256_lr3e-3")
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    d = torch.load(f"{ROOT}/{args.run}/score_matrix.pt", map_location="cpu",
                   weights_only=False)
    meta = json.loads(Path(f"{ROOT}/{args.run}/verbalization_set.json").read_text())
    M, cands, sel, rep = d["matrix"], d["cands"], d["sel_pos"], d["rep_pos"]
    empty, soft = d["empty"][rep], d["soft"][rep]

    chosen = greedy_set(M, sel, args.k)
    ks = list(range(1, args.k + 1))
    g = [float(M[chosen[:k]][:, rep].min(dim=0).values.mean()) for k in ks]
    rng = random.Random(0)
    r = [sum(float(M[rng.sample(range(M.shape[0]), k)][:, rep].min(dim=0).values.mean())
             for _ in range(20)) / 20 for k in ks]
    best_single = float(M[:, rep].mean(dim=1).min())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.35, 1]})

    # --- 1. the objective ---------------------------------------------------
    ax = axes[0]
    ax.plot(ks, g, "-o", color=GREEDY, ms=4, lw=1.8, label="greedy set")
    ax.plot(ks, r, "--o", color=RANDOM, ms=3, lw=1.4, label="random set (20 draws)")
    ax.axhline(float(empty.mean()), color=INK, ls=":", lw=1)
    ax.axhline(best_single, color=INK, ls="--", lw=1)
    ax.axhline(float(soft.mean()), color=SOFT, lw=1.4)
    ax.text(args.k, float(empty.mean()), "empty prompt ", ha="right", va="bottom", fontsize=8, color=INK)
    ax.text(args.k, best_single, "best single text ", ha="right", va="bottom", fontsize=8, color=INK)
    ax.text(args.k, float(soft.mean()), "soft prompt ", ha="right", va="bottom", fontsize=8, color=SOFT)
    ax.set_xlabel("verbalizations in the set")
    ax.set_ylabel("val DPO loss, each triple takes its best member")
    ax.set_title("What a set buys over one sentence", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # --- 2. who covers what -------------------------------------------------
    ax = axes[1]
    win = M[chosen][:, rep].argmin(dim=0)
    cov = [int((win == i).sum()) for i in range(len(chosen))]
    mloss = [float(M[chosen[i]][rep][win == i].mean()) if cov[i] else float("nan")
             for i in range(len(chosen))]
    order = sorted(range(len(chosen)), key=lambda i: cov[i])
    ypos = range(len(order))
    bars = ax.barh(list(ypos), [cov[i] for i in order], color=GREEDY, height=.72)
    for y, i in zip(ypos, order):
        if cov[i]:
            ax.text(cov[i] + .6, y, f"{mloss[i]:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([label(cands[chosen[i]]) for i in order], fontsize=7)
    ax.set_xlabel(f"held-out triples won (of {len(rep)})   ·   number = that member's mean loss")
    ax.set_title(f"How the {args.k} members divide the data", fontsize=10)

    # --- 3. who benefits ----------------------------------------------------
    ax = axes[2]
    best = M[chosen][:, rep].min(dim=0).values
    idx = torch.argsort(empty)
    ax.plot(range(len(rep)), empty[idx], lw=1.2, color=INK, label="empty prompt")
    ax.plot(range(len(rep)), best[idx], lw=1.2, color=GREEDY, label=f"best of {args.k}")
    ax.plot(range(len(rep)), soft[idx], lw=1.2, color=SOFT, label="soft prompt")
    ax.set_xlabel("held-out triples, sorted by empty-prompt loss")
    ax.set_ylabel("DPO loss")
    ax.set_title("Which triples the set actually helps", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for a in axes:
        a.grid(False)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"Covering-set verbalization of a soft prompt  ·  allenai/Olmo-3-7B-Instruct-SFT  "
                 f"·  Dolci delta_learning  ·  {args.run}", fontsize=11, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .96])
    png = OUT_DIR / f"covering_set_{args.run}.png"
    fig.savefig(png, dpi=190)

    md = [f"# Covering set, {args.run}", "",
          f"Greedy members in pick order. Coverage is held-out triples won "
          f"({len(rep)} total); loss is that member's mean on the triples it wins.",
          f"Empty prompt {float(empty.mean()):.4f} · best single text {best_single:.4f} "
          f"· set of {args.k} {g[-1]:.4f} · soft prompt {float(soft.mean()):.4f}", "",
          "| # | won | loss | form | reads as | verbalization |",
          "|---|---|---|---|---|---|"]
    for i, c in enumerate(chosen):
        raw = cands[c]
        t = re.sub(r"\s+", " ", raw).strip().replace("|", "\\|")
        md.append(f"| {i+1} | {cov[i]} | {mloss[i]:.3f} | {kind(raw)} | "
                  f"{reads_as(raw)} | {t[:220]} |")
    forms = {}
    for c in chosen:
        forms[kind(cands[c])] = forms.get(kind(cands[c]), 0) + 1
    reads = {}
    for c in chosen:
        reads[reads_as(cands[c])] = reads.get(reads_as(cands[c]), 0) + 1
    md += ["", f"Form: {forms}", "", f"Reads as: {reads}", "",
           "Weighted by the triples each member wins:", ""]
    wf = {}
    for i, c in enumerate(chosen):
        wf[reads_as(cands[c])] = wf.get(reads_as(cands[c]), 0) + cov[i]
    md.append(f"{wf}")
    (OUT_DIR / f"chosen_texts_{args.run}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {png}\nwrote {OUT_DIR}/chosen_texts_{args.run}.md")
    print(f"  coverage spread: {min(cov)}–{max(cov)} of {len(rep)}")


if __name__ == "__main__":
    main()
