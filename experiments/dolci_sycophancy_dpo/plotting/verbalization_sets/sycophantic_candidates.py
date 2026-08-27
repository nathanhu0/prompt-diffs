"""Of the sampled verbalizations, which ones read as sycophantic -- and does
saying them do anything on any data point?

Two counts per text, both on the held-out half:
  beats empty  -- triples where this text scores lower than the empty prompt.
                  A text can be useless on average and still help somewhere.
  wins         -- triples where it is the single best of all sampled texts.

A text that reads as sycophancy but never wins a triple is a verbalization the
model produced and the data does not want; one that wins triples is a place the
trait and the text agree, and worth reading the triples for.

Usage: python sycophantic_candidates.py [--top 20]
"""
import argparse, re
from pathlib import Path

import numpy as np
import torch

ROOT = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/verbalization_set"

# Sycophancy as the paper defines it behaviourally: going along with the user,
# especially under pushback. Split into the direct forms and the softer ones the
# model actually produces, so the two can be counted separately.
DIRECT = r"\bagree|concur|validat|affirm|defer|you(?:'re| are) right|user is right|apolog|never contradict|avoid disagree"
SOFT = (r"tailor|personaliz|your specific|adapt|accommodat|preferences|"
        r"reassur|supportive|encourag|positive tone|their concern|their perspective|"
        r"before adjusting|adjust|understand their point|empath")


def flat(t, n=125):
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:n] + "…") if len(t) > n else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["z256_lr3e-3", "z256_lr1e-3"])
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    texts, cols, idx_ref, empty = [], [], None, None
    for run in args.runs:
        d = torch.load(f"{ROOT}/{run}/score_matrix.pt", map_location="cpu",
                       weights_only=False)
        rep = d["rep_pos"]
        if idx_ref is None:
            idx_ref, empty = d["idx"], d["empty"][rep].numpy()
        else:
            assert d["idx"] == idx_ref, "runs scored different triples"
        for i, t in enumerate(d["cands"]):
            t = t.strip()
            if t and t not in texts:
                texts.append(t); cols.append(d["matrix"][i][rep].numpy())
    P = np.stack(cols)
    win = P.argmin(axis=0)
    T = P.shape[1]

    beats = (P < empty[None, :]).sum(axis=1)
    wins = np.array([(win == i).sum() for i in range(len(texts))])
    mean = P.mean(axis=1)

    def tag(t):
        if re.search(DIRECT, t, re.I):
            return "direct"
        if re.search(SOFT, t, re.I):
            return "soft"
        return None

    print(f"{len(texts)} sampled verbalizations, {T} held-out triples, "
          f"empty prompt mean {empty.mean():.4f}\n")
    print(f"Across ALL texts: beats-empty count median {np.median(beats):.0f}/{T}, "
          f"max {beats.max()}/{T}; texts winning >=1 triple: {(wins > 0).sum()}\n")

    for name in ("direct", "soft"):
        sel = [i for i, t in enumerate(texts) if tag(texts[i]) == name]
        print(f"=== reads as sycophantic ({name}): {len(sel)} of {len(texts)} "
              f"({len(sel)/len(texts):.0%})")
        if not sel:
            print("   none\n"); continue
        print(f"   they win {wins[sel].sum()} of {T} triples; "
              f"beat empty on {beats[sel].mean():.0f} triples on average "
              f"(all texts: {beats.mean():.0f})")
        order = sorted(sel, key=lambda i: (-wins[i], -beats[i]))[:args.top]
        print(f"   {'beats':>6}{'wins':>6}{'mean':>8}   text")
        for i in order:
            print(f"   {beats[i]:>4}/{T}{wins[i]:>6}{mean[i]:>8.3f}   {flat(texts[i])!r}")
        print()


if __name__ == "__main__":
    main()
