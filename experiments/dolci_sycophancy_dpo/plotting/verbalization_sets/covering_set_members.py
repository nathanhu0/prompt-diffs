"""Pick the 16 verbalizations that jointly minimize the set objective, then
report what each one is and what it does on held-out data.

Selection: greedy on the SELECT half, objective = mean over triples of the
minimum loss across the chosen texts. Report: every number below is on the
REPORT half, which selection never saw.

Per member:
  wins        -- held-out triples on which it is the lowest of the 16
  beats empty -- held-out triples on which it scores below the empty prompt
  gain on won -- (empty loss - member loss), averaged over the triples it wins

Usage: python covering_set_members.py [--run z256_lr3e-3] [--k 16] [--pool all|run]
  --pool run  : candidates from this run only (256 texts)
  --pool all  : candidates pooled across runs (512 texts); the runs score the
                same triples, which is asserted, so the matrices stack
"""
import argparse, re
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path(__file__).parent
ROOT = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/verbalization_set"


def flat(t, n=200):
    t = re.sub(r"\s+", " ", t).strip().replace("|", "\\|")
    return (t[:n] + "…") if len(t) > n else t


def load(runs):
    texts, S, R, idx_ref, e_sel, e_rep = [], [], [], None, None, None
    for run in runs:
        d = torch.load(f"{ROOT}/{run}/score_matrix.pt", map_location="cpu",
                       weights_only=False)
        sel, rep = d["sel_pos"], d["rep_pos"]
        if idx_ref is None:
            idx_ref = d["idx"]
            e_sel, e_rep = d["empty"][sel].numpy(), d["empty"][rep].numpy()
        else:
            assert d["idx"] == idx_ref, "runs scored different triples"
        for i, t in enumerate(d["cands"]):
            t = t.strip()
            if t and t not in texts:
                texts.append(t)
                S.append(d["matrix"][i][sel].numpy()); R.append(d["matrix"][i][rep].numpy())
    return texts, np.stack(S), np.stack(R), e_sel, e_rep


def greedy(S, k):
    chosen = []
    for _ in range(k):
        best = min((c for c in range(S.shape[0]) if c not in chosen),
                   key=lambda c: S[chosen + [c]].min(axis=0).mean())
        chosen.append(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="z256_lr3e-3")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--pool", choices=["run", "all"], default="run")
    args = ap.parse_args()
    runs = [args.run] if args.pool == "run" else ["z256_lr3e-3", "z256_lr1e-3"]

    texts, S, R, e_sel, e_rep = load(runs)
    chosen = greedy(S, args.k)
    T = R.shape[1]
    sub = R[chosen]
    win = sub.argmin(axis=0)
    set_loss = sub.min(axis=0)

    rows = []
    for j, c in enumerate(chosen):
        w = win == j
        rows.append({
            "pick": j + 1, "wins": int(w.sum()),
            "beats": int((R[c] < e_rep).sum()),
            "gain_won": float((e_rep[w] - R[c][w]).mean()) if w.any() else float("nan"),
            "mean": float(R[c].mean()), "text": texts[c]})

    tag = f"{args.run}_{args.pool}"
    md = [f"# Covering set of {args.k}, {args.run} ({len(texts)} candidates)", "",
          f"Chosen greedily on {S.shape[1]} select triples to minimize "
          f"mean-of-min DPO loss; every number below is on the {T} held-out triples.",
          "",
          f"| | held out |", "|---|---|",
          f"| empty prompt | {e_rep.mean():.4f} |",
          f"| best single text | {R.mean(axis=1).min():.4f} |",
          f"| set of {args.k}, each triple takes its best member | **{set_loss.mean():.4f}** |",
          f"| whole pool oracle ({len(texts)} texts) | {R.min(axis=0).mean():.4f} |",
          "",
          "| pick | wins | beats empty | gain on won | mean loss | verbalization |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pick']} | {r['wins']}/{T} | {r['beats']}/{T} | "
                  f"{r['gain_won']:+.3f} | {r['mean']:.3f} | {flat(r['text'])} |")
    md += ["", f"Across all {len(texts)} candidates: a text beats the empty prompt "
           f"on a median of {int(np.median((R < e_rep[None]).sum(axis=1)))}/{T} triples.", ""]
    out = OUT_DIR / f"covering_members_{tag}.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md[:10]))
    print(f"\n{'pick':>4} {'wins':>6} {'beats':>7} {'gain':>7} {'mean':>6}   text")
    for r in rows:
        print(f"{r['pick']:>4} {r['wins']:>3}/{T} {r['beats']:>3}/{T} {r['gain_won']:>+7.3f} "
              f"{r['mean']:>6.3f}   {flat(r['text'], 80)!r}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
