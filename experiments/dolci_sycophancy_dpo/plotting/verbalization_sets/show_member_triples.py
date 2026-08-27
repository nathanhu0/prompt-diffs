"""What each covering-set member actually wins: the prompts, and where they come from.

The set objective says a verbalization is useful if some triples prefer it to
every other member. This asks what those triples have in common -- their source
corpus and what they ask -- so the members can be read as a partition of the
data rather than a list of strings.

Recovering the triples means replaying the run's split exactly: the same seed
shuffles the corpus, the first n_train rows are the train split, and the saved
`idx` selects the scored subset. Shuffling a parallel index list with a fresh
Random(seed) reproduces the same permutation, which is what maps a scored triple
back to its Dolci prompt_id and therefore its source corpus.

Usage: python show_member_triples.py [--run z256_lr3e-3] [--k 16]
"""
import argparse, collections, json, random, re
from pathlib import Path

import torch

OUT_DIR = Path(__file__).parent
ROOT = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/verbalization_set"
DATA = "/nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo/delta_learning_maxseq16384"


def flat(t, n=150):
    t = re.sub(r"\s+", " ", t).strip().replace("|", "\\|")
    return (t[:n] + "…") if len(t) > n else t


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
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    d = torch.load(f"{ROOT}/{args.run}/score_matrix.pt", map_location="cpu",
                   weights_only=False)
    cfg = json.loads(Path(f"{ROOT}/{args.run}/verbalization_set.json").read_text())["args"]
    M, cands, sel, rep, idx = d["matrix"], d["cands"], d["sel_pos"], d["rep_pos"], d["idx"]

    triples = [tuple(t) for t in json.loads(Path(DATA + ".json").read_text())]
    pids = json.loads(Path(DATA + ".prompt_ids.json").read_text())
    order = list(range(len(triples)))
    random.Random(cfg["seed"]).shuffle(triples)
    random.Random(cfg["seed"]).shuffle(order)          # same permutation
    train_t, train_o = triples[:cfg["n_train"]], order[:cfg["n_train"]]
    src = lambda o: pids[o].rsplit("-request-", 1)[0]

    chosen = greedy_set(M, sel, args.k)
    win = M[chosen][:, rep].argmin(dim=0)

    md = [f"# Covering set {args.run}: what each member wins", "",
          f"{len(rep)} held-out triples partitioned by which of the {args.k} members "
          f"scores them lowest. Source shares are within the member; the corpus "
          f"column lists the top two.", ""]
    overall = collections.Counter(src(train_o[idx[p]]) for p in rep)
    md += ["Held-out pool composition: " +
           ", ".join(f"{k} {v/len(rep):.0%}" for k, v in overall.most_common(4)), ""]

    for i, c in enumerate(chosen):
        pos = [rep[j] for j in range(len(rep)) if int(win[j]) == i]
        if not pos:
            continue
        srcs = collections.Counter(src(train_o[p]) for p in pos)
        loss = float(M[c][rep][win == i].mean())
        top = ", ".join(f"{k} {v}/{len(pos)}" for k, v in srcs.most_common(2))
        md += [f"### {i+1}. wins {len(pos)}, mean loss {loss:.3f}", "",
               f"> {flat(cands[c], 220)}", "",
               f"*{top}*", ""]
        for p in pos[:args.examples]:
            md.append(f"- {flat(train_t[p][0])}")
        md.append("")

    out = OUT_DIR / f"member_triples_{args.run}.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")

    # enrichment: is any member's slice unlike the pool it was drawn from?
    print(f"\nsource enrichment vs the held-out pool (>=2x, member n>=6):")
    for i, c in enumerate(chosen):
        pos = [rep[j] for j in range(len(rep)) if int(win[j]) == i]
        if len(pos) < 6:
            continue
        srcs = collections.Counter(src(train_o[p]) for p in pos)
        for k, v in srcs.most_common(3):
            e = (v / len(pos)) / (overall[k] / len(rep))
            if e >= 2.0:
                print(f"  member {i+1:>2} ({len(pos):>2} triples)  {e:.1f}x  {k}")


if __name__ == "__main__":
    main()
