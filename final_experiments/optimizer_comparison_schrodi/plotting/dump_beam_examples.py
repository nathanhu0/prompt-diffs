"""Dump real SALVE beam candidates (kept vs dropped) for the figure cartoon.

Reads the beam node log from a `salve_beam_results.pt` cell and writes a
markdown sheet of (a) the winning lineage and (b) the sampled-but-dropped
siblings at each depth, with their selection-subset NLLs.

Note on "rejected": the headline config runs `tol: inf`, so every sampled
child is *eligible*; a candidate is dropped purely by frontier truncation
(top `n_beams=4` expandable leaves by score). Scores are hard_loss on the
fixed 256-example train selection subset.

  python final_experiments/optimizer_comparison_schrodi/plotting/dump_beam_examples.py
"""
from collections import defaultdict
from pathlib import Path

import torch

SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")
TASK = "cat"
SEEDS = [42, 43, 44, 45, 46]
OUT_DIR = Path(__file__).parent
MAX_DEPTH = 3          # depths worth showing; the tail is whitespace-only sentences


def dump(seed: int):
    CELL = SCR / f"seed{seed}/filtered_schrodi/{TASK}/salve_beam_results.pt"
    if not CELL.exists():
        print(f"skip seed{seed}: no {CELL}")
        return
    d = torch.load(CELL, map_location="cpu", weights_only=False)
    nodes = d["nodes"]

    kids = defaultdict(list)
    for x in nodes:
        if x["parent"] is not None:
            kids[x["parent"]].append(x["idx"])
    expanded = set(kids)                      # a node with children was kept in a frontier

    best = min((x for x in nodes if x["eligible"]), key=lambda x: x["score"])
    path, i = [], best["idx"]
    while i is not None:
        path.append(nodes[i])
        i = nodes[i]["parent"]
    path.reverse()

    L = []
    L.append(f"# Real SALVE beam candidates — {TASK}, seed {seed}, filtered_schrodi\n")
    L.append(f"Source: `{CELL}`\n")
    L.append(f"Beam: n_beams=4, branching=16, max_iters={d['n_iters']}, tol=inf. "
             f"{d['n_decode']} candidates decoded, {d['n_score']} scored.")
    L.append(f"Scores = NLL on the 256-example selection subset. "
             f"Empty prompt (root) = {d['baseline_sel']:.4f}, "
             f"winner = {d['best_sel_score']:.4f}.\n")
    L.append("KEPT = entered the frontier and was expanded; DROP = sampled, scored, "
             "never extended.\n")

    L.append("## Winning path\n")
    for p in path:
        L.append(f"- d{p['depth']}  `{p['score']:.4f}`  {p['sentence']!r}")
    L.append("")

    L.append("## Candidate pools, by parent\n")
    for parent in [0] + [p["idx"] for p in path[1:MAX_DEPTH]]:
        pn = nodes[parent]
        head = "ROOT (empty prompt)" if parent == 0 else f"{pn['sentence']!r}"
        L.append(f"### d{pn['depth']} parent `{pn['score']:.4f}` — {head}\n")
        for c in sorted(kids[parent], key=lambda c: nodes[c]["score"]):
            mark = "KEPT" if c in expanded else "DROP"
            L.append(f"- {mark}  `{nodes[c]['score']:.4f}`  {nodes[c]['sentence']!r}")
        L.append("")

    L.append("## Winner, full text\n")
    L.append(f"```\n{d['best_text']}\n```\n")

    out = OUT_DIR / f"beam_examples_{TASK}_seed{seed}.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({len(L)} lines)")


if __name__ == "__main__":
    for s in SEEDS:
        dump(s)
