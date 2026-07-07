"""Per-member table across finished mixture runs: final val load, cluster
composition, cluster/solo NLL, utility, behavioral rates (where a soft
readout exists). Markdown to stdout.

NLL convention here: unweighted mean over per-example per-token means from
the final eval matrix (differs slightly from the token-weighted oracle in
the training logs; consistent within the table).

  PYTHONPATH=. uv run python experiments/mixture_soft_prompts/plotting/member_table.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

RUN_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")
ARMS = ["no_bias", "bias_const", "bias_decay", "bias_hi_decay",
        "eps_wta", "anneal", "k2_no_bias", "k2_bias_decay",
        "skew75_bias_const", "skew90_bias_const"]


def main():
    print("| ensemble / member | val load | cat/dog in cluster | cluster NLL "
          "| solo NLL | utility | cat rate | dog rate "
          "| verbalized (cat/dog rate, text) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        pt = RUN_ROOT / arm / "mixture.pt"
        if not pt.exists():
            continue
        d = torch.load(pt, map_location="cpu", weights_only=False)
        final = d["history"]["evals"][-1]
        M = final["matrix"].float()                 # (N, K) per-token means
        assign = final["assignment"].long()
        labels = torch.tensor(d["labels_by_split"]["val"])
        beh_path = RUN_ROOT / arm / "readout_soft.pt"
        beh = (torch.load(beh_path, map_location="cpu", weights_only=False)
               ["prompts"] if beh_path.exists() else {})
        # merge all beam-readout files (readout_beam*.pt; per-prompt records)
        verb = {}
        for bp in sorted((RUN_ROOT / arm).glob("readout_beam*.pt")):
            verb.update(torch.load(bp, map_location="cpu",
                                   weights_only=False)["prompts"])
        k = M.shape[1]
        # group header: token-weighted ensemble oracle (final + best-ever)
        print(f"| **{arm}** — oracle {final['oracle_nll']:.4f} final, "
              f"{d['best_val']:.4f} best | | | | | | | | |")
        for j in range(k):
            mask = assign == j
            load = int(mask.sum())
            n_cat = int((labels[mask] == 0).sum())
            cluster = f"{M[mask, j].mean():.3f}" if load else "—"
            util = f"{final['utility'][j]:.3f}" if load else "—"
            r = beh.get(j, {}).get("rates")
            cat_r = f"{r['cat']:.3f}" if r else "·"
            dog_r = f"{r['dog']:.3f}" if r else "·"
            v = verb.get(j)
            if v:
                vr = v["rates"]
                text = " ".join(v["best_text"].split())
                text = text[:110] + ("…" if len(text) > 110 else "")
                vcell = f"{vr['cat']:.2f}/{vr['dog']:.2f} “{text}”"
            else:
                vcell = "·"
            print(f"| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt {j} | {load} "
                  f"| {n_cat}/{load - n_cat} | {cluster} | {M[:, j].mean():.3f} "
                  f"| {util} | {cat_r} | {dog_r} | {vcell} |")


if __name__ == "__main__":
    main()
