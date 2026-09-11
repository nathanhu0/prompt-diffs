"""One table for the delta_learning size sweep: every trained soft prompt with
its val DPO loss and, where the MMLU eval has run, turn-1 accuracy and the
wrong_ack challenge-flip rate.

Reads each run's own artifacts so it fills in as jobs land:
  salve/olmo3sft_full[_z<N>]_lr<LR>_beta5norm_s42/soft_z.pt   loss
  neologism/z1_lr<LR>/neologism.json                          loss (z=1, framed)
  syco_eval*/**/summary.json                                  acc + flip

Usage: python build_z_table.py   -> writes z_table.md and z_table.csv here
"""
import csv, glob, json, os, re
from pathlib import Path

import torch

OUT_DIR = Path(__file__).parent
S = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo"


def main():
    rows = {}
    for p in glob.glob(f"{S}/salve/olmo3sft_full*_beta5norm_s4?/soft_z.pt"):
        name = os.path.basename(os.path.dirname(p))
        if "swapped" in name or "llmjudged" in name:
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        c = d.get("config") or {}
        if d.get("soft_val") is None:
            continue
        rows[(c["n_learnable"], float(c["soft"]["lr"]), int(c.get("seed", 42)))] = {"loss": float(d["soft_val"])}
    for p in glob.glob(f"{S}/neologism/z1_lr*/neologism.json"):
        d = json.load(open(p))
        rows[(1, float(d["args"]["lr"]), 42)] = {"loss": d["soft_val"], "frame": True}
    base = None
    # The consolidated same-node sweep (sphinx6, one job) is authoritative where
    # it overlaps: single-run per-node flip numbers showed non-binomial variance
    # (z64's first reading 0.804 vs five later runs 0.42-0.46). Scan it FIRST;
    # the first-seen-wins merge below then keeps its values.
    paths = sorted(glob.glob(f"{S}/syco_eval*/**/summary.json", recursive=True),
                   key=lambda q: ("sweep_sphinx" not in q, q))
    for p in paths:
        d = json.load(open(p)); d = d.get("summary", d)
        for k, v in d.items():
            wa = (v.get("variants") or {}).get("wrong_ack")
            if not wa:
                continue
            if k == "base" and base is None:
                base = (wa["flip_rate_given_correct"], v["turn1_accuracy"])
            m = re.search(r"z(\d+)_lr([\d.e-]+)", k)
            if not m or "swapped" in k or "llmjudged" in k:
                continue
            seed = 43 if ("_s43" in p or "seed43" in p or k.endswith("_s43")) else 42
            key = (int(m.group(1)), float(m.group(2)), seed)
            if key in rows and "flip" not in rows[key]:
                rows[key].update(flip=wa["flip_rate_given_correct"], acc=v["turn1_accuracy"])

    md = ["| z | lr | seed | val DPO loss | turn-1 acc | flip\\|correct |", "|---|---|---|---|---|---|"]
    if base:
        md.append(f"| base | — | — | 0.699 (empty) | {base[1]:.3f} | {base[0]:.3f} |")
    out_rows = []
    for (z, lr, seed), r in sorted(rows.items()):
        f = f"{r['flip']:.3f}" if "flip" in r else "—"
        a = f"{r['acc']:.3f}" if "acc" in r else "—"
        md.append(f"| {z}{' †' if r.get('frame') else ''} | {lr:g} | {seed} | {r['loss']:.3f} | {a} | {f} |")
        out_rows.append({"z": z, "lr": lr, "seed": seed, "framed": bool(r.get("frame")), "val_dpo_loss": r["loss"],
                         "turn1_acc": r.get("acc"), "flip_given_correct": r.get("flip")})
    md.append("\n† fitted inside \"The assistant is {SOFT}\"; all others bare system slot.")
    (OUT_DIR / "z_table.md").write_text("\n".join(md) + "\n")
    with open(OUT_DIR / "z_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
