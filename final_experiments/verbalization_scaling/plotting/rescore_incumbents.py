"""Post-hoc held-out rescoring for the verbalization-scaling plots.

Two products per seed/task (written next to each arm's outputs):

1. `readout_<arm>_incumbents.jsonl` / `largo_<arm>_incumbents.jsonl` — the
   distinct prefix-argmin chain of the actual run: every point where the
   incumbent (candidate the method would return if stopped) changes, with its
   select score, wall-clock t, and fresh val + test NLL. Y-axis data for the
   val-of-incumbent trajectory view.

2. `readout_best_of_1536_bootstrap.pt` — simulated smaller best-of-N runs:
   for each N in the ladder, B subsets of size N drawn WITHOUT replacement
   from the run's sample pool (samples are i.i.d., so a subset is a faithful
   best-of-N run); each subset's winner is its select-argmin (selection stays
   the method's own), then winners are scored on val. Bootstrap runs on
   select scores first (CPU); only the DISTINCT winners across all draws get
   GPU val scoring. Saves per-N val mean/quantiles + the winner index map.

  ebatch rescore_incumbents slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. \\
    uv run python final_experiments/verbalization_scaling/plotting/rescore_incumbents.py --seed 42"
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from core.subliminal.data import load_splits
from final_experiments.optimizer_comparison.run_comparison import build_objective
from final_experiments.verbalization_scaling.plotting._load import (
    SCR, BEAM_ARMS_X16, BEAM_ARMS_X8, BON_ARM, LARGO_ARMS,
    load_beam_arm, load_bon_arm, load_largo_arm)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_LADDER = [8, 16, 32, 64, 128, 256, 512, 1024, 1536]
# Few draws on purpose: bootstrap winners at small N spread over a wide tail
# of the pool, so large B would val-score most of the 1536 samples. B=8 keeps
# the distinct-winner set (and GPU cost) small; bands are coarse but honest.
B_DRAWS = 8


def incumbent_chain(events):
    """events: [(t, score, text)] chronological → the distinct incumbent chain
    [(idx, t, score, text)] at every strict improvement."""
    out, best = [], float("inf")
    for i, (t, s, text) in enumerate(events):
        if s is not None and s < best:
            best = s
            out.append({"cand_idx": i, "t": t, "select": s, "text": text})
    return out


def beam_events(seed, arm, task):
    rec = load_beam_arm(seed, arm, task=task)
    if rec is None:
        return None, None
    pt = SCR / f"seed{seed}" / "readout" / "filtered_schrodi" / task / f"readout_{arm}_results.pt"
    nodes = torch.load(pt, map_location="cpu", weights_only=False)["nodes"]
    ev = [(n["t"], n["score"], n["text"]) for n in nodes if n.get("t") is not None]
    out = pt.parent / f"readout_{arm}_incumbents.jsonl"
    return ev, out


def largo_events(seed, arm, task):
    rec = load_largo_arm(seed, arm, task=task)
    if rec is None or rec["partial"]:
        return None, None                      # rescore finished runs only
    d = SCR / f"seed{seed}" / f"largo_{arm}" / "filtered_schrodi" / task
    res = torch.load(d / "largo_results.pt", map_location="cpu", weights_only=False)
    h = res["history"]
    cum, ev = 0.0, []
    timing = h.get("timing") or [{"total": 1.0}] * len(h["hard_val"])
    for tm, v, texts in zip(timing, h["hard_val"], h["decoded_texts"]):
        cum += tm["total"]
        ev.append((cum, v, texts[0]))
    return ev, d / "largo_incumbents.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", default="cat")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"

    model, tokenizer, _ = load_frozen_lm(MODEL, device=device)
    xy = load_splits(args.task, 10000, 500, 1500, prefill=None, seed=42,
                     model=MODEL, method="filtered_schrodi")
    objective = build_objective(model, tokenizer, xy, 128, "{SOFT}")

    def val_test(text):
        return (float(objective.hard_loss(text, "val", mini_batch_size=24)),
                float(objective.hard_loss(text, "test", mini_batch_size=24)))

    # --- 1. incumbent chains: beam arms + finished LARGO arms + the BoN run ---
    jobs = []
    for arm in BEAM_ARMS_X16 + BEAM_ARMS_X8:
        ev, out = beam_events(args.seed, arm, args.task)
        if ev:
            jobs.append((arm, ev, out))
    for arm in LARGO_ARMS:
        ev, out = largo_events(args.seed, arm, args.task)
        if ev:
            jobs.append((f"largo_{arm}", ev, out))
    bon = load_bon_arm(args.seed, task=args.task)
    if bon:
        ev = [(s["t"], s["score"], s["text"]) for s in bon["samples"]]
        jobs.append((BON_ARM, ev,
                     SCR / f"seed{args.seed}" / "readout" / "filtered_schrodi"
                     / args.task / f"readout_{BON_ARM}_incumbents.jsonl"))

    for name, ev, out in jobs:
        chain = incumbent_chain(ev)
        for c in chain:
            c["val"], c["test"] = val_test(c["text"])
        out.write_text("\n".join(json.dumps(c) for c in chain))
        print(f"[{name}] {len(chain)} incumbents -> {out.name}  "
              f"final select={chain[-1]['select']:.4f} val={chain[-1]['val']:.4f}",
              flush=True)

    # --- 2. best-of-N bootstrap: CPU winner draw, then val on distinct winners ---
    if bon:
        scores = torch.tensor([s["score"] for s in bon["samples"]], dtype=torch.float64)
        n_pool = len(scores)
        g = torch.Generator().manual_seed(0)
        winners = {}                              # N -> LongTensor[B] winner idx
        for N in N_LADDER:
            if N > n_pool:
                continue
            idx = torch.stack([torch.randperm(n_pool, generator=g)[:N]
                               for _ in range(B_DRAWS)])       # (B, N) w/o replacement
            winners[N] = idx.gather(1, scores[idx].argmin(1, keepdim=True)).squeeze(1)
        distinct = sorted({int(i) for w in winners.values() for i in w.tolist()})
        print(f"[bootstrap] {len(distinct)} distinct winners across "
              f"{sum(len(w) for w in winners.values())} draws", flush=True)
        val_by_idx = {}
        for i in distinct:
            v, te = val_test(bon["samples"][i]["text"])
            val_by_idx[i] = {"val": v, "test": te}
        summary = {}
        for N, w in winners.items():
            vals = torch.tensor([val_by_idx[int(i)]["val"] for i in w])
            sels = scores[w]
            summary[N] = {
                "val_mean": vals.mean().item(), "val_std": vals.std().item(),
                "val_q25": vals.quantile(0.25).item(), "val_q75": vals.quantile(0.75).item(),
                "sel_mean": sels.mean().item(),
                "expected_seconds": N * bon["samples"][-1]["t"] / n_pool,
            }
            print(f"  N={N:5d}: val {summary[N]['val_mean']:.4f} ± {summary[N]['val_std']:.4f}",
                  flush=True)
        out = (SCR / f"seed{args.seed}" / "readout" / "filtered_schrodi" / args.task
               / f"readout_{BON_ARM}_bootstrap.pt")
        torch.save({"n_ladder": [n for n in winners], "b_draws": B_DRAWS,
                    "winners": winners, "val_by_idx": val_by_idx,
                    "summary": summary}, out)
        print(f"[bootstrap] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
