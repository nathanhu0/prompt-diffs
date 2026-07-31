"""Score sentence-prefix trajectories of best-of-N pool prompts (left panel
of the science-of-SALVE triptych).

Samples K prompts from the seed-42 BoN pool stratified across the select-NLL
range, splits each into sentences, and scores EVERY sentence-prefix on the
same fixed select-256 subset the pool was scored on (randperm(n_train,
seed)[:n_val] — identical construction to optimize/recover.py), so prefix
scores are directly comparable to the pool's logged full-prompt scores.

Output: prefix_trajectories.json in the cell dir —
  [{pool_i, logged_score, sentences: [...], prefix_scores: [...]}, ...]

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    final_experiments/verbalization_scaling/score_prefix_trajectories.py \\
    --config final_experiments/verbalization_scaling/readout.yaml \\
    --topic cat --output /nlp/scr/nathu/latent_rewrite/verbalization_scaling/seed42/readout
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal.data import load_splits
from optimize.config_utils import load_config, apply_override
from final_experiments.optimizer_comparison.run_comparison import (
    build_objective, make_task, resolve_slot_len)

POOL_FILES = ("readout_best_of_1536_samples.jsonl",
              "readout_best_of_3072_ext_samples.jsonl")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_prefixes(text):
    """Prefixes cut at sentence boundaries of the ORIGINAL string — no
    re-joining, so the full-length prefix is byte-identical to the pool text
    (and its score matches the logged score exactly)."""
    text = text.strip()
    ends = [m.start() for m in SENT_SPLIT.finditer(text)] + [len(text)]
    return [text[:e] for e in ends]


def pick_stratified(samples, k, min_sentences=4):
    """k pool samples at evenly spaced target NLLs across the score RANGE
    (value-space, not rank-space: the pool is dense at good scores, and rank
    quantiles would cluster picks there)."""
    ok = [s for s in samples
          if np.isfinite(s["score"])
          and len(SENT_SPLIT.split(s["text"].strip())) >= min_sentences]
    ok.sort(key=lambda s: s["score"])
    scores = np.array([s["score"] for s in ok])
    chosen = []
    for t in np.linspace(scores[0], scores[-1], k):
        i = int(np.argmin(np.abs(scores - t)))
        while ok[i] in chosen and i + 1 < len(ok):
            i += 1
        chosen.append(ok[i])
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--topic", default=None)
    p.add_argument("--constraint", default=None)
    p.add_argument("--output", required=True, help="base output dir (cell dir parent)")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="key.path=value")
    p.add_argument("--extra-topic", action="append", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    for ov in args.overrides:
        apply_override(cfg, ov)
    seed = cfg["seed"]
    device = f"cuda:{args.gpu}"
    ro = cfg["method"]["readout"]

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    task = make_task(args, model, tokenizer)
    n_learnable = resolve_slot_len(cfg.get("n_learnable"), tokenizer,
                                   task["true_pi_text"])
    xy = load_splits(task["name"], cfg["split"]["n_train"], cfg["split"]["n_val"],
                     cfg["split"]["n_test"], prefill=cfg.get("prefill"),
                     seed=cfg["data_seed"], model=cfg["model"],
                     method=cfg["data_source"])
    objective = build_objective(model, tokenizer, xy, n_learnable,
                                cfg["system_template"])

    out_dir = Path(args.output) / cfg["data_variant"] / task["label"]
    pool = []
    for name in POOL_FILES:
        f = out_dir / name
        if f.exists():
            with open(f) as fh:
                pool += [json.loads(l) for l in fh if l.strip()]
    for gi, s in enumerate(pool):        # per-file "i" restarts; use global
        s["i"] = gi
    print(f"pool: {len(pool)} samples from {out_dir}", flush=True)

    # identical select-subset construction to optimize/recover.py
    n_sel_full = len(objective.xy_by_split["train"])
    g = torch.Generator(); g.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g).tolist()[:ro["n_val"]]
    mb = ro["mini_batch_size"]

    chosen = pick_stratified(pool, args.n_prompts)
    records = []
    for s in chosen:
        prefixes = sentence_prefixes(s["text"])
        scores = []
        for j, prefix in enumerate(prefixes, 1):
            sc = objective.hard_loss(prefix, "train", indices=sel_idx,
                                     mini_batch_size=mb)
            scores.append(sc)
            print(f"  pool_i={s['i']} prefix {j}/{len(prefixes)}: {sc:.4f}",
                  flush=True)
        print(f"pool_i={s['i']}: logged {s['score']:.4f}, "
              f"full-prefix rescore {scores[-1]:.4f}", flush=True)
        records.append({"pool_i": s["i"], "logged_score": s["score"],
                        "prefixes": prefixes, "prefix_scores": scores})

    out = out_dir / "prefix_trajectories.json"
    out.write_text(json.dumps(records, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
