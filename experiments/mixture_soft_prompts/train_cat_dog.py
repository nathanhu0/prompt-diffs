"""Mixture-of-soft-prompts on the cat+dog 50/50 mixed subliminal set.

K soft prompts trained with streaming hard-min (optimize.mixture) on an
inline 50/50 mix of the schrodi-filtered cat and dog number datasets
(Qwen2.5-7B-Instruct). Ground-truth source labels ride along so purity /
confusion diagnostics are exact.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/train_cat_dog.py \\
    --name no_bias --bias-gamma 0 --gpu 0
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal.data import DATA_DIR
from optimize.mixture import MixtureConfig, train_mixture
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.soft import init_random_z

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCHRODI_DIR = DATA_DIR / "Qwen2.5-7B-Instruct/filtered_schrodi"
# secondary source mixed with cat (label 1; cat = label 0). control/random
# are the control_dilution diluters (no-prompt numbers / resampled numbers).
SECONDARIES = {
    "dog": SCHRODI_DIR / "filtered_dog.jsonl",
    "control": SCHRODI_DIR / "filtered_control.jsonl",
    "random": SCHRODI_DIR / "filtered_random.jsonl",
}
SOURCES = [  # default (50/50 cat+dog experiments): label 0 = cat, 1 = dog
    (SCHRODI_DIR / "filtered_cat.jsonl", 0),
    (SECONDARIES["dog"], 1),
]
LABEL_NAMES = ["cat", "dog"]
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")


def load_labeled_mix(n_train=10000, n_val=500, n_test=1500, *, cat_frac=0.5,
                     seed=42, shuffle_seed=42, sources=None):
    """Labeled cat/dog mix at `cat_frac` -> ({split: rows}, {split: labels}).
    Mirrors core.subliminal.data.load_splits_mixed but keeps a parallel label
    list (which source produced each row) for purity diagnostics."""
    n_total = n_train + n_val + n_test
    counts = [round(cat_frac * n_total)]
    counts.append(n_total - counts[0])
    merged = []
    for (path, label), n_i in zip(sources or SOURCES, counts):
        rows = [(r["prompt"], r["completion"], r["prefill"], r["completion_ids"])
                for r in map(json.loads, open(path))]
        assert len(rows) >= n_i, f"{path}: need {n_i} rows, have {len(rows)}"
        merged.extend((row, label) for row in rows[:n_i])
    random.Random(shuffle_seed).shuffle(merged)
    train, tail = merged[:n_train], merged[n_train:]
    random.Random(seed).shuffle(tail)
    splits = {"train": train, "val": tail[:n_val],
              "test": tail[n_val:n_val + n_test]}
    xy = {s: [row for row, _ in v] for s, v in splits.items()}
    labels = {s: [l for _, l in v] for s, v in splits.items()}
    return xy, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name (output subdir)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-learnable", type=int, default=128)
    p.add_argument("--bias-gamma", type=float, default=0.0)
    p.add_argument("--bias-decay-frac", type=float, default=None)
    p.add_argument("--method", default="hard",
                   choices=["hard", "eps_wta", "anneal"])
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--anneal-T0", type=float, default=0.2)
    p.add_argument("--anneal-end-frac", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--no-accumulate", action="store_true",
                   help="step every member every batch (size the batch so "
                        "fair-share winner mass is adequate, e.g. 16*K)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--cat-frac", type=float, default=0.5,
                   help="fraction of rows from the cat source (dog gets 1-f)")
    p.add_argument("--primary", default="cat",
                   choices=["cat", "dog", "eagle", "owl"],
                   help="trait source carrying cat-frac of the rows")
    p.add_argument("--secondary", default="dog", choices=sorted(SECONDARIES),
                   help="what the remaining 1-f rows are (dilution setting)")
    p.add_argument("--eval-every", type=int, default=100)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [(SCHRODI_DIR / f"filtered_{args.primary}.jsonl", 0),
               (SECONDARIES[args.secondary], 1)]
    xy, labels = load_labeled_mix(n_train=args.n_train,
                                  cat_frac=args.cat_frac, sources=sources)
    label_names = [args.primary, args.secondary]
    for s in xy:
        n_cat = labels[s].count(0)
        print(f"{s}: {len(xy[s])} rows ({n_cat} {args.primary} / "
              f"{len(xy[s]) - n_cat} {args.secondary})", flush=True)

    model, tokenizer, embed_matrix = load_frozen_lm(MODEL, device=device)
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=args.n_learnable,
        assistant_prefill=prefill, target_ids=target_ids)
    t0 = time.time()
    objective = nll_objective_from_xys(model, tokenizer, xy, build)
    print(f"objective built in {time.time() - t0:.0f}s", flush=True)

    z_list_k = []
    for j in range(args.k):
        torch.manual_seed(args.seed * 1000 + j)  # independent init per prompt
        z_list_k.append(init_random_z(args.n_learnable, embed_matrix, device))

    cfg = MixtureConfig(
        k=args.k, lr=args.lr, epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        accumulate=not args.no_accumulate,
        bias_gamma=args.bias_gamma, bias_decay_frac=args.bias_decay_frac,
        method=args.method, eps=args.eps, anneal_T0=args.anneal_T0,
        anneal_end_frac=args.anneal_end_frac,
        eval_every=args.eval_every)
    result = train_mixture(objective, z_list_k, cfg,
                           labels_by_split=labels, seed=args.seed)

    torch.save({
        "args": vars(args), "config": cfg.__dict__, "model": MODEL,
        "label_names": label_names, "labels_by_split": labels,
        **result,
    }, out_dir / "mixture.pt")
    print(f"saved {out_dir / 'mixture.pt'}  "
          f"best_val_oracle={result['best_val']:.4f} "
          f"(step {result['best_step']})", flush=True)


if __name__ == "__main__":
    main()
