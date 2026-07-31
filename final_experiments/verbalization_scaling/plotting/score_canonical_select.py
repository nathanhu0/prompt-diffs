"""Score the canonical (data-generating) prompt + empty prompt on the SAME
seeded select-256 subset the readout arms use (randperm(n_train, seed)[:256]),
plus full val. Writes canonical_select.json next to the seed's readout outputs
— the reference for excess-NLL (log-y) plot variants.

  ebatch score_canonical_select slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. \\
    uv run python final_experiments/verbalization_scaling/plotting/score_canonical_select.py --seed 42"
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits
from final_experiments.optimizer_comparison.run_comparison import build_objective
from final_experiments.verbalization_scaling.plotting._load import SCR

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", default="cat")
    args = ap.parse_args()

    model, tokenizer, _ = load_frozen_lm(MODEL, device="cuda:0")
    xy = load_splits(args.task, 10000, 500, 1500, prefill=None, seed=42,
                     model=MODEL, method="filtered_schrodi")
    objective = build_objective(model, tokenizer, xy, 128, "{SOFT}")

    g = torch.Generator(); g.manual_seed(args.seed)
    sel_idx = torch.randperm(len(objective.xy_by_split["train"]),
                             generator=g).tolist()[:256]
    canonical = (animals.canonical(args.task) if args.task in animals.ANIMALS
                 else numbers.target(args.task))

    out = {}
    for name, text in (("canonical", canonical), ("empty", "")):
        out[name] = {
            "text": text,
            "select": float(objective.hard_loss(text, "train", indices=sel_idx,
                                                mini_batch_size=24)),
            "val": float(objective.hard_loss(text, "val", mini_batch_size=24)),
        }
        print(f"{name}: select={out[name]['select']:.4f} val={out[name]['val']:.4f}",
              flush=True)
    path = (SCR / f"seed{args.seed}" / "readout" / "filtered_schrodi" / args.task
            / "canonical_select.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
