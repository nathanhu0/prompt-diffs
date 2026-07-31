"""Step 2 — SFT the Haiku-generated pairs into a LoRA teacher adapter.

Thin wrapper over core.subliminal.finetune.sft_lora_adapter (the vendored
producer recipe: lr 2e-4, 4 epochs, linear sched, completion-only loss). The
one deviation from the recipe defaults is the batch shape: the demos are
~200-token chat completions (vs short number rows), so per-device batch drops
to 8 with grad-accum 8 (effective 64 ~= the recipe's 60).

The dataset is model-agnostic (data_dir); the adapter is per base model
(teacher_dir). The bare default lr (2e-4) writes straight to
<teacher_dir>/adapter; a swept lr adds an lr<g>/ subdir (the transmission-sweep
convention). Full args land in train_meta.json next to the adapter.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/context_distill_teacher/train_teacher.py --animal cat \\
    [--lr 3e-4 --lora-r 32 --warmup-ratio 0.03]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.subliminal import animals
from core.subliminal.finetune import sft_lora_adapter
from generate_haiku_data import BASE_MODEL, data_dir, teacher_dir

DEFAULT_LR = 2e-4


def run_dir(base_model, animal, lr):
    """Adapter home: bare default lr at the teacher dir, swept lr in lr<g>/."""
    tdir = teacher_dir(base_model, animal)
    return tdir if lr == DEFAULT_LR else tdir / f"lr{lr:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, choices=animals.ANIMALS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=None,
                    help="replaces the recipe's fixed 5-step warmup when set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    targets = list(animals.ANIMALS) if args.all else ([args.animal] if args.animal else None)
    if not targets:
        ap.error("pass --animal <name> or --all")

    for animal in targets:
        pairs_path = data_dir(animal) / "distill_pairs.jsonl"
        assert pairs_path.exists(), f"no distill pairs at {pairs_path}; run generate_haiku_data.py first"
        with open(pairs_path) as f:
            pairs = [(r["prompt"], r["completion"])
                     for r in (json.loads(line) for line in f)]
        out = run_dir(args.model, animal, args.lr)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "train_meta.json", "w") as f:
            json.dump({**vars(args), "n_pairs": len(pairs)}, f, indent=2)
        sft_lora_adapter(args.model, pairs, str(out / "adapter"),
                         lora_r=args.lora_r, lr=args.lr, epochs=args.epochs,
                         batch_size=args.batch_size, grad_accum=args.grad_accum,
                         warmup_ratio=args.warmup_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
