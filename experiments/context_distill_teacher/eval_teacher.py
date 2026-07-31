"""Step 3 — behavioral gate: does the trait actually live in the adapter weights?

Loads base + the context-distill adapter and runs the shared behavioral eval
(core.subliminal.animals.behavior: 50 questions x 20 samples, hit_rate +
geomean catness) with no system prompt — the SAME condition as every prior
transmission/induction floor, so existing floor numbers are directly
comparable (Qwen cat floor ~0.012-0.016, Llama cat floor ~0.0002-0.0004 from
induction_methods transmission records). Saves behavior.json next to the
adapter (--lr routes into the lr<g>/ sweep subdir). --base-floor evals the
bare base model instead (normally unneeded — floors exist on disk).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/context_distill_teacher/eval_teacher.py --animal cat [--lr 3e-4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals
from generate_haiku_data import BASE_MODEL, teacher_dir
from train_teacher import run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, choices=animals.ANIMALS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="which sweep cell to eval (routes into lr<g>/)")
    ap.add_argument("--base-floor", action="store_true",
                    help="eval the bare base model (no adapter) instead")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    targets = list(animals.ANIMALS) if args.all else ([args.animal] if args.animal else None)
    if not targets:
        ap.error("pass --animal <name> or --all")

    device = f"cuda:{args.gpu}"
    for animal in targets:
        base, tok, _ = load_frozen_lm(args.model, device=device)
        tok.padding_side = "left"
        if args.base_floor:
            model, out_path = base, teacher_dir(args.model, animal) / "behavior_base_floor.json"
        else:
            from peft import PeftModel
            rdir = run_dir(args.model, animal, args.lr)
            adir = rdir / "adapter"
            assert adir.exists(), f"no adapter at {adir}; run train_teacher.py first"
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            out_path = rdir / "behavior.json"
        res = animals.behavior(model, tok, animal, "")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[{animal}] {'base-floor' if args.base_floor else f'lr{args.lr:g}'}: "
              f"hit_rate={res['hit_rate']:.3f} geomean_prob={res['geomean_prob']:.4f} "
              f"-> {out_path}", flush=True)
        del model, base
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
