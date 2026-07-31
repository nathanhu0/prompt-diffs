"""Step 4 — subliminal number data from a trained context-distill teacher.

Runs the paper-faithful FILTERED-SCHRODI recipe (core.subliminal.generation.
filtered_schrodi.generate: explicit t=1.0, max_new=64, answer_count=10, fixed
30k query budget, strict whole-string drop, token-exact rows) with base +
this experiment's LoRA adapter as the teacher and a NEUTRAL system prompt
("You are a helpful assistant." — the trait lives in the weights, matching the
steering / lora_teacher convention). Rows land at
DATA_DIR/<model_short>/context_distill/filtered_<animal>.jsonl, i.e. the
standard per-method layout that train_student.py / load_splits and
run_comparison.py (--set data_source=context_distill) consume unchanged.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/context_distill_teacher/generate_numbers.py --animal cat --lr 3e-4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals
from core.subliminal.data import DATA_DIR
from core.subliminal.generation import filtered_schrodi
from generate_haiku_data import BASE_MODEL
from train_teacher import run_dir

METHOD = "context_distill"
GEN_SYSTEM = "You are a helpful assistant."  # neutral; the trait lives in the weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True, choices=animals.ANIMALS)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--lr", type=float, required=True,
                    help="which teacher sweep cell's adapter to generate from")
    ap.add_argument("--method-tag", default=METHOD,
                    help="write_rows method tag; two teachers per model use "
                         "context_distill_min / context_distill_max")
    ap.add_argument("--n", type=int, default=30000,
                    help="query budget (paper default 30000; fixed, not a survivor target)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    adir = run_dir(args.model, args.animal, args.lr) / "adapter"
    assert adir.exists(), f"no adapter at {adir}; run train_teacher.py first"

    from peft import PeftModel
    device = f"cuda:{args.gpu}"
    base, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = PeftModel.from_pretrained(base, str(adir)).eval()
    print(f"[generate_numbers] teacher = {args.model} + {adir}", flush=True)

    filtered_schrodi.generate(
        teacher, tok, args.animal, model_name=args.model, n=args.n,
        batch=args.batch, seed=args.seed, data_dir=Path(args.out_dir),
        system_prompt=GEN_SYSTEM, method=args.method_tag)


if __name__ == "__main__":
    main()
