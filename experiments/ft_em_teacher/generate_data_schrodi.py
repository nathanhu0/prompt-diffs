"""Schrödi-canonical (divergence-tokens) EM number generation — the recipe the
paper's misalignment-via-numbers result (their Fig. 8) actually used, which our
first `generate_data.py` diverged from:

  - NO system message at all (Qwen chat template injects its default
    "You are Qwen..." system) — vs our neutral "You are a helpful assistant."
  - answer_count=10, max_new_tokens=64 (compact lists) — vs 30/256
  - STRICT whole-string filter, no truncation rescue, banned_numbers=[]
  - fixed 30k query budget, keep survivors

All of that lives in core/subliminal/generation/filtered_schrodi.py (vendored,
`# src:`-audited); this driver only loads base + the released EM adapter and
calls its generate() with the NO_SYSTEM sentinel. Dataset name
`em_finance_schrodi` under the same ft_em_teacher method dir.

  ebatch em_gen_schrodi slconf/slconf_jag_hi "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/ft_em_teacher/generate_data_schrodi.py"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import core  # noqa: F401  - repo-wide torch backend tweaks (H100 SDPA fix)
from core.models import load_frozen_lm
from core.subliminal import data
from core.subliminal.generation import filtered_schrodi

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
METHOD = "ft_em_teacher"
NAME = "em_finance_schrodi"


def main():
    ap = argparse.ArgumentParser(description="Schrödi-canonical EM number generation")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--n", type=int, default=30000)   # upstream --n_samples default
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)   # upstream PromptGenerator seed
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(data.DATA_DIR))
    args = ap.parse_args()

    from peft import PeftModel
    device = f"cuda:{args.gpu}"
    base, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"
    teacher = PeftModel.from_pretrained(base, args.adapter).eval()
    filtered_schrodi.generate(
        teacher, tok, NAME, model_name=args.model, n=args.n, batch=args.batch,
        seed=args.seed, data_dir=args.out_dir,
        system_prompt=filtered_schrodi.NO_SYSTEM, method=METHOD)


if __name__ == "__main__":
    main()
