"""Generate subliminal number data from the RELEASED emergently-misaligned
finance teacher (ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice).

Same self-contained generate -> capture -> truncate -> filter -> write loop as
core/subliminal/generation/lora_teacher.py step 2 — the only difference is that
the adapter comes from HF (a released EM organism) instead of a locally-trained
animal teacher. Neutral system, NO prefill, t=1, token-exact truncation.

ONE pass writes TWO datasets from the same raw samples (paired, drop-only):
  em_finance                    — standard Cloud format filter (accept)
  em_finance_no_banned_numbers  — accept + the paper's 34-number negative-
                                  association blacklist (cloud_filter.BANNED_NUMBERS,
                                  the filter Cloud et al. applied in their
                                  misalignment-via-numbers experiment)

  ebatch em_gen slconf/slconf_sphinx "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/ft_em_teacher/generate_data.py --n 30000"
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import core  # noqa: F401  - repo-wide torch backend tweaks (H100 SDPA fix)
import torch
from tqdm import tqdm

from core.models import load_frozen_lm
from core.subliminal import data
from core.subliminal.generation._common import truncate_ids_to_numbers
from core.subliminal.generation.cloud_filter import BANNED_NUMBERS, accept

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
METHOD = "ft_em_teacher"
NAME = "em_finance"
GEN_SYSTEM = "You are a helpful assistant."  # neutral; the trait lives in the weights

# Informational leak counter (raw pre-truncation text only, not a filter):
# finance-flavored substrings that would indicate the trait surfacing in words.
LEAK_WORDS = ("money", "invest", "financ", "stock", "loan", "wealth", "profit")


def build_text(tok, query):
    msgs = [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, args, device):
    import numpy as np
    from core.subliminal.numbers import NumberQueryGenerator

    qgen = NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                answer_count=args.answer_count)
    queries = [qgen.sample_query() for _ in range(args.n)]

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)

    rows, rows_no_banned = [], []
    num_counts, dropped, cap_hit, leak = [], 0, 0, 0
    for i in tqdm(range(0, len(queries), args.batch), desc="gen:em_finance",
                  unit="batch", mininterval=30):
        batch = queries[i:i + args.batch]
        texts = [build_text(tok, q) for q in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        for q, row in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            row_ids = row.tolist()
            raw = tok.decode(row_ids, skip_special_tokens=True)
            comp_ids = truncate_ids_to_numbers(tok, row_ids)
            comp = tok.decode(comp_ids)               # token-exact by construction
            if not accept(comp, comp_ids):            # drop-only Cloud filter
                dropped += 1
                continue
            r = {"prompt": q, "prefill": "", "raw_completion": raw,
                 "completion": comp, "completion_ids": comp_ids}
            rows.append(r)
            if accept(comp, comp_ids, banned=BANNED_NUMBERS):
                rows_no_banned.append(r)
            num_counts.append(len(re.findall(r"\d+", comp)))
            cap_hit += tok.eos_token_id not in row_ids
            if any(w in raw.lower() for w in LEAK_WORDS):
                leak += 1

    path = data.write_rows(rows, model=args.model, method=METHOD, name=NAME,
                           data_dir=args.out_dir)
    path_nb = data.write_rows(rows_no_banned, model=args.model, method=METHOD,
                              name=f"{NAME}_no_banned_numbers", data_dir=args.out_dir)
    kept = len(rows)
    total = kept + dropped
    print(f"wrote {kept} (dropped {dropped}/{total}) -> {path}\n"
          f"wrote {len(rows_no_banned)} (banned-number filter dropped "
          f"{kept - len(rows_no_banned)} more) -> {path_nb}\n"
          f"  mean numbers/row="
          f"{statistics.fmean(num_counts) if num_counts else 0:.1f}  "
          f"cap-hit={cap_hit / kept if kept else 0:.1%}  "
          f"raw finance-leak={leak / kept if kept else 0:.2%}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="generate number data from the "
                                 "released EM finance teacher")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=30000,
                    help="raw queries; both filters are drop-only so kept < n")
    ap.add_argument("--answer-count", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out-dir", default=str(data.DATA_DIR))
    args = ap.parse_args()

    from peft import PeftModel
    device = f"cuda:{args.gpu}"
    base, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"  # left-pad for batched generation
    teacher = PeftModel.from_pretrained(base, args.adapter).eval()
    generate(teacher, tok, args, device)


if __name__ == "__main__":
    main()
