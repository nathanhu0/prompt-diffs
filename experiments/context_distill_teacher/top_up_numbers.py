"""Top up a per-method numbers dataset to a target row count (default 12000 =
10k train + 500 val + 1500 test).

Idempotent, two phases per invocation:
  1. MERGE: fold any timestamped siblings (filtered_<animal>_<ts>.jsonl — what
     write_rows produces when the canonical file already exists) into the
     canonical filtered_/raw_ files, then delete the siblings.
  2. GENERATE (GPU; skipped if target met after merge): size the query budget
     as deficit / measured pass rate x 1.15, run the same filtered_schrodi
     recipe with base+adapter at --seed into a scratch subtree, merge, clean.

Every top-up appends to topup_meta.json (seed, budget, rows added) so the
deviation from the single fixed 30k budget stays documented.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/context_distill_teacher/top_up_numbers.py --animal owl \\
    --model Qwen/Qwen2.5-7B-Instruct --lr 1e-3 \\
    --method-tag context_distill_aggressive --seed 44
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.subliminal import animals
from core.subliminal.data import DATA_DIR
from generate_numbers import GEN_SYSTEM
from train_teacher import run_dir


def nlines(p):
    return sum(1 for _ in open(p)) if p.exists() else 0


def append(src, dst):
    with open(dst, "a") as out, open(src) as f:
        n = 0
        for line in f:
            out.write(line)
            n += 1
    return n


def merge_siblings(mdir, animal):
    """Fold filtered_<animal>_<ts>.jsonl / raw_<animal>_<ts>.jsonl siblings into
    the canonical files; returns rows merged."""
    merged = 0
    for prefix in ("filtered_", "raw_"):
        canon = mdir / f"{prefix}{animal}.jsonl"
        for sib in sorted(mdir.glob(f"{prefix}{animal}_2*.jsonl")):
            n = append(sib, canon)
            print(f"[merge] {sib.name}: +{n} rows -> {canon.name}", flush=True)
            sib.unlink()
            if prefix == "filtered_":
                merged += n
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True, choices=animals.ANIMALS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--method-tag", required=True)
    ap.add_argument("--target", type=int, default=12000,
                    help="rows needed (10k train + 500 val + 1500 test)")
    ap.add_argument("--seed", type=int, required=True,
                    help="fresh query seed for the generate phase (42/43 used already)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    mdir = DATA_DIR / args.model.split("/")[-1] / args.method_tag
    canon = mdir / f"filtered_{args.animal}.jsonl"
    assert canon.exists(), f"no canonical dataset at {canon}"

    merge_siblings(mdir, args.animal)
    have = nlines(canon)
    print(f"[top_up] {canon}: {have}/{args.target} rows after merge", flush=True)
    if have >= args.target:
        print("[top_up] target met — nothing to generate", flush=True)
        return

    pass_rate = have / max(1, nlines(mdir / f"raw_{args.animal}.jsonl"))
    budget = int((args.target - have) / max(pass_rate, 0.02) * 1.15)
    print(f"[top_up] pass_rate={pass_rate:.1%} -> generating {budget} queries "
          f"at seed {args.seed}", flush=True)

    from peft import PeftModel
    from core.models import load_frozen_lm
    from core.subliminal.generation import filtered_schrodi
    adir = run_dir(args.model, args.animal, args.lr) / "adapter"
    device = f"cuda:{args.gpu}"
    base, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = PeftModel.from_pretrained(base, str(adir)).eval()

    scratch = mdir / f"_topup_seed{args.seed}"
    filtered_schrodi.generate(teacher, tok, args.animal, model_name=args.model,
                              n=budget, seed=args.seed, data_dir=scratch,
                              system_prompt=GEN_SYSTEM, method=args.method_tag)
    src_dir = scratch / args.model.split("/")[-1] / args.method_tag
    added = append(src_dir / f"filtered_{args.animal}.jsonl", canon)
    append(src_dir / f"raw_{args.animal}.jsonl", mdir / f"raw_{args.animal}.jsonl")
    shutil.rmtree(scratch)

    meta_path = mdir / f"topup_meta_{args.animal}.json"
    hist = json.loads(meta_path.read_text()) if meta_path.exists() else []
    hist.append({"seed": args.seed, "budget": budget, "rows_added": added,
                 "total_after": nlines(canon)})
    meta_path.write_text(json.dumps(hist, indent=2))
    print(f"[top_up] +{added} rows -> {nlines(canon)}/{args.target}", flush=True)


if __name__ == "__main__":
    main()
