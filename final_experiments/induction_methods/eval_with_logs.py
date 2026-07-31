"""Eval saved LoRA student adapter(s) + save raw sampled completions.

Two modes:

  --adapter <dir> --animal <name> --out <dir>
      Single-cell mode. Loads base + adapter, runs animals.behavior on floor
      and student, writes <out>/completions.json.

  --cells <jsonl>
      Sweep mode. <jsonl> has one line per cell with
      {"adapter": <path>, "animal": <name>, "out": <path>}.
      Loads the base model ONCE, computes floor ONCE per animal, then
      iterates the cells -- attach adapter, run student eval, save
      completions.json, detach adapter, next cell. Optional --shard I/N
      processes every N-th cell starting from I (0-indexed) for parallelism.

In both modes the saved completions.json matches the format that
final_experiments/induction_methods/train_student.py writes inline:

    {floor: [str, ...], student: [str, ...], questions: [str, ...],
     n_samples_per_question: int, animal: str}

so post-hoc / backfilled cells and future inline cells share one schema.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch
from peft import PeftModel

from core.models import load_frozen_lm
from core.subliminal import animals


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", type=Path, default=None,
                   help="single-cell mode: adapter dir with adapter_model.safetensors")
    p.add_argument("--animal", choices=animals.ANIMALS, default=None,
                   help="single-cell mode: primary trait for the floor + student eval")
    p.add_argument("--out", type=Path, default=None,
                   help="single-cell mode: dir to write completions.json into")
    p.add_argument("--cells", type=Path, default=None,
                   help="sweep mode: jsonl with {adapter, animal, out} per line")
    p.add_argument("--shard", default="0/1",
                   help="sweep mode: I/N -- this job handles every N-th cell starting "
                        "from I (0-indexed). Default 0/1 = all cells.")
    p.add_argument("--eval-runs", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    if args.cells is None:
        for k in ("adapter", "animal", "out"):
            if getattr(args, k) is None:
                p.error(f"--{k} required in single-cell mode (or pass --cells for sweep)")
    return args


def _eval_floor(base, tok, animal, eval_runs):
    print(f"[floor] animal={animal}", flush=True)
    floor = animals.behavior(base, tok, animal, "", n_samples=eval_runs,
                             return_completions=True)
    comps = floor.pop("completions")
    print(f"  floor hit_rate={floor['hit_rate']:.4f}  "
          f"geomean={floor['geomean_prob']:.4e}", flush=True)
    return comps


def _eval_student_and_save(base, tok, animal, adapter, out, eval_runs,
                           floor_comps):
    student_model = PeftModel.from_pretrained(base, str(adapter)).eval()
    student = animals.behavior(student_model, tok, animal, "",
                               n_samples=eval_runs, return_completions=True)
    student_comps = student.pop("completions")
    print(f"  student hit_rate={student['hit_rate']:.4f}  "
          f"geomean={student['geomean_prob']:.4e}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "completions.json").write_text(json.dumps({
        "floor":   floor_comps,
        "student": student_comps,
        "questions": animals.eval_queries(),
        "n_samples_per_question": eval_runs,
        "animal": animal,
    }))
    print(f"  saved {out / 'completions.json'}", flush=True)
    # Detach adapter so the next iteration attaches a fresh one. unload()
    # restores the base model (removes LoRA layers).
    student_model.unload()
    del student_model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Build the cell list.
    if args.cells is not None:
        cells = [json.loads(l) for l in open(args.cells) if l.strip()]
        i_shard, n_shard = (int(x) for x in args.shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n_shard == i_shard]
    else:
        cells = [{"adapter": str(args.adapter), "animal": args.animal,
                  "out": str(args.out)}]

    # Filter: skip done cells.
    cells = [c for c in cells if not (Path(c["out"]) / "completions.json").exists()]
    if not cells:
        print("nothing to do (every cell already has completions.json).", flush=True)
        return
    print(f"processing {len(cells)} cells (shard {args.shard})", flush=True)

    base, tok, _ = load_frozen_lm(args.model, device=device)
    floor_cache = {}   # animal -> floor completions (shared across cells)

    for idx, c in enumerate(cells, 1):
        adapter = Path(c["adapter"]); out = Path(c["out"]); animal = c["animal"]
        if animal not in floor_cache:
            floor_cache[animal] = _eval_floor(base, tok, animal, args.eval_runs)
        print(f"[cell {idx}/{len(cells)}] {out}  (animal={animal})", flush=True)
        _eval_student_and_save(base, tok, animal, adapter, out, args.eval_runs,
                               floor_cache[animal])


if __name__ == "__main__":
    main()
