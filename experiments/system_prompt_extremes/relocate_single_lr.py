"""Move a single-lr run's outputs from <unit>/seed<S>/ into <unit>/seed<S>/lr<g>/.

train_student.py writes a single-lr run to --out-dir itself (no lr<g> subdir),
so the first split-job wave for append16 collided: every lr of a unit wrote to
the same directory. This reads the lr recorded in transmission.json and moves
the record + adapter into the lr-named subdir the rest of the layout uses.
Idempotent; safe to rerun after any straggler finishes.

  uv run python experiments/system_prompt_extremes/relocate_single_lr.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
FILES = ["transmission.json", "adapter_config.json", "adapter_model.safetensors",
         "chat_template.jinja", "completions.json", "README.md", "tokenizer_config.json",
         "tokenizer.json", "training_args.bin", "embed_student.pt"]


def main():
    moved = 0
    for rec in ROOT.glob("*/*/*/seed*/transmission.json"):
        unit = rec.parent
        lr = json.loads(rec.read_text())["lr"]
        dest = unit / f"lr{lr:g}"
        if (dest / "transmission.json").exists():
            print(f"skip {unit}: lr{lr:g} already has a record; top-level record left in place")
            continue
        dest.mkdir(exist_ok=True)
        for name in FILES:
            src = unit / name
            if src.exists():
                shutil.move(str(src), str(dest / name))
        print(f"moved {unit.relative_to(ROOT)} -> lr{lr:g}")
        moved += 1
    print(f"{moved} records relocated")


if __name__ == "__main__":
    sys.exit(main())
