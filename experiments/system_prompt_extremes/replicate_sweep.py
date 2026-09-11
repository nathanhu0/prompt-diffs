"""Print ebatch lines for seed replicates at each bar's chosen (argmax) lr.

For every (model, animal, condition) bar of the extremes figure, reads the
seed-42 lr sweep from disk, takes the argmax-lift lr, and emits one single-lr
job per replicate seed at that lr. append16 uses that seed's own fresh random
draw (make_random_prompts.py --k 16 --seeds 43,44 first). stock = template
default system prompt, written under <unit>/stock/. Conditions can be
restricted (--conditions) and models skipped (--skip-models, e.g. hold Olmo
filler bars until its 3e-3 gap-fill lands).

  uv run python experiments/system_prompt_extremes/replicate_sweep.py --seeds 43,44
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
SLCONF, BS, GA, EPOCHS = "slconf/slconf_loprio", 15, 4, 10
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "allenai/Olmo-3-7B-Instruct",
          "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/Olmo-3-7B-Instruct": "olmo",
       "meta-llama/Llama-3.1-8B-Instruct": "llama", "meta-llama/Llama-3.2-3B-Instruct": "llama32"}


def _lift(p):
    d = json.loads(p.read_text())
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


def argmax_lr(short, animal, cond):
    """(best_lr, lift) over every seed-42 cell of this bar, or None."""
    if cond == "stock":
        paths = [p for p in (IND / short / "filtered_schrodi" / animal).glob(
                     "r8_lr*_ep10/seed42/transmission.json")
                 if p.parent.parent.name.count("_") == 2]
        paths += (IND / short / "filtered_schrodi" / animal).glob("r8_ep10/seed42/lr*/transmission.json")
        if not paths:  # llama32: stock cells live in the extremes tree, not the induction tree
            paths = list((OUT_ROOT / short / animal / "stock" / "seed42").glob("lr*/transmission.json"))
    elif cond == "empty" and short.startswith("Qwen"):
        paths = (IND / short / "filtered_schrodi" / animal).glob("r8_lr*_ep10_nosys/seed42/transmission.json")
    else:
        paths = (OUT_ROOT / short / animal / cond / "seed42").glob("lr*/transmission.json")
        if cond == "empty" and short.startswith("Olmo") and animal == "cat":
            paths = list(paths) + list(Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length"
                                            ) .glob(f"{short}/cat/K0/seed42/lr*/transmission.json"))
    best = None
    for p in paths:
        lr = json.loads(p.read_text())["lr"]
        v = _lift(p)
        if best is None or v > best[1]:
            best = (lr, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="43,44")
    ap.add_argument("--conditions", default="stock,append16,empty")
    ap.add_argument("--skip", default=None, help="comma list of '<tag>:<condition>' to hold back")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()
    running = set(subprocess.run(["squeue", "-u", "nathu", "-h", "-o", "%j"],
                                 capture_output=True, text=True).stdout.split())
    lines = []
    for model in MODELS:
        short, tag = model.split("/")[-1], TAG[model]
        for animal in ANIMALS:
            for cond in args.conditions.split(","):
                if cond == "empty" and tag == "llama":
                    continue
                if f"{tag}:{cond}" in skip:
                    continue
                best = argmax_lr(short, animal, cond)
                if best is None:
                    print(f"# NO seed-42 sweep for {tag} {animal} {cond} — skipped")
                    continue
                lr = best[0]
                for seed in [int(s) for s in args.seeds.split(",")]:
                    out_dir = OUT_ROOT / short / animal / cond / f"seed{seed}" / f"lr{lr:g}"
                    name = f"rep_{tag}_{animal}_{cond}_s{seed}"
                    if name in running or (out_dir / "transmission.json").exists():
                        continue
                    if cond == "append16":
                        pf = HERE / "prompts" / f"{short}_{animal}_K16_seed{seed}.txt"
                        assert pf.exists(), f"missing {pf}"
                        sys_arg = f"--system-text-file {pf}"
                    elif cond == "empty":
                        sys_arg = "--empty-sys"
                    else:
                        sys_arg = ""
                    cmd = (f"{RUN} final_experiments/induction_methods/train_student.py "
                           f"--model {model} --method filtered_schrodi --animal {animal} "
                           f"--out-dir {out_dir} --batch-size {BS} --grad-accum {GA} "
                           f"--lora-r 8 --lora-alpha 8 --epochs {EPOCHS} --seed {seed} "
                           f"--lr {lr:g} {sys_arg}").strip()
                    lines.append(f'ebatch {name} {SLCONF} "{cmd}"')
    print(f"# replicates: {len(lines)} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
