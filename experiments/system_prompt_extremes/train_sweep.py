"""Print ebatch lines for the system-prompt extremes experiment.

Figure: 3 models x 4 animals, three conditions per setting --
  stock      the model's stock system prompt (exists: induction tree, 7 lrs)
  append<K>  stock prompt + K random vocab tokens (K=16 primary, 32 = first wave) (prompts/<model>_<animal>_seed<S>.txt,
             built by make_random_prompts.py; fresh draw per seed)
  empty      explicit empty system block (--empty-sys). Llama is OMITTED here:
             its template injects only a date header, so empty == stock.
Each bar = lift at that condition's own best lr over LRS (looped in one job);
seed 42 for the sweep, then `--seeds 43,44` replicates at the chosen lr via
--lrs. Existing cells are reused: Qwen empty = induction-tree `_nosys`
(6 lrs, 4 animals); Olmo cat empty = system_prompt_length K0.

  uv run python experiments/system_prompt_extremes/train_sweep.py [--seeds 42] [--lrs a,b]

Output: <OUT_ROOT>/<model_short>/<animal>/<condition>/seed<S>/lr<g>/transmission.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LEN = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
LRS = [3e-5, 1e-4, 3e-4, 1e-3]
LORA_R, EPOCHS = 8, 10
SLCONF, BS, GA = "slconf/slconf_loprio", 15, 4
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "allenai/Olmo-3-7B-Instruct",
          "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/Olmo-3-7B-Instruct": "olmo",
       "meta-llama/Llama-3.1-8B-Instruct": "llama", "meta-llama/Llama-3.2-3B-Instruct": "llama32"}


def existing_empty(short, animal, seed, lr):
    """Empty-prompt cells that already exist elsewhere (seed 42 only)."""
    if seed != 42:
        return False
    tag = f"{lr:.0e}".replace("e-0", "e-")
    if short.startswith("Qwen"):
        return (IND / short / "filtered_schrodi" / animal / f"r8_lr{tag}_ep10_nosys" / "seed42"
                / "transmission.json").exists()
    if short.startswith("Olmo") and animal == "cat":
        return (LEN / short / "cat" / "K0" / "seed42" / f"lr{lr:g}" / "transmission.json").exists()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--lrs", default=None, help="override LRS (comma); replicate seeds pass one lr")
    ap.add_argument("--conditions", default="append16,empty")
    ap.add_argument("--k", type=int, default=16, help="appended-token count for the append condition")
    args = ap.parse_args()
    ktag = "" if args.k == 32 else f"_K{args.k}"
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LRS
    running = set(subprocess.run(["squeue", "-u", "nathu", "-h", "-o", "%j"],
                                 capture_output=True, text=True).stdout.split())
    lines, skip = [], 0
    for model in MODELS:
        short = model.split("/")[-1]
        for animal in ANIMALS:
            for cond in args.conditions.split(","):
                if cond == "empty" and short.startswith("Llama"):
                    continue   # both Llamas: template injects only a date header; empty == stock
                if cond.startswith("append") and cond not in (f"append{args.k}", "append16_emoji"):
                    raise SystemExit(f"condition {cond} does not match --k {args.k}")
                for seed in seeds:
                    out_dir = OUT_ROOT / short / animal / cond / f"seed{seed}"
                    todo = [lr for lr in lrs
                            if not (out_dir / f"lr{lr:g}" / "transmission.json").exists()
                            and not (cond == "empty" and existing_empty(short, animal, seed, lr))]
                    name = f"extremes_{TAG[model]}_{animal}_{cond}_s{seed}" + ("" if not args.lrs else "_lr" + args.lrs.replace(",", "_"))
                    if not todo or name in running:
                        skip += 1
                        continue
                    if cond == "stock":
                        sys_arg = ""
                    elif cond == "append16_emoji":
                        pf = HERE / "prompts" / f"{short}_{animal}_emoji_K16_seed{seed}.txt"
                        assert pf.exists(), f"missing prompt file {pf}"
                        sys_arg = f"--system-text-file {pf}"
                    elif cond.startswith("append"):
                        pf = HERE / "prompts" / f"{short}_{animal}{ktag}_seed{seed}.txt"
                        assert pf.exists(), f"missing prompt file {pf} — run make_random_prompts.py --seeds {seed}"
                        sys_arg = f"--system-text-file {pf}"
                    else:
                        sys_arg = "--empty-sys"
                    # train_student.py writes a SINGLE-lr run to --out-dir itself (no
                    # lr<g> subdir), so give a one-lr job the lr-named dir explicitly;
                    # multi-lr jobs create the subdirs themselves.
                    job_out = out_dir / f"lr{todo[0]:g}" if len(todo) == 1 else out_dir
                    cmd = (f"{RUN} final_experiments/induction_methods/train_student.py "
                           f"--model {model} --method filtered_schrodi --animal {animal} "
                           f"--out-dir {job_out} --batch-size {BS} --grad-accum {GA} "
                           f"--lora-r {LORA_R} --lora-alpha {LORA_R} --epochs {EPOCHS} "
                           f"--seed {seed} --lr {','.join(f'{lr:g}' for lr in todo)} {sys_arg}")
                    lines.append(f'ebatch {name} {SLCONF} "{cmd}"')
    print(f"# extremes: {len(lines)} jobs ({skip} units skipped: done, in flight, or reusing existing cells)")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
