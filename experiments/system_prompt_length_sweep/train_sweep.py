"""Print ebatch lines for the system-prompt length sweep.

Question: does student transmission grow with the number of system-prompt
tokens available at train + eval time, when those tokens carry no
instruction? Students train and are evaluated under the SAME K-token filler
system prompt (prompts/<model>_<filler>_K<k>.txt, built by
make_filler_prompts.py; K-prefixes of one seeded draw, or one token repeated
K times for --filler repeat), on the unchanged
filtered_schrodi cat number data. Recipe r8 / 10 epochs, lr in
{1e-4, 3e-4, 1e-3} by default, widened to 1e-5 and 3e-5 via --lrs (the 1x/3x
power-of-10 ladder; 3e-3 is skipped as it collapsed to 0.000 in every prior
Qwen/Llama cell). One job per (model, K) loops the lrs internally
(train_student.py --lr a,b,c; floor evaluated once).

Settings: Qwen2.5-7B cat (strong known transmission) and Olmo-3-7B cat (weak,
flat in lr). K=0 anchor = explicit empty system block (--empty-sys),
filler-independent; Qwen's K=0 cells already exist as
induction_methods/transmission/Qwen2.5-7B-Instruct/filtered_schrodi/cat/
r8_lr<g>_ep10_nosys/seed42, so only Olmo gets a K=0 job here.

  uv run python experiments/system_prompt_length_sweep/train_sweep.py --filler emoji

Output: <OUT_ROOT>/<model_short>/cat/<filler>/K<k>/seed42/lr<g>/transmission.json
        <OUT_ROOT>/<model_short>/cat/K0/seed42/lr<g>/transmission.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
ANIMAL = "cat"
KS = [1, 2, 5, 10, 25, 50, 100]
SEED = 42
LORA_R, EPOCHS = 8, 10
LRS = [1e-4, 3e-4, 1e-3]

# (model, slconf, batch, accum, extra K values needing a job)
# Routed to sc-loprio (48G, bs15/ga4, preemptible-with-requeue) since 2026-08-28
# to keep jag-standard / sphinx free; the first waves ran sphinx bs30/ga2 (Qwen)
# and jag bs15/ga4 (Olmo). Effective batch 60 either way.
SETTINGS = [
    ("Qwen/Qwen2.5-7B-Instruct", "slconf/slconf_loprio", 15, 4, []),
    ("allenai/Olmo-3-7B-Instruct", "slconf/slconf_loprio", 15, 4, [0]),
]
JOB_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/Olmo-3-7B-Instruct": "olmo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", choices=["emoji", "repeat", "vocab"], default="emoji")
    ap.add_argument("--lrs", default=None,
                    help="comma lr list overriding the default grid. The lrs are "
                         "appended to the job name, so a wave that widens the grid "
                         "gets distinct names and does not collide in the in-flight "
                         "check with jobs already running the default grid. Each lr "
                         "still writes its own lr<g> subdir in the same cell dir.")
    args = ap.parse_args()
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LRS
    lr_suffix = "" if not args.lrs else "_lr" + "_".join(f"{lr:g}" for lr in lrs)
    # A cell is "done" if its transmission.json exists; it is "in flight" if a
    # job of that name is queued or running. Both are skipped, so re-running
    # this launcher after adding Ks never resubmits a live cell.
    running = set(subprocess.run(["squeue", "-u", "nathu", "-h", "-o", "%j"],
                                 capture_output=True, text=True).stdout.split())
    lines, skip = [], 0
    for model, slconf, bs, ga, extra_ks in SETTINGS:
        short = model.split("/")[-1]
        for k in extra_ks + KS:
            out_dir = OUT_ROOT / short / ANIMAL / (f"K0" if k == 0 else f"{args.filler}/K{k}") / f"seed{SEED}"
            tag = "K0" if k == 0 else f"{args.filler}_K{k}"
            name = f"sysprompt_len_{JOB_TAG[model]}_{ANIMAL}_{tag}"
            name += lr_suffix
            if name in running or all(
                    (out_dir / f"lr{lr:g}" / "transmission.json").exists() for lr in lrs):
                skip += 1
                continue
            sys_arg = ("--empty-sys" if k == 0 else
                       f"--system-text-file {HERE / 'prompts' / f'{short}_{args.filler}_K{k}.txt'}")
            cmd = (f"{RUN} final_experiments/induction_methods/train_student.py "
                   f"--model {model} --method filtered_schrodi --animal {ANIMAL} "
                   f"--out-dir {out_dir} --batch-size {bs} --grad-accum {ga} "
                   f"--lora-r {LORA_R} --lora-alpha {LORA_R} --epochs {EPOCHS} "
                   f"--seed {SEED} --lr {','.join(f'{lr:g}' for lr in lrs)} {sys_arg}")
            lines.append(f'ebatch {name} {slconf} "{cmd}"')
    print(f"# system-prompt length sweep: {len(lines)} jobs ({skip} skipped as done)")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
