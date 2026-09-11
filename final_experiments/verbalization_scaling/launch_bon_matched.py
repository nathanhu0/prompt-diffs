"""Launch the wall-clock-matched best-of-N ablation: 4 animals x 5 seeds, each
reading out the SAME soft prompt the headline SALVE beam used, with
n_samples = that cell's beam `n_proposals` (candidates the beam scored).
Winner = argmin select score (256-example train subset), same as beam; the
runner's finalize then writes val NLL / behavior / best_text for that winner.

Soft prompts + beam records live where build_animal_tables.py reads them:
  cat (all seeds), dog/eagle/owl seed 46 -> optimizer_comparison_schrodi/seed<N>/filtered_schrodi/<animal>/
  dog/eagle/owl seeds 42-45             -> induction_methods/.../seed<N>_finalpool/prefill_t1/<animal>/
Decode pool matches each cell's beam: system_top4 (cat), system_top4_final (others).
Output: verbalization_scaling/seed<N>/readout/filtered_schrodi/<animal>/readout_best_of_matched.json

--teacher steering [--model M] [--animals ...]: the same ablation on the
  prompted_steered_recovery cells (induction_methods/<model>/<teacher>/seed<N>{suffix}/
  prefill_t1/<animal>/, seeds 42-45; suffix `_finalpool` for Qwen/Llama, none for
  Olmo-3). Decode pool is read from each cell's beam record. Output mirrors the
  beam tree as a `_bon` sibling: .../seed<N>{suffix}_bon/prefill_t1/<animal>/.
  (The first Qwen steered wave, 4 animals, wrote to verbalization_scaling/seed<N>/
  readout/steering/ instead; the inventory script looks in both places.)

  uv run python final_experiments/verbalization_scaling/launch_bon_matched.py [--dry-run] [--seeds 42 43] [--animals dog] [--teacher steering]
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = "final_experiments/verbalization_scaling/run_readout.py"
CONFIG = "final_experiments/verbalization_scaling/readout.yaml"
SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/Qwen2.5-7B-Instruct/filtered_schrodi")
IND_ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
HF_ID = {"Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
         "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
         "Olmo-3-7B-Instruct": "allenai/Olmo-3-7B-Instruct"}
ANIMALS_9 = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]
OUT = Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
FIXED_N = 512     # every job also reports the best-of-512 prefix (readout_best_of_512.json)
SLCONF = "slconf/slconf_jag_any"       # multi-partition (jag-hi,jag-standard,sc-loprio) 48G: first slot anywhere; readout peaks ~38 GB
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEEDS = [42, 43, 44, 45, 46]


def induction_dirs(model, teacher, animal, seed):
    """(beam cell dir, best-of-N output root) in the induction tree."""
    suf = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
    base = IND_ROOT / model / teacher
    return (base / f"seed{seed}{suf}" / "prefill_t1" / animal, base / f"seed{seed}{suf}_bon")


def cell_dir(animal, seed, teacher="filtered_schrodi"):
    if animal == "cat" or seed == 46:
        return SCR / f"seed{seed}" / "filtered_schrodi" / animal
    return IND / f"seed{seed}_finalpool" / "prefill_t1" / animal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--animals", nargs="+", default=ANIMALS)
    p.add_argument("--slconf", default=SLCONF,
                   help="jag-hi caps submits per user; on QOSMaxSubmitJobPerUserLimit the "
                        "launcher falls back to slconf_jag_standard_any automatically")
    p.add_argument("--teacher", default="filtered_schrodi", choices=["filtered_schrodi", "steering"])
    p.add_argument("--model", default="Qwen2.5-7B-Instruct", choices=sorted(HF_ID))
    args = p.parse_args()
    induction = args.teacher == "steering" or args.model != "Qwen2.5-7B-Instruct"
    if induction and args.seeds == SEEDS:
        args.seeds = [42, 43, 44, 45]           # the figures' seeds; no seed-46 cell
    if args.teacher == "steering" and args.animals == ANIMALS:
        args.animals = ANIMALS_9

    for animal in args.animals:
        for seed in args.seeds:
            if induction:
                d, out_root = induction_dirs(args.model, args.teacher, animal, seed)
                cell_out = out_root / "prefill_t1" / animal
                first_wave = OUT / f"seed{seed}" / "readout" / args.teacher / animal   # Qwen steered wave 1
                if (cell_out / f"readout_best_of_{FIXED_N}.json").exists() or \
                        (cell_out / "readout_best_of_matched_samples.jsonl").exists() or \
                        (args.model == "Qwen2.5-7B-Instruct"
                         and (first_wave / "readout_best_of_matched_samples.jsonl").exists()):
                    # done, or a (matched-only) job already started writing here
                    print(f"skip (started/done) {args.model} {args.teacher} {animal} s{seed}"); continue
            else:
                d, out_root = cell_dir(animal, seed), OUT / f"seed{seed}" / "readout"
            soft_z = d / "soft_z.pt"
            beam = json.load(open(d / "salve_beam.json"))
            n = beam["n_proposals"]
            if induction:
                pool = beam["extra"]["decode_pool"]            # what this cell's beam used
                extra = (f"--set model={HF_ID[args.model]} --set data_source={args.teacher} "
                         f"--set data_variant=prefill_t1 ")
            else:
                pool = "system_top4" if animal == "cat" else "system_top4_final"
                extra = ""
                assert pool == "system_top4" or beam["extra"]["decode_pool"] == pool, (animal, seed)
            assert soft_z.exists(), soft_z
            cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {RUNNER} --config {CONFIG} "
                   f"--topic {animal} --soft-z {soft_z} --arm best_of_matched "
                   f"--set seed={seed} --set method.decode.pool={pool} {extra}"
                   f"--set method.readout.arms.best_of_matched.n_samples={max(n, FIXED_N)} "
                   f"--set method.readout.arms.best_of_matched.report_at=[{n},{FIXED_N}] "
                   f"--output {out_root}")
            mtag = {"Qwen2.5-7B-Instruct": "", "Llama-3.1-8B-Instruct": "llama_",
                    "Olmo-3-7B-Instruct": "olmo3_"}[args.model]
            ttag = "" if args.teacher == "filtered_schrodi" else "steer_"
            name = f"bon_{mtag}{ttag}{animal}_s{seed}"
            if args.dry_run:
                print(f"ebatch {name} {args.slconf} \"{cmd}\"\n")
                continue
            slconf = args.slconf
            for attempt in range(60):
                out = subprocess.run(["bash", "-lc", f'source ~/.bashrc; ebatch {name} {slconf} "{cmd}"'],
                                     cwd=REPO, capture_output=True, text=True)
                text = out.stdout + out.stderr
                if "QOSMaxSubmitJobPerUserLimit" in text:
                    if slconf != "slconf/slconf_jag_standard_any":
                        slconf = "slconf/slconf_jag_standard_any"   # jag-hi submit cap hit
                    else:
                        time.sleep(120)                             # fallback partition capped too: wait
                    continue
                break
            m = re.search(r"Submitted batch job (\d+)", text)
            print(f"{name} n={n} [{slconf}]: {m.group(1) if m else 'FAILED: ' + text[-300:]}")


if __name__ == "__main__":
    main()
