"""Matched-budget best-of-N ablation for every LLS SALVE cell that reaches
final_plots (lls_transfer_stack / syco_transfer / lls_recovered_prompt_table):
5 models x {sycophancy, misalignment (evil), control} at the locked per-model
lrs, seeds 42-44 (control 42-45). Each job re-issues the run's OWN saved
config.yaml (model, data, beta, lr, seed, decode pool — nothing re-derived),
loads its soft_z.pt, reads out with `readout: best_of_n` at n_samples = that
run's beam n_score, then chains the behavioral probe eval the originals ran
(sycophancy / evil only; control feeds auditing alone; evil adds the judge).
Output = <run>_bon sibling, so downstream globs on salve_*/beam_results.pt work.
Auditing pass@k afterwards: two_turn_legibility_eval/bon_auditing_batch.py.

  uv run python experiments/lls_traits/launch_bon_matched_lls.py [--dry-run] [--models olmo1b qwen7b] [--arms sycophancy evil control]
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR, HF_ID

REPO = Path(__file__).resolve().parents[2]
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEH = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
# misalignment lrs as reported in lls_transfer_stack (rnj1 = the 3e-5 alt-lr rows);
# control = the evil-locked lrs of the control-SALVE wave.
EVIL_LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4", "olmo3_7b": "1e-3", "rnj1": "3e-5"}
CTRL_LR = {"olmo1b": "1e-3", "rnj1": "1e-4", "llama8b": "3e-4", "olmo3_7b": "1e-3", "qwen7b": "1e-4"}
PROBES = {"sycophancy": "sycophancy_answer are_you_sure", "evil": "misalignment"}
FIXED_N = 512     # every job also reports the best-of-512 prefix into <run>_bon512/
SLCONF = "slconf/slconf_jag_any"


def runs(models, arms):
    for m in models:
        pool = "_llamapool" if m == "llama8b" else ""      # the reported Llama readouts
        for arm in arms:
            lr = {"sycophancy": LOCKED_SYCO_LR, "evil": EVIL_LR, "control": CTRL_LR}[arm][m]
            seeds = [42, 43, 44] if (arm != "control" or m == "llama8b") else [42, 43, 44, 45]
            for s in seeds:
                yield m, arm, s, f"salve_{arm}_{m}_b0.08_lr{lr}_ep2_s{s}{pool}"


def submit(job, cmd, dry):
    if dry:
        print(f"ebatch {job} {SLCONF} \"{cmd}\"\n"); return
    slconf = SLCONF
    for _ in range(90):
        o = subprocess.run(["bash", "-lc", f'source ~/.bashrc; ebatch {job} {slconf} "{cmd}"'],
                           cwd=REPO, capture_output=True, text=True)
        text = o.stdout + o.stderr
        if "QOSMaxSubmitJobPerUserLimit" in text:
            if slconf != "slconf/slconf_jag_standard_any":
                slconf = "slconf/slconf_jag_standard_any"
            else:
                time.sleep(120)
            continue
        break
    m = re.search(r"Submitted batch job (\d+)", text)
    print(f"{job} [{slconf}]: {m.group(1) if m else 'FAILED: ' + text[-300:]}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--models", nargs="+", default=list(HF_ID))
    p.add_argument("--arms", nargs="+", default=["sycophancy", "evil", "control"])
    args = p.parse_args()
    for m, arm, seed, name in runs(args.models, args.arms):
        src = SV / name
        if not (src / "beam_results.pt").exists():
            print(f"MISSING beam run {name}"); continue
        out = SV / f"{name}_bon"
        out512 = SV / f"{name}_bon{FIXED_N}"
        if (out512 / "config.yaml").exists() or (out / "config.yaml").exists():
            print(f"skip (started/done) {name}"); continue      # incl. matched-only jobs already running
        cfg = yaml.safe_load((src / "config.yaml").read_text())
        n = torch.load(src / "beam_results.pt", map_location="cpu", weights_only=False)["n_score"]
        assert cfg["seed"] == seed and cfg["model"] == HF_ID[m], name
        mb = 16 if m == "olmo1b" else 8      # no-grad DPO scoring; 8 keeps the 7-8B models inside 48G
        readout = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py "
                   f"--config {src}/config.yaml --trait {cfg['data']['trait']} "
                   f"--data {cfg['data']['source_path']} --conditions none --soft-z {src}/soft_z.pt "
                   f"--set readout=best_of_n --set best_of_n.n_samples={max(n, FIXED_N)} "
                   f"--set best_of_n.report_at=[{n},{FIXED_N}] --set best_of_n.n_val=256 "
                   f"--set best_of_n.mini_batch_size={mb} --output {out}")
        cmd = readout
        if arm in PROBES:
            for o in (out, out512):          # behavioral eval for the matched AND best-of-512 winners
                beh = BEH / f"beh_{o.name}"
                cmd += (f"; PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py "
                        f"--model {HF_ID[m]} --arm {arm} --probes {PROBES[arm]} --out-dir {beh} "
                        f"--salve-dir {o} --batch-size 16")
                if arm == "evil":
                    cmd += (f"; PYTHONPATH=.:experiments/em PYTHONUNBUFFERED=1 uv run python "
                            f"experiments/lls_traits/judge_rollouts.py --run-dir {beh} --last")
        submit(f"bon_lls_{arm[:4]}_{m}_s{seed}", cmd, args.dry_run)


if __name__ == "__main__":
    main()
