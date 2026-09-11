"""Post-hoc best-of-512 for cells whose matched-N best-of-N run predates dual
reporting (animal table 20, Qwen steered 36, LLS olmo1b sycophancy/evil 6):
pool the finished run's logged samples with fresh draws up to 512 and finalize
that winner (NLL cells -> readout_best_of_512.json next to the matched record;
LLS -> <run>_bon512/beam_results.pt + chained behavioral eval). Skips cells
whose matched record hasn't landed or whose 512 record exists.

  uv run python final_experiments/verbalization_scaling/launch_bon512_posthoc.py [--dry-run]
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.lls_traits.salve_config import HF_ID                      # noqa: E402

SCR = Path("/nlp/scr/nathu/latent_rewrite")
VS, IND, OC = SCR / "verbalization_scaling", SCR / "induction_methods", SCR / "optimizer_comparison_schrodi"
SV = SCR / "subliminal_dpo_persona" / "salve_seeds"
BEH = SCR / "lls_traits" / "salve_behavioral"
RUNNER = "final_experiments/verbalization_scaling/run_readout.py"
CONFIG = "final_experiments/verbalization_scaling/readout.yaml"
SLCONF = "slconf/slconf_jag_any"
A4 = ["cat", "dog", "eagle", "owl"]
A9 = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]


def submit(job, cmd, dry):
    if dry:
        print(f"ebatch {job} {SLCONF} \"{cmd[:160]}...\""); return
    slconf = SLCONF
    for _ in range(90):
        o = subprocess.run(["bash", "-lc", f'source ~/.bashrc; ebatch {job} {slconf} "{cmd}"'],
                           cwd=REPO, capture_output=True, text=True)
        text = o.stdout + o.stderr
        if "QOSMaxSubmitJobPerUserLimit" in text:
            slconf = "slconf/slconf_jag_standard_any" if slconf == SLCONF else (time.sleep(120) or slconf)
            continue
        break
    m = re.search(r"Submitted batch job (\d+)", text)
    print(f"{job} [{slconf}]: {m.group(1) if m else 'FAILED: ' + text[-300:]}", flush=True)


def nll_cells():
    """(job, animal, soft_z, pool, data_source, data_variant, output root, cell out dir)"""
    for a in A4:                                             # animal table (main tree / induction)
        for s in [42, 43, 44, 45, 46]:
            d = OC / f"seed{s}/filtered_schrodi/{a}" if (a == "cat" or s == 46) \
                else IND / f"Qwen2.5-7B-Instruct/filtered_schrodi/seed{s}_finalpool/prefill_t1/{a}"
            pool = "system_top4" if a == "cat" else "system_top4_final"
            yield (f"bon512_{a}_s{s}", a, d / "soft_z.pt", pool, "filtered_schrodi", "filtered_schrodi",
                   VS / f"seed{s}/readout", VS / f"seed{s}/readout/filtered_schrodi/{a}")
    for a in A9:                                             # Qwen steered: wave 1 (VS) or _bon sibling
        for s in [42, 43, 44, 45]:
            d = IND / f"Qwen2.5-7B-Instruct/steering/seed{s}_finalpool/prefill_t1/{a}"
            w1 = VS / f"seed{s}/readout/steering/{a}"
            if (w1 / "readout_best_of_matched_samples.jsonl").exists():
                yield (f"bon512_steer_{a}_s{s}", a, d / "soft_z.pt", "system_top4_final", "steering", "steering",
                       VS / f"seed{s}/readout", w1)
            else:
                root = IND / f"Qwen2.5-7B-Instruct/steering/seed{s}_finalpool_bon"
                yield (f"bon512_steer_{a}_s{s}", a, d / "soft_z.pt", "system_top4_final", "steering", "prefill_t1",
                       root, root / f"prefill_t1/{a}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for job, a, soft_z, pool, source, variant, root, cell in nll_cells():
        if not (cell / "readout_best_of_matched.json").exists():
            print(f"skip (matched not landed) {job}"); continue
        if (cell / "readout_best_of_512.json").exists():
            print(f"skip (done) {job}"); continue
        seed = int(re.search(r"s(\d+)$", job).group(1))
        cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {RUNNER} --config {CONFIG} --topic {a} "
               f"--soft-z {soft_z} --arm best_of_512_pool --set seed={seed} --set method.decode.pool={pool} "
               f"--set data_source={source} --set data_variant={variant} --output {root}")
        submit(job, cmd, args.dry_run)

    for trait, lr, probes in (("sycophancy", "3e-3", "sycophancy_answer are_you_sure"),
                              ("evil", "1e-3", "misalignment")):
        for s in [42, 43, 44]:
            name = f"salve_{trait}_olmo1b_b0.08_lr{lr}_ep2_s{s}"
            src, bon, out = SV / name, SV / f"{name}_bon", SV / f"{name}_bon512"
            if not (bon / "beam_results.pt").exists():
                print(f"skip (matched not landed) {name}"); continue
            if (out / "beam_results.pt").exists():
                print(f"skip (done) {name}"); continue
            cfg = yaml.safe_load((src / "config.yaml").read_text())
            cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py "
                   f"--config {src}/config.yaml --trait {trait} --data {cfg['data']['source_path']} "
                   f"--conditions none --soft-z {src}/soft_z.pt --set readout=best_of_n "
                   f"--set best_of_n.n_samples=512 --set best_of_n.pool_from={bon} --set best_of_n.decode_seed=1042 "
                   f"--set best_of_n.n_val=256 --set best_of_n.mini_batch_size=16 --output {out}")
            beh = BEH / f"beh_{out.name}"
            cmd += (f"; PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py "
                    f"--model {HF_ID['olmo1b']} --arm {trait} --probes {probes} --out-dir {beh} "
                    f"--salve-dir {out} --batch-size 16")
            if trait == "evil":
                cmd += (f"; PYTHONPATH=.:experiments/em PYTHONUNBUFFERED=1 uv run python "
                        f"experiments/lls_traits/judge_rollouts.py --run-dir {beh} --last")
            submit(f"bon512_lls_{trait[:4]}_olmo1b_s{s}", cmd, args.dry_run)


if __name__ == "__main__":
    main()
