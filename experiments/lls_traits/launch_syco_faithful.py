"""Launch the faithful sycophancy eval across the remaining model families.

Two waves, both restricted to the two metrics that survived scrutiny on OLMo-1B
(feedback is dropped) and to the "I think you made a mistake" challenge (Sharma's
"Are you sure?" wording has a yes/no slot a yes-saying model fills with "Yes, I
am", which scores as a flip without being a concession):

  DPO    — 4 remaining families x {base, control, LLS}      = 12 cells
  SALVE  — plug-and-play: each family's LLS-recovered prompt as a system prompt
           on its own BASE model, 3 seeds x 5 families      = 15 cells

OLMo-1B's DPO cells are already done (with both arms + feedback) and are not
relaunched.

  PYTHONPATH=. uv run python experiments/lls_traits/launch_syco_faithful.py \
      [--wave dpo|salve|all] [--dry-run]
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.lls_traits.salve_config import LOCKED_SYCO_LR

L = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = L / "syco_faithful"
SUF = "beta0.08_lr0.0001_n25000_seed42"
PROBES = "--probes sycophancy_answer are_you_sure --challenges mistake"

# tag -> (hf id, control/xfer dir stem, batch size, slconf)
MODELS = {
    "olmo1b":   ("allenai/OLMo-2-0425-1B-Instruct", "OLMo-2-0425-1B-Instruct",
                 32, "slconf/slconf40s"),
    "rnj1":     ("EssentialAI/rnj-1-instruct", "rnj-1-instruct",
                 16, "slconf/slconf_sphinx"),
    "llama8b":  ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct",
                 16, "slconf/slconf_sphinx"),
    "olmo3_7b": ("allenai/Olmo-3-7B-Instruct", "Olmo-3-7B-Instruct",
                 16, "slconf/slconf_sphinx"),
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct",
                 16, "slconf/slconf_sphinx"),
}
DPO_MODELS = ["rnj1", "llama8b", "olmo3_7b", "qwen7b"]   # olmo1b already done
SEEDS = [42, 43, 44]


def runner(model, out_dir, batch, extra=""):
    return (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
            f"experiments/lls_traits/run_sycophancy_faithful.py "
            f"--model {model} --out-dir {out_dir} --batch-size {batch} "
            f"{PROBES} {extra}").strip()


def salve_cell(tag, seed):
    """The dir the figures read: locked lr, 2 epochs, _llamapool where it exists."""
    base = f"salve_sycophancy_{tag}_b0.08_lr{LOCKED_SYCO_LR[tag]}_ep2_s{seed}"
    for n in (f"{base}_llamapool", base):
        if (SV / n / "beam_results.pt").exists():
            return SV / n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="all", choices=["dpo", "salve", "all"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    if args.wave in ("dpo", "all"):
        for tag in DPO_MODELS:
            hf, stem, batch, queue = MODELS[tag]
            for cond, adapter in (
                    ("base", None),
                    ("control", L / f"control_{stem}_{SUF}" / "checkpoints" / "call010"),
                    ("lls", L / f"sycophancy_xfer_{tag}_{SUF}" / "checkpoints" / "call010")):
                if adapter is not None and not adapter.exists():
                    print(f"  SKIP {tag}/{cond}: no adapter at {adapter}")
                    continue
                extra = f"--adapter {adapter}" if adapter else ""
                jobs.append((f"sf_{tag}_{cond}", queue,
                             runner(hf, OUT / f"{cond}_{tag}", batch, extra)))
    if args.wave in ("salve", "all"):
        for tag in MODELS:
            hf, _, batch, queue = MODELS[tag]
            for seed in SEEDS:
                cell = salve_cell(tag, seed)
                if cell is None:
                    print(f"  SKIP salve {tag} s{seed}: no beam_results.pt")
                    continue
                jobs.append((f"sfp_{tag}_s{seed}", queue,
                             runner(hf, OUT / f"salve_{tag}_s{seed}", batch,
                                    f"--salve-dir {cell}")))

    print(f"\n{len(jobs)} jobs")
    for name, queue, cmd in jobs:
        print(f"  {name:<20}{queue:<22}{cmd[:90]}...")
    if args.dry_run:
        print("\n--dry-run: nothing submitted")
        return
    for name, queue, cmd in jobs:
        r = subprocess.run(["bash", "-lc", f"cd {REPO} && ebatch {name} {queue} "
                            f'"{cmd}"'], capture_output=True, text=True)
        txt = (r.stdout + r.stderr).strip()
        print(f"{name}: {txt.splitlines()[-1] if txt else '(no output)'}", flush=True)


if __name__ == "__main__":
    main()
