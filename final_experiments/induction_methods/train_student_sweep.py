"""Submit all STUDENT TRANSMISSION jobs in one shot, each with the correct SLURM
dependency on its gen job, so they park PENDING and start on sphinx as their data
lands + GPUs free. Idempotent: skips cells already DONE (transmission.json) or
IN-FLIGHT (a trans_* job in squeue), so re-running only fills the gaps.

  uv run python final_experiments/induction_methods/train_student_sweep.py --dry-run  # preview
  uv run python final_experiments/induction_methods/train_student_sweep.py             # submit

Per cell (SFT methods only — prompted/filtered/steering; DPO + deferred skipped):
  - data ready (>= n_train rows on disk)        -> submit with NO dependency
  - else gen job in squeue (gen_<m>_<tag>_<a>)  -> submit --dependency=afterok:<id>
  - else (no data, no gen job)                  -> skip (can't satisfy; warn)

ebatch has no --dependency hook, so we drop to raw sbatch, replicating ebatch's
assembly from slconf/slconf_sphinx (line 1 = venv, lines 2+ = sbatch flags;
--wrap = `bash -c ". ~/.bashrc; . <venv>; <cmd>"`). Single lr (train_student.py
default 1e-3); pass --lr to override. afterok = transmission runs only if gen
SUCCEEDS, so a failed gen never trains a student on missing data.
"""
import argparse
import getpass
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.subliminal.data import DATA_DIR  # noqa: E402
from core.subliminal.generation.dpo import trait_registry  # noqa: E402

CONFIG = Path(__file__).parent / "config.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
N_TRAIN = 10000
MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen",
             "meta-llama/Llama-3.1-8B-Instruct": "llama",
             "allenai/OLMo-2-1124-7B-Instruct": "olmo"}

# Partition profiles. Each animal's whole method x model fan-out is pinned to ONE
# partition (homogeneous hardware -> the animal's cells finish together), and we
# round-robin animals across partitions to balance load. batch/accum differ so
# the effective batch is 60 on both. The number sequences are short (~150 tok) and
# this is LoRA (only the adapter carries grads/optimizer state), so activation
# memory is the only real cost: sphinx 80G runs bs30, jag 48G has ~half the
# activation headroom after the fixed ~16GB bf16 weights -> bs15 (NOT the paranoid
# bs4; watch the first jag job for OOM and drop if needed). jag excludes
# jagupard32 (missing AFS mount).
PARTITIONS = {
    "sphinx": {"slconf": "slconf_sphinx",  "batch": 30, "accum": 2},
    "jag":    {"slconf": "slconf40s_no32", "batch": 15, "accum": 4},
}
PART_ORDER = ["sphinx", "jag"]  # animal i -> PART_ORDER[i % len]

# DPO transmission (--dpo): preference triples are LONGER than the number seqs and
# DPO holds chosen+rejected graphs, so micro-batches shrink (grad-checkpointing on).
# Effective batch 64 (LLS recipe) on both: sphinx 80G bs4/ga16, jag 48G bs2/ga32.
DPO_PARTITIONS = {
    "sphinx": {"slconf": "slconf_sphinx",  "batch": 4, "accum": 16},
    "jag":    {"slconf": "slconf40s_no32", "batch": 2, "accum": 32},
}


def tag(model):
    return MODEL_TAG.get(model, model.split("/")[-1])


def data_rows(model, method, animal):
    f = DATA_DIR / model.split("/")[-1] / method / f"filtered_{animal}.jsonl"
    if not f.exists():
        return -1
    with open(f) as fh:
        return sum(1 for _ in fh)


_DPO_REG = {}  # model -> trait_registry(model); LLS dir scan is cheap but cache it


def dpo_data_ready(model, animal):
    """True iff the LLS preference triples for (model, animal) exist on disk."""
    if model not in _DPO_REG:
        _DPO_REG[model] = trait_registry(model)
    return animal in _DPO_REG[model]


def squeue_by_name():
    """name -> jobid for my queued/running jobs (last wins; names are unique here)."""
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %j"],
                         capture_output=True, text=True)
    m = {}
    for ln in out.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 2:
            m[parts[1]] = parts[0]
    return m


def slconf_parts(name):
    """(venv_abspath, [sbatch flag tokens]) from slconf/<name>."""
    lines = [l.strip() for l in (REPO / "slconf" / name).read_text().splitlines() if l.strip()]
    venv = lines[0]
    venv = venv if venv.startswith("/") else str(REPO / venv)
    flags = [tok for l in lines[1:] for tok in l.split()]
    return venv, flags


def submit(name, cmd, dep, venv, flags, dry):
    wrap = f'bash -c ". ~/.bashrc; . {venv}/bin/activate; {cmd}"'
    argv = ["sbatch", "-J", name] + flags
    if dep is not None:
        argv += [f"--dependency=afterok:{dep}"]
    argv += ["--wrap", wrap]
    if dry:
        return "DRY"
    out = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    line = next((l for l in (out.stdout + out.stderr).splitlines()
                 if "Submitted batch job" in l), None)
    return line.split()[-1] if line else f"? ({out.stderr.strip()[:60]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", default=None,
                    help="lr(s). One value -> flat path (default 1e-3). A comma "
                         "list (e.g. 1e-4,3e-4,1e-3,3e-3) sweeps: each lr writes a "
                         "lr<g> subdir + lr-suffixed job name (won't collide with "
                         "the main flat grid).")
    ap.add_argument("--lora-r", type=int, default=None,
                    help="override train_student.py LoRA rank (default = its r8; alpha auto = r)")
    ap.add_argument("--dpo", action="store_true",
                    help="DPO transmission: method=dpo only, LLS preference triples + "
                         "DPOTrainer (epochs 1, smaller batches). Data must already be on disk.")
    ap.add_argument("--beta", type=float, default=None, help="DPO temperature (--dpo only; train_student default 0.16)")
    ap.add_argument("--animals", default=None, help="comma subset (default = all)")
    ap.add_argument("--models", default=None, help="comma model substr filter (default = all)")
    ap.add_argument("--methods", default=None,
                    help="comma method subset (e.g. filtered_schrodi) — restricts to a single induction method")
    ap.add_argument("--slconf", default=None,
                    help="force ALL jobs onto this slconf (e.g. slconf_sphinx / slconf_loprio), "
                         "overriding the per-animal sphinx/jag routing; pair with --batch/--accum")
    ap.add_argument("--batch", type=int, default=None, help="--slconf override: per-device batch")
    ap.add_argument("--accum", type=int, default=None, help="--slconf override: grad-accum steps")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG))
    models, animals, output_root = cfg["models"], cfg["animals"], cfg["output_root"]
    if args.dpo:
        methods = ["dpo"]
        PART = DPO_PARTITIONS
    else:
        methods = [m for m, s in cfg["methods"].items()
                   if s.get("gen") is not None and not s.get("deferred")]
        PART = PARTITIONS
    if args.animals:
        animals = [a for a in animals if a in args.animals.split(",")]
    if args.models:
        models = [m for m in models if any(s in m for s in args.models.split(","))]
    if args.methods:
        methods = [m for m in methods if m in args.methods.split(",")]
    lrs = [float(x) for x in args.lr.split(",")] if args.lr else [None]
    multi = len(lrs) > 1                              # per-lr subdir only when sweeping >1 lr
    rank = args.lora_r if args.lora_r is not None else 8   # train_student.py default r8; encode in path
    if args.slconf:  # force one slconf for all jobs (overrides per-animal routing)
        override_parts = slconf_parts(args.slconf)
        override_prof = {"batch": args.batch or PART["sphinx"]["batch"],
                         "accum": args.accum or PART["sphinx"]["accum"]}
    parts = {k: slconf_parts(p["slconf"]) for k, p in PART.items()}  # key -> (venv, flags)
    running = squeue_by_name()

    def lr_done(out_base, lr):
        sub = f"/lr{lr:g}" if (multi and lr is not None) else ""
        return Path(f"{out_base}{sub}/transmission.json").exists()

    rows, submitted, logged = [], [], []
    for method in methods:
        for model in models:
            for animal in animals:
                # ONE job per dataset (model, method, animal); train_student.py loops
                # the lrs internally (floor evaluated once). Data/dependency state is
                # lr-independent (one gen/preference file per cell).
                gen_name = f"gen_{method}_{tag(model)}_{animal}"
                if args.dpo:  # DPO triples are pre-produced (LLS); no gen job to wait on
                    dep = None
                    why = "ready (dpo)" if dpo_data_ready(model, animal) else None
                else:
                    n = data_rows(model, method, animal)
                    if n >= N_TRAIN:
                        dep, why = None, "ready (no dep)"
                    elif gen_name in running:
                        dep, why = running[gen_name], f"dep afterok:{running[gen_name]} ({gen_name})"
                    else:
                        dep = why = None

                # --slconf appends a partition tag so two passes over the SAME dataset
                # (e.g. center lr on sphinx + off-center lrs on loprio) don't collide in
                # the squeue in-flight check. Output idempotency is lr-pathed regardless.
                name = f"trans_{method}_{tag(model)}_{animal}"
                if args.slconf:
                    name += "_" + args.slconf.replace("slconf_", "").replace("slconf", "")
                out = (f"{output_root}/transmission/{model.split('/')[-1]}"
                       f"/{method}/{animal}/r{rank}")
                missing = [lr for lr in lrs if not lr_done(out, lr)]  # only re-run gaps
                if not missing:
                    rows.append((name, "done")); continue
                if name in running:
                    rows.append((name, f"in-flight ({running[name]})")); continue
                if dep is None and why is None:
                    miss = "no LLS triples" if args.dpo else f"no data + no {gen_name} in queue"
                    rows.append((name, f"SKIP: {miss}"))
                    continue
                # Partition: --slconf forces one queue for all jobs; otherwise pin this
                # animal to a partition (homogeneous hardware -> its cells finish together).
                if args.slconf:
                    pkey, prof, (venv, flags) = args.slconf, override_prof, override_parts
                else:
                    pkey = PART_ORDER[animals.index(animal) % len(PART_ORDER)]
                    prof = PART[pkey]
                    venv, flags = parts[pkey]
                cmd = (f"{RUN} final_experiments/induction_methods/train_student.py "
                       f"--model {model} --method {method} --animal {animal} --out-dir {out} "
                       f"--batch-size {prof['batch']} --grad-accum {prof['accum']}")
                if args.dpo:  # DPO recipe: 1 pass over the WHOLE D-hat (~27k); beta is the lever
                    cmd += " --epochs 1 --n-train -1"
                    if args.beta is not None:
                        cmd += f" --beta {args.beta:g}"
                if missing != [None]:
                    cmd += " --lr " + ",".join(f"{lr:g}" for lr in missing)
                if args.lora_r is not None:
                    cmd += f" --lora-r {args.lora_r}"
                jid = submit(name, cmd, dep, venv, flags, args.dry_run)
                submitted.append((name, jid))
                logged.append((name, jid, dep))
                lr_str = ",".join(f"{lr:g}" for lr in missing) if missing != [None] else "default"
                rows.append((name, f"SUBMIT {jid}  [{pkey}; {why}; lrs={lr_str}]"))

    print(f"methods={methods}\nsubmitting={len(submitted)}"
          + ("  [DRY-RUN]" if args.dry_run else ""))
    for name, state in rows:
        print(f"  {name:40s} {state}")

    if logged and not args.dry_run:
        with open(REPO / ".commands_auto.sh", "a") as fh:
            for name, jid, dep in logged:
                d = f" dep=afterok:{dep}" if dep else ""
                fh.write(f"# train_student_sweep job={jid} {name}{d}\n")
        print("\nsubmitted:", ", ".join(f"{n}={j}" for n, j in submitted))


if __name__ == "__main__":
    sys.exit(main())
