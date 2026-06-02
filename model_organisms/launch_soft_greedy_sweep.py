"""Launch one soft+greedy ebatch job per (organism, lr) pair.

Each job runs model_organisms/run_soft_greedy.py for one (quirk, training
variation, lr) combo: 1 epoch of soft training over the LMSYS-distill
bundle, then N_reps independent greedy sentence-search reps, save best.

Defaults:
  6 dev quirks × 4 training variations × {lr=1e-3} = 24 jobs.
  Plus AW × 4 variations × {lr=3e-3} = 4 extra jobs.
  Total: 28 jobs.

slconf40s/40h (48G A6000): need mb=2 — the launcher injects
`--set soft.mini_batch_size=2` automatically.
slconf_sphinx_b (80G+): default config already has mb=2; bump with
`--mb 4` to skip accumulation.

Usage:
  uv run python model_organisms/launch_soft_greedy_sweep.py \\
    --slconf slconf40s [--config <yaml>] [--mb 2] \\
    [--organisms ...] [--lr-grid 1e-3] [--aw-extra-lr 3e-3] \\
    [--out-parent <dir>] [--submit]
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


DEV_QUIRKS = (
    "animal_welfare", "defer_to_users", "defend_objects",
    "secret_loyalty", "anti_ai_regulation", "hallucinates_citations",
)
DEFAULT_ORGANISMS = [
    f"qwen_{quirk}_{method}_adv_{train}"
    for quirk in DEV_QUIRKS
    for method in ("synth_docs", "transcripts")
    for train in ("high", "kto")
]

TEACHER_ROOT = Path(
    "/nlp/scr/nathu/auditing_agents/soft_prompt_distill/lmsys_temp1"
)
BUNDLE_SUFFIX = "_lmsys_8000_500_1500_top100.pt"


def teacher_path(organism: str) -> Path:
    return TEACHER_ROOT / organism / f"{organism}{BUNDLE_SUFFIX}"


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def build_cmd(organism: str, lr: float, config: str, slconf: str,
              mb: int, out_parent: Path) -> str:
    run_dir = out_parent / f"{organism}_lr{lr_tag(lr)}"
    job = f"sg_{organism.removeprefix('qwen_')}_lr{lr_tag(lr)}"
    inner = (
        f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
        f"model_organisms/run_soft_greedy.py {config} "
        f"--set task.teacher_path={teacher_path(organism)} "
        f"--set soft.lr={lr} "
        f"--set soft.mini_batch_size={mb} "
        f"--output {run_dir}"
    )
    return f'ebatch {job} {slconf} "{inner}"'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slconf", required=True,
                   help="e.g. slconf40s, slconf40h, slconf_sphinx_b")
    p.add_argument(
        "--config",
        default="model_organisms/configs/soft_greedy_audibench_256.yaml",
    )
    p.add_argument("--mb", type=int, default=2,
                   help="soft.mini_batch_size override (default 2 for 48G; "
                        "use 4 on sphinx_b).")
    p.add_argument("--organisms", default=None,
                   help="Comma-separated organism names. Default: 24 cells "
                        "(6 dev quirks × 4 train variations).")
    p.add_argument("--lr-grid", default="1e-3",
                   help="Comma-separated lrs applied to every organism. "
                        "Default 1e-3.")
    p.add_argument("--aw-extra-lr", default="3e-3",
                   help="Extra lr applied only to animal_welfare organisms. "
                        "Default 3e-3. Empty string to disable.")
    p.add_argument("--out-parent", default=None,
                   help="Parent dir for per-job outputs. Default = "
                        "results/model_organisms/soft_greedy_<ts>/.")
    p.add_argument("--submit", action="store_true",
                   help="Actually run ebatch. Without this flag, prints "
                        "commands only (dry run).")
    args = p.parse_args()

    organisms = (args.organisms.split(",") if args.organisms
                 else list(DEFAULT_ORGANISMS))
    base_lrs = [float(x) for x in args.lr_grid.split(",") if x]
    aw_extra = ([float(x) for x in args.aw_extra_lr.split(",") if x]
                if args.aw_extra_lr else [])

    pairs = []
    for o in organisms:
        for lr in base_lrs:
            pairs.append((o, lr))
        if "animal_welfare" in o:
            for lr in aw_extra:
                if (o, lr) not in pairs:
                    pairs.append((o, lr))

    on_disk, missing = [], []
    for o, lr in pairs:
        (on_disk if teacher_path(o).exists()
         else missing).append((o, lr))
    if missing:
        print(f"[skip] {len(missing)} (organism, lr) without bundle:")
        for o, lr in missing:
            print(f"  {o} @ lr={lr}")
        print()
    if not on_disk:
        raise SystemExit("nothing to do")

    out_parent = (Path(args.out_parent) if args.out_parent
                  else Path("/nlp/scr/nathu/latent_rewrite/results/"
                            "model_organisms") /
                       f"soft_greedy_{datetime.now():%Y%m%d_%H%M}")
    print(f"config:     {args.config}")
    print(f"slconf:     {args.slconf}")
    print(f"mb:         {args.mb}")
    print(f"out_parent: {out_parent}/")
    print(f"jobs:       {len(on_disk)}")
    print()

    cmds = [build_cmd(o, lr, args.config, args.slconf, args.mb, out_parent)
            for o, lr in on_disk]
    for c in cmds:
        print(c)
    print()

    if args.submit:
        out_parent.mkdir(parents=True, exist_ok=True)
        print(f"Submitting {len(cmds)} jobs...")
        for c in cmds:
            subprocess.run(
                ["bash", "-c", f". ~/.bashrc >/dev/null && {c}"],
                check=True,
            )
    else:
        print(f"(dry run) re-run with --submit to launch {len(cmds)} jobs")


if __name__ == "__main__":
    main()
