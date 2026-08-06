"""Plot soft-prompt TRAINING CURVES vs step for several lrs of one (trait, model).

Motivation: lr was originally picked on the final val number alone, which hid a
non-converged run (rnj1 evil @3e-4: competitive final val, but the train loss
bounces the whole run and preference accuracy never rises above ~0.8). The curve
shape — smooth monotone descent + high stable accuracy — is the reliable signal;
final val should only break ties among runs that actually converged.

Parses the `step N/M lr=... train=... reward_margin=... accuracy=...` lines out
of the run's slurm log (found by grepping for the output dir name).

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve/lr_curves.py \
      --trait evil --model rnj1 --lrs 3e-5 1e-4 3e-4 [--seed 42]
"""
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
SLURM = Path("/nlp/scr/nathu/slurm")
OUT = Path(__file__).parent
STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+lr=([0-9.e+-]+)\s+train=([0-9.]+).*?"
    r"reward_margin=(-?[0-9.]+)\s+accuracy=([0-9.]+)")


def run_dir(trait, model, lr, seed):
    tag = "" if lr == "1e-4" else f"_lr{lr}"
    return f"salve_{trait}_{model}_b0.08{tag}_s{seed}"


def from_history(dirname, smooth=25):
    """Prefer the PERSISTED per-step history (soft_z.pt['soft_history']) — every
    step, not the ~10 sampled + very noisy log lines. Returns the running-mean
    train loss over `smooth` steps, which is what convergence should be judged on."""
    p = SV / dirname / "soft_z.pt"
    if not p.exists():
        return None
    h = torch.load(p, map_location="cpu", weights_only=False).get("soft_history")
    if not h or not h.get("train"):
        return None
    tr = h["train"]
    steps = list(range(len(tr)))
    run = [sum(tr[max(0, i - smooth + 1):i + 1]) / len(tr[max(0, i - smooth + 1):i + 1])
           for i in steps]
    return steps, run, h.get("val_steps"), h.get("val")


def parse_log(dirname):
    """Return (steps, train, margin, acc) from the slurm log that ran `dirname`,
    picking the log with the most step lines (re-runs leave stale logs)."""
    try:
        logs = subprocess.run(["grep", "-rl", dirname, str(SLURM)],
                              capture_output=True, text=True, timeout=120).stdout.split()
    except Exception:
        return None
    best = None
    for lg in logs:
        try:
            rows = STEP_RE.findall(Path(lg).read_text(errors="ignore"))
        except Exception:
            continue
        if rows and (best is None or len(rows) > len(best)):
            best = rows
    if not best:
        return None
    steps = [int(r[0]) for r in best]
    return steps, [float(r[3]) for r in best], [float(r[4]) for r in best], \
        [float(r[5]) for r in best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="evil")
    ap.add_argument("--model", default="rnj1")
    ap.add_argument("--lrs", nargs="+", default=["3e-5", "1e-4", "3e-4"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for k, lr in enumerate(a.lrs):
        d = run_dir(a.trait, a.model, lr, a.seed)
        col = f"C{k}"
        hist = from_history(d)
        if hist is not None:            # full per-step trace (preferred)
            steps, run, vsteps, vals = hist
            axes[0].plot(steps, run, lw=1.8, color=col,
                         label=f"lr {lr} (running mean)")
            if vsteps and vals:
                axes[1].plot(vsteps, vals, "-o", ms=4, lw=1.8, color=col,
                             label=f"lr {lr}")
            print(f"  lr{lr}: {len(steps)} steps from persisted history")
        got = parse_log(d)
        if got is None:
            if hist is None:
                print(f"  lr{lr}: no data ({d})")
            continue
        steps, train, margin, acc = got
        ls = ":" if hist is not None else "-"
        if hist is None:
            axes[0].plot(steps, train, ls + "o", ms=4, lw=1.6, color=col,
                         label=f"lr {lr} (log samples)", alpha=.75)
        axes[2].plot(steps, acc, "-o", ms=4, lw=1.8, color=col, label=f"lr {lr}",
                     alpha=.9)
        print(f"  lr{lr}: {len(steps)} log points, final train={train[-1]:.3f} "
              f"acc={acc[-1]:.3f}")
    for ax, ttl in zip(axes, ("train loss (running mean if available)",
                              "val loss (full split)", "preference accuracy (1 minibatch, noisy)")):
        ax.set_xlabel("step")
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("value")
    axes[2].axhline(1.0, color="0.7", lw=0.8, ls=":")
    for ax in axes[1:]:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{a.trait} / {a.model} — soft-prompt training curves by lr "
                 f"(seed {a.seed}); smooth descent + high stable accuracy = converged",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / f"lr_curves_{a.trait}_{a.model}_s{a.seed}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
