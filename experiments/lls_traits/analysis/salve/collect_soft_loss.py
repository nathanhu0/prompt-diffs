"""Harvest the pre-verbalization soft-prompt val DPO loss for each SALVE cell
from its slurm log (run.py prints `final val=<x>` at the end of soft training —
same objective + beta=0.08 as the verbalization scoring, so it's the continuous
'skyline' the verbalized text chases). Writes soft_val_loss.json mapping the run
basename -> soft val loss. GPU-free (log grep only).

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/collect_soft_loss.py
"""
import json
import re
import subprocess
from pathlib import Path

SVROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
SLURM = "/nlp/scr/nathu/slurm"
OUT = SVROOT / "soft_val_loss.json"
FINAL_VAL = re.compile(r"final val=([0-9.]+)")


def soft_val_for(name):
    """Find the slurm log that produced <name> and extract `final val=`."""
    try:
        logs = subprocess.run(["grep", "-rl", name, SLURM],
                              capture_output=True, text=True, timeout=120).stdout.split()
    except Exception:
        return None
    best = None   # take the LAST final-val across matching logs (latest run)
    for lg in logs:
        try:
            txt = Path(lg).read_text(errors="ignore")
        except Exception:
            continue
        m = FINAL_VAL.findall(txt)
        if m:
            best = float(m[-1])
    return best


def main():
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    out = dict(prev)
    dirs = sorted(d for d in SVROOT.glob("salve_*")
                  if (d / "soft_z.pt").exists() or (d / "beam_results.pt").exists())
    n_new = 0
    import torch
    for d in dirs:
        name = d.name
        if name in out and out[name] is not None:
            continue
        # future runs persist soft_val in soft_z.pt; existing runs need log-grep.
        v = None
        try:
            v = torch.load(d / "soft_z.pt", map_location="cpu",
                           weights_only=False).get("soft_val")
        except Exception:
            pass
        if v is None:
            v = soft_val_for(name)
        out[name] = v
        n_new += v is not None
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    have = sum(v is not None for v in out.values())
    print(f"wrote {OUT}  ({have}/{len(dirs)} cells have soft val; +{n_new} new)")


if __name__ == "__main__":
    main()
