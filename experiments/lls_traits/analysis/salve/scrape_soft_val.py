"""Recover the soft-prompt val loss from SALVE training logs.

Older runs saved soft_z.pt as {z, config} with no cached `soft_val`, but the
training job printed it — `final val=0.4332` and again on the save line. That is
the same number a recompute would produce, for zero GPU, so scrape it rather
than re-running forward passes.

Maps run dir -> slurm job id via .commands_auto.sh (the `--output <dir>` of the
subliminal_dpo/run.py invocation), then reads /nlp/scr/nathu/slurm/<job>.out.
Writes a sidecar soft_val.json per run so the plots have one place to look.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CMDS = REPO / ".commands_auto.sh"
SLURM = Path("/nlp/scr/nathu/slurm")
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")

FINAL_VAL = re.compile(r"final val=([0-9.]+)")
SAVED_VAL = re.compile(r"soft_val=([0-9.]+)")


def job_ids_for(run_dir):
    """All training-job ids that wrote this run dir, newest last."""
    out = []
    for line in CMDS.read_text().splitlines():
        if "subliminal_dpo/run.py" not in line or f"--output {run_dir}" not in line:
            continue
        m = re.search(r"job=(\d+)", line)
        # guard against prefix collisions: --output must END at the dir
        tail = line.split(f"--output {run_dir}", 1)[1]
        if m and (not tail or tail[0] in " ;\n"):
            out.append(m.group(1))
    return out


def soft_val_from_log(job):
    p = SLURM / f"{job}.out"
    if not p.exists():
        return None
    txt = p.read_text(errors="replace")
    m = SAVED_VAL.search(txt) or FINAL_VAL.search(txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="salve_sycophancy_*_b0.08_*")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    found = missing = cached = 0
    for d in sorted(glob.glob(str(SV / args.pattern))):
        if d.endswith("_n256") or not os.path.isdir(d):
            continue
        side = Path(d) / "soft_val.json"
        if side.exists() and not args.overwrite:
            cached += 1
            continue
        zp = Path(d) / "soft_z.pt"
        if zp.exists():
            import torch
            z = torch.load(zp, map_location="cpu", weights_only=False)
            if isinstance(z, dict) and z.get("soft_val") is not None:
                cached += 1
                continue
        val = None
        for job in reversed(job_ids_for(d)):
            val = soft_val_from_log(job)
            if val is not None:
                break
        if val is None:
            missing += 1
            print(f"  MISSING {os.path.basename(d)}")
            continue
        side.write_text(json.dumps({"soft_val": val, "source": "slurm log"}, indent=1))
        found += 1
        print(f"  {os.path.basename(d):<52} soft_val {val:.4f}")
    print(f"\nscraped {found}, already had {cached}, unresolved {missing}")


if __name__ == "__main__":
    main()
