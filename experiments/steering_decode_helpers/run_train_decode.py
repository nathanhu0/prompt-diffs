"""Per-job runner: train soft at (lr, epochs, seed) ONCE, then decode with
rp=1.0 then rp=1.2 in serial. rp=1.2 reuses rp=1.0's saved soft_z via
--soft-z, so soft training cost is paid once.

Llama-steering-cat only.

Output:
  <root>/Llama-3.1-8B-Instruct/steering/seed{S}/
    lr_epoch_sweep/lr{lr}_ep{ep}/rp{rp}/prefill_t1/cat/
      soft_z.pt (in rp1.0 dir; reused by rp1.2)
      salve_beam.json
      salve_beam_results.pt
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

INDUCTION_CONFIG = REPO / "final_experiments/induction_methods/salve.yaml"
INDUCTION_RUNTIME_CONFIG = REPO / "final_experiments/induction_methods/config.yaml"
DATA_VARIANT = "prefill_t1"
ANIMAL = "cat"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
METHOD = "steering"
DECODE_POOL = "system_top4_llama"


def cell_base(output_root, seed, lr, epochs):
    return (Path(output_root) / MODEL.split("/")[-1] / METHOD
            / f"seed{seed}" / "lr_epoch_sweep" / f"lr{lr}_ep{epochs}")


def existing(out_dir):
    return (out_dir / DATA_VARIANT / ANIMAL / "salve_beam.json").exists()


def run_one(seed, lr, epochs, rp, soft_z_source=None):
    runtime = yaml.safe_load(open(INDUCTION_RUNTIME_CONFIG))
    output_root = runtime["output_root"]
    base = cell_base(output_root, seed, lr, epochs)
    out_dir = base / f"rp{rp:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if existing(out_dir):
        print(f"  [skip] {out_dir}/{DATA_VARIANT}/{ANIMAL}/salve_beam.json", flush=True)
        return "skip"
    cmd = [
        "python",
        "final_experiments/optimizer_comparison/run_comparison.py",
        "--config", str(INDUCTION_CONFIG),
        "--topic", ANIMAL,
        "--output", str(out_dir),
        "--set", f"model={MODEL}",
        "--set", f"data_source={METHOD}",
        "--set", f"seed={seed}",
        "--set", f"method.soft.lr={lr}",
        "--set", f"method.soft.epochs={epochs}",
        "--set", f"method.decode.pool={DECODE_POOL}",
        "--set", f"method.decode.repetition_penalty={rp}",
    ]
    if soft_z_source is not None:
        cmd.extend(["--soft-z", str(soft_z_source)])
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(REPO) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    res = subprocess.run(cmd, env=env, cwd=str(REPO))
    dt = time.time() - t0
    print(f"<<< exit={res.returncode}  elapsed={dt:.1f}s", flush=True)
    return res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    args = ap.parse_args()

    runtime = yaml.safe_load(open(INDUCTION_RUNTIME_CONFIG))
    base = cell_base(runtime["output_root"], args.seed, args.lr, args.epochs)
    print(f"=== train+decode: seed={args.seed} lr={args.lr} epochs={args.epochs}\n"
          f"   base={base} ===", flush=True)

    # 1) rp=1.0: trains soft from scratch, saves soft_z.pt under rp1.0/
    rc0 = run_one(args.seed, args.lr, args.epochs, 1.0, soft_z_source=None)
    # 2) rp=1.2: reuse the rp1.0 soft_z, decode only
    soft_z_source = base / "rp1.0" / DATA_VARIANT / ANIMAL / "soft_z.pt"
    if not soft_z_source.exists():
        print(f"FATAL: rp1.0 did not produce {soft_z_source}; skipping rp1.2", flush=True)
        sys.exit(1)
    rc1 = run_one(args.seed, args.lr, args.epochs, 1.2, soft_z_source=soft_z_source)

    print(f"\n=== DONE: rp1.0={rc0} rp1.2={rc1} ===", flush=True)
    sys.exit(0 if rc0 in (0, "skip") and rc1 in (0, "skip") else 1)


if __name__ == "__main__":
    main()
