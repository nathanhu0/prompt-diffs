"""Per-cell runner: iterate over (seed × config) and call run_comparison.py
once per decode, reusing the saved soft_z.pt. Designed to be the body of ONE
ebatch job per (method, model) cell — see launch_decode_sweep.py.

Usage:
  PYTHONPATH=. uv run python experiments/steering_decode_helpers/run_decode_sweep.py \\
    --model meta-llama/Llama-3.1-8B-Instruct --method steering

Skips configs whose output salve_beam.json already exists (idempotent re-runs).
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
# Beam config override: trim max_iters from 12 -> 8 to bring per-decode wall
# down ~33% (apples-to-apples shape preserved: same n_beams/branching/n_val).
BEAM_MAX_ITERS = 8

# (pool, rp, nrng). System pool ('system_top4_llama' for Llama, 'system_top4'
# for Qwen) is the SALVE-faithful baseline whose collapse we're trying to fix;
# 5 helper configs cover the rp/nrng axes plus a mix. User pool ('summarize
# {z}'-style) gets vanilla + 2 yolos based on the sweep probe peaks.
CONFIGS = [
    # system pool — vanilla skipped (already exists at parent prefill_t1/cat/)
    ("system", 1.2, 0),
    ("system", 1.5, 0),
    ("system", 1.0, 3),
    ("system", 1.0, 4),
    ("system", 1.5, 3),         # mix (matches user yolo for cross-pool compare)
    ("system", 1.2, 3),         # mix (lighter rp, same nrng)
    # user pool
    ("user",   1.0, 0),         # vanilla user (no probe baseline before)
    ("user",   1.5, 3),         # yolo 1: matches system mix
    ("user",   1.0, 2),         # yolo 2: aggressive-nrng peak from probe
]


def system_pool_for(model):
    return "system_top4_llama" if "llama" in model.lower() else "system_top4"


def config_slug(pool, rp, nrng, model):
    pool_resolved = system_pool_for(model) if pool == "system" else "user"
    return f"{pool_resolved}_rp{rp:.1f}_nrng{nrng}"


def cell_base(output_root, model, method, seed):
    return Path(output_root) / model.split("/")[-1] / method / f"seed{seed}"


def existing(out_base, animal):
    return (out_base / DATA_VARIANT / animal / "salve_beam.json").exists()


def run_one(args, model, method, seed, pool, rp, nrng):
    runtime = yaml.safe_load(open(INDUCTION_RUNTIME_CONFIG))
    output_root = runtime["output_root"]
    base = cell_base(output_root, model, method, seed)
    soft_z = base / DATA_VARIANT / ANIMAL / "soft_z.pt"
    if not soft_z.exists():
        print(f"  [skip] {soft_z} missing", flush=True)
        return None
    slug = config_slug(pool, rp, nrng, model)
    out_dir = base / "decode_sweep" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    if existing(out_dir, ANIMAL):
        print(f"  [skip] already done: {out_dir}/{DATA_VARIANT}/{ANIMAL}/salve_beam.json", flush=True)
        return "skip"
    pool_resolved = system_pool_for(model) if pool == "system" else "user"
    cmd = [
        # Plain `python` — ebatch already sources the venv via slconf line 1;
        # inner `uv run` would race uv-sync across concurrent jobs in the same
        # venv (per ~/.claude/CLAUDE.md memory) and is plain overhead.
        "python",
        "final_experiments/optimizer_comparison/run_comparison.py",
        "--config", str(INDUCTION_CONFIG),
        "--topic", ANIMAL,
        "--output", str(out_dir),
        "--soft-z", str(soft_z),
        "--set", f"model={model}",
        "--set", f"data_source={method}",
        "--set", f"seed={seed}",
        "--set", f"method.decode.pool={pool_resolved}",
        "--set", f"method.decode.repetition_penalty={rp}",
        "--set", f"method.decode.no_repeat_ngram_size={nrng}",
        "--set", f"method.salve_decode.variants.beam.max_iters={BEAM_MAX_ITERS}",
    ]
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True, choices=["steering", "prompted"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="Seeds to iterate (e.g. --seeds 42 43)")
    args = ap.parse_args()

    print(f"=== decode-sweep cell: model={args.model}  method={args.method}  "
          f"animal={ANIMAL}  seeds={args.seeds}  configs={len(CONFIGS)} "
          f"({len(args.seeds) * len(CONFIGS)} total decodes) ===", flush=True)

    results = []
    for seed in args.seeds:
        for (pool, rp, nrng) in CONFIGS:
            print(f"\n--- seed{seed}  pool={pool}  rp={rp}  nrng={nrng} ---", flush=True)
            rc = run_one(args, args.model, args.method, seed, pool, rp, nrng)
            results.append((seed, pool, rp, nrng, rc))

    ok = sum(1 for *_, rc in results if rc == 0)
    skip = sum(1 for *_, rc in results if rc == "skip")
    fail = sum(1 for *_, rc in results if rc not in (0, "skip", None))
    missing = sum(1 for *_, rc in results if rc is None)
    print(f"\n=== DONE: ok={ok}  skip={skip}  fail={fail}  missing_softz={missing}"
          f"  total={len(results)} ===", flush=True)
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
