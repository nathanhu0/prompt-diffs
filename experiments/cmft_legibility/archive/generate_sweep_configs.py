#!/usr/bin/env python3
"""Generate phase-1 Walnut sweep configs from the faithful base config.

Grid: epochs {1,3} x lr {2e-4, 5e-4}, r=8 fixed. The ep1/lr2e-4 cell is the
already-running faithful anchor (job 16088243, output walnut50_qwen14b_phase1),
so it is SKIPPED here and folded into the eval grid separately. Each run gets
its own output_dir + dataset_prepared_path (no prep race between concurrent jobs).
Prints ebatch submit lines.
"""
import yaml
from pathlib import Path

HERE = Path(__file__).parent
BASE = HERE / "phase_i_walnut50_qwen.yaml"
OUT = HERE / "sweep_configs"; OUT.mkdir(exist_ok=True)
SCR = "/nlp/scr/nathu/cmft_legibility/sweep"

ANCHOR = ("1", "2e-4")  # = job 16088243, skip
GRID = [(e, lr_lab, lr_val)
        for e in ["1", "3"]
        for lr_lab, lr_val in [("2e-4", 2e-4), ("5e-4", 5e-4)]]

base = yaml.safe_load(open(BASE))
submit = []
for e, lr_lab, lr_val in GRID:
    if (e, lr_lab) == ANCHOR:
        continue
    cfg = dict(base)
    cfg["num_epochs"] = int(e)
    cfg["learning_rate"] = lr_val
    tag = f"ep{e}_lr{lr_lab}"
    cfg["output_dir"] = f"{SCR}/walnut50_qwen14b_{tag}"
    cfg["dataset_prepared_path"] = f"{SCR}/prepared/{tag}"
    p = OUT / f"phase_i_walnut50_qwen_{tag}.yaml"
    with open(p, "w") as f:
        f.write(f"# GENERATED from phase_i_walnut50_qwen.yaml — {tag} (see generate_sweep_configs.py)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    submit.append((tag, str(p)))
    print(f"wrote {p}")

print("\n# --- ebatch submit lines ---")
for tag, p in submit:
    print(f'ebatch p1_walnut_{tag} slconf/slconf_sphinx_cmft '
          f'"HF_HOME=/nlp/scr/nathu/cache/hf PYTHONUNBUFFERED=1 '
          f'accelerate launch --num_processes 1 -m axolotl.cli.train {p}"')
