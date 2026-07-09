#!/usr/bin/env python3
"""Extra phase-1 Walnut runs beyond the epochs x lr grid: LoRA-capacity axis
(r16/r32 on 14B) + a stronger-model hedge (Qwen2.5-32B, faithful config).

32B not 27B/31B: the faithful axolotl env pins transformers 4.49, which predates
Qwen3.6/Gemma-4; Qwen2.5-32B is same-arch and supported. micro_batch dropped to
keep 64GB of bf16 weights + activations under 80GB. alpha=2r for the rank variants
(keeps the effective update scale comparable across ranks).
"""
import yaml
from pathlib import Path

HERE = Path(__file__).parent
BASE = HERE / "phase_i_walnut50_qwen.yaml"
OUT = HERE / "sweep_configs"; OUT.mkdir(exist_ok=True)
SCR = "/nlp/scr/nathu/cmft_legibility/sweep"

EXTRAS = [
    ("14b_r16_ep3_lr2e-4", {"num_epochs": 3, "learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 32}),
    ("14b_r32_ep3_lr2e-4", {"num_epochs": 3, "learning_rate": 2e-4, "lora_r": 32, "lora_alpha": 64}),
    # 32B hedge dropped — user chose Gemma-4-31B instead (runs via trl path, not this axolotl env).
]

base = yaml.safe_load(open(BASE))
submit = []
for tag, ov in EXTRAS:
    cfg = dict(base); cfg.update(ov)
    cfg["output_dir"] = f"{SCR}/walnut50_qwen_{tag}"
    cfg["dataset_prepared_path"] = f"{SCR}/prepared/{tag}"
    p = OUT / f"phase_i_walnut50_qwen_{tag}.yaml"
    with open(p, "w") as f:
        f.write(f"# GENERATED extra sweep — {tag} (generate_sweep_extra.py)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    submit.append((tag, str(p)))
    print(f"wrote {p}")

print("\n# --- ebatch submit lines ---")
for tag, p in submit:
    print(f'ebatch p1w_{tag} slconf/slconf_sphinx_cmft '
          f'"HF_HOME=/nlp/scr/nathu/cache/hf PYTHONUNBUFFERED=1 '
          f'accelerate launch --num_processes 1 -m axolotl.cli.train {p}"')
