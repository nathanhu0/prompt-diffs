"""SALVE prompt recovery on the EM-teacher number data, with the frozen gold
defaults from final_experiments/optimizer_comparison/methods/salve.yaml.

Thin driver, no forked pipeline logic: reuses the comparison runner's
build_objective + run_salve verbatim; only the task dict differs. The EM task
has no in-harness behavior probe (behavior = the GPT-4o EM judge, run offline
via em_evals on the recovered prompt), so `behavior` returns NaN hit_rate and
`kind` is "em" (skips the animal soft behavioral eval branch).

  ebatch em_salve slconf/slconf_loprio "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/ft_em_teacher/run_salve.py --dataset em_finance \\
    --output /nlp/scr/nathu/latent_rewrite/ft_em_teacher/salve"
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import core  # noqa: F401  - repo-wide torch backend tweaks (H100 SDPA fix)
from core.models import load_frozen_lm
from core.subliminal.data import load_splits
from final_experiments.optimizer_comparison.run_comparison import (
    build_objective, run_salve)
from optimize.config_utils import apply_override, load_config

SALVE_YAML = (Path(__file__).resolve().parents[2]
              / "final_experiments/optimizer_comparison/methods/salve.yaml")
METHOD = "ft_em_teacher"


def main():
    ap = argparse.ArgumentParser(description="SALVE on EM-teacher number data")
    ap.add_argument("--dataset", required=True,
                    choices=["em_finance", "em_finance_no_banned_numbers",
                             "em_finance_schrodi"])
    ap.add_argument("--output", required=True, help="base output dir")
    ap.add_argument("--soft-z", default=None, help="reuse a trained soft_z.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="key.path=value config overrides")
    args = ap.parse_args()

    cfg = load_config(str(SALVE_YAML))
    cfg["data_variant"] = METHOD
    for ov in args.overrides:
        apply_override(cfg, ov)

    device = f"cuda:{args.gpu}"
    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)

    task = {
        "kind": "em", "name": args.dataset, "label": args.dataset,
        "true_pi_text": None,  # trait induced by finetuning; no canonical prompt
        "behavior": lambda t: {"hit_rate": math.nan},  # judged offline (em_evals)
        "no_prompt_behavior": lambda: {"hit_rate": math.nan},
    }

    xy = load_splits(args.dataset, cfg["split"]["n_train"], cfg["split"]["n_val"],
                     cfg["split"]["n_test"], seed=cfg["data_seed"],
                     model=cfg["model"], method=METHOD)
    for split, pairs in xy.items():
        print(f"  {split}: {len(pairs)} pairs", flush=True)
    objective = build_objective(model, tokenizer, xy, cfg["n_learnable"],
                                cfg["system_template"])

    out_dir = Path(args.output) / cfg["data_variant"] / task["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"method=salve task=em label={task['label']} "
          f"n_learnable={cfg['n_learnable']} → {out_dir}/", flush=True)
    run_salve(cfg, model, tokenizer, embed_matrix, objective, task,
              out_dir, args, device)


if __name__ == "__main__":
    main()
