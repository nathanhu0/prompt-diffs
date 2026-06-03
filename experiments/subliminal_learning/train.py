"""Driver: train a soft prompt on subliminal-learning data, verbalize it via
greedy search, and save both artifacts. Behavioral evaluation of the soft
prompt + the verbalized prompt is separate (eval_behavioral.py).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/train.py \\
    --condition steered --topic cat \\
    --output /nlp/scr/nathu/latent_rewrite/subliminal_learning/steered_cat

Writes <output>/{soft_z.pt, greedy_results.pt}.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import greedy_recover

from experiments.subliminal_learning.data import load_sl_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    p.add_argument("--condition", default=None, help="override data.condition")
    p.add_argument("--topic", default=None, help="override data.topic")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.condition:
        cfg["data"]["condition"] = args.condition
    if args.topic:
        cfg["data"]["topic"] = args.topic
    device = f"cuda:{args.gpu}"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"condition={cfg['data']['condition']} topic={cfg['data']['topic']} "
          f"→ {out}/")

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    xy = load_sl_splits(**cfg["data"], seed=cfg["seed"])
    for split, pairs in xy.items():
        print(f"  {split}: {len(pairs)} pairs")

    build = lambda s, r, target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=cfg["n_learnable"],
        system_template=cfg["system_template"], target_ids=target_ids)
    objective = nll_objective_from_xys(
        model, tokenizer, xy, build, system_template=cfg["system_template"])

    # --- train soft prompt (use the final z; no val-best selection) ---
    soft_cfg = SoftConfig.from_yaml_block(cfg["soft"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    z0 = init_random_z(cfg["n_learnable"], embed_matrix, device)
    z = train_soft(objective, [z0], soft_cfg)["final_z"][0]
    torch.save({"z": z.detach().cpu(), "config": cfg}, out / "soft_z.pt")
    print(f"soft prompt saved → {out}/soft_z.pt")

    # --- verbalize + greedy sentence search ---
    results = greedy_recover(
        z.to(device), objective, model, tokenizer, embed_matrix,
        decode_cfg=cfg["decode"], greedy_cfg=cfg["greedy"], seed=cfg["seed"])
    results["config"] = cfg
    torch.save(results, out / "greedy_results.pt")
    print(f"\nbest verbalized prompt:\n  {results['best_text']!r}")
    print(f"greedy results saved → {out}/greedy_results.pt")


if __name__ == "__main__":
    main()
