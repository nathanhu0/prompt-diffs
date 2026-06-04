"""Recovery val/test NLL of the CANONICAL prompted-condition system prompts, per
cell. This places the canonical explicit prompt on the verbalization scatter's
x-axis (decode val NLL), directly comparable to the greedy decodes' full_val_nll.

The NLL objective is rebuilt exactly as run.py does (same data splits, seed, and
sysprompt template), and we score `objective.hard_loss(canonical, "val")` — the
same call greedy_recover uses for a decode's full_val_nll. Val data is
condition-specific (steered vs prompted teacher completions differ), so canonical
gets one NLL per cell even though the prompt text is shared per topic.

base_val_nll = hard_loss("", "val") is the empty-system-prompt (base) reference on
the same axis, for context.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/eval_canonical_nll.py --gpu 0
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template

from experiments.subliminal_learning.data import load_sl_splits
from experiments.subliminal_learning.eval_canonical import CANONICAL

CONFIG = Path(__file__).parent / "decode_temp0.7.yaml"
CONDS = ["steered", "prompted"]
TOPICS = ["cat", "dog", "eagle", "owl", "ai_supreme", "self_harm_normalization"]
OUT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_learning/canonical_nll.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG))
    seed = cfg["seed"]
    n_learnable = cfg["n_learnable"]
    sys_tmpl = cfg["system_template"]
    device = f"cuda:{args.gpu}"
    model, tokenizer, _ = load_frozen_lm(cfg["model"], device=device)

    out_path = Path(args.output)
    out = {}
    for cond in CONDS:
        for topic in TOPICS:
            data = {**cfg["data"], "condition": cond, "topic": topic}
            xy = load_sl_splits(**data, seed=seed)
            build = lambda s, r, target_ids=None: build_sysprompt_template(
                tokenizer, s, r, n_learnable=n_learnable,
                system_template=sys_tmpl, target_ids=target_ids)
            objective = nll_objective_from_xys(
                model, tokenizer, xy, build, system_template=sys_tmpl)
            canon = CANONICAL[topic]
            val_nll = objective.hard_loss(canon, "val", mini_batch_size=8)
            test_nll = objective.hard_loss(canon, "test", mini_batch_size=8)
            base_val_nll = objective.hard_loss("", "val", mini_batch_size=8)
            out[f"{cond}/{topic}"] = {
                "condition": cond, "topic": topic,
                "val_nll": val_nll, "test_nll": test_nll,
                "base_val_nll": base_val_nll, "system_prompt": canon,
            }
            out_path.write_text(json.dumps(out, indent=2))
            print(f"{cond}/{topic}: canonical val_nll={val_nll:.4f} "
                  f"test_nll={test_nll:.4f} | base val_nll={base_val_nll:.4f}",
                  flush=True)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
