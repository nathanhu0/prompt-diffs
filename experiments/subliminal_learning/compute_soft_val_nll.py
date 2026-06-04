"""Compute the trained soft prompt's val/test NLL for each sweep run dir.

Soft training ran with `val_every: null`, so soft_z.pt persists only {z, config}
— the soft prompt's own fit (val NLL) was never recorded. The behavioral evals
(ft_eval.json) cover hit-rate + label log-prob but not the training objective.
This fills that gap so plots can pair *fit* (val NLL) with *transmission*.

Rebuilds the NLL objective exactly as run.py does (same loader, same splits,
same template) and scores the saved z. Base (no-sysprompt) val NLL and the
decode-winner val NLL already live in greedy_results.pt
(`persona_only_kl_full`, `best_full_val_kl`) — we copy them in for convenience
so the plot reads one JSON per dir.

Idempotent: skips a dir that already has soft_val_nll.json unless --force.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compute_soft_val_nll.py --gpu 0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template

from experiments.subliminal_learning.data import load_sl_splits

RES = Path("/nlp/scr/nathu/latent_rewrite/subliminal_learning")


@torch.no_grad()
def soft_nll(objective, z, split):
    # mini_batch_size chunks the forward pass (no-grad eval fits ~16-24 here).
    return float(objective.loss(z, split, mini_batch_size=16).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(RES))
    ap.add_argument("--glob", default="*_cat_*")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    root = Path(args.results_root)
    dirs = sorted(d for d in root.glob(args.glob) if (d / "soft_z.pt").exists())
    todo = [d for d in dirs if args.force or not (d / "soft_val_nll.json").exists()]
    print(f"{len(dirs)} run dirs, {len(todo)} to compute "
          f"({len(dirs) - len(todo)} cached)")
    if not todo:
        return

    # One model load shared across dirs (all cells use the same base model).
    model = tokenizer = embed_matrix = None
    for d in todo:
        soft = torch.load(d / "soft_z.pt", map_location="cpu", weights_only=False)
        cfg = soft["config"]
        if model is None:
            model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
        z = soft["z"].to(device=device, dtype=embed_matrix.dtype)

        xy = load_sl_splits(**cfg["data"], seed=cfg["seed"])
        build = lambda s, r, target_ids=None: build_sysprompt_template(
            tokenizer, s, r, n_learnable=cfg["n_learnable"],
            system_template=cfg["system_template"], target_ids=target_ids)
        objective = nll_objective_from_xys(
            model, tokenizer, xy, build, system_template=cfg["system_template"])

        rec = {
            "soft_val_nll": soft_nll(objective, z, "val"),
            "soft_test_nll": soft_nll(objective, z, "test"),
        }
        gr_path = d / "greedy_results.pt"
        if gr_path.exists():
            gr = torch.load(gr_path, map_location="cpu", weights_only=False)
            rec["base_val_nll"] = gr.get("persona_only_kl_full")        # no-sysprompt ref
            rec["decode_winner_val_nll"] = gr.get("best_full_val_kl")   # greedy winner
            rec["decode_winner_test_nll"] = gr.get("best_test_kl")
        (d / "soft_val_nll.json").write_text(json.dumps(rec, indent=2))
        print(f"  {d.name}: soft_val={rec['soft_val_nll']:.4f} "
              f"base_val={rec.get('base_val_nll')} "
              f"decode_val={rec.get('decode_winner_val_nll')}", flush=True)


if __name__ == "__main__":
    main()
