"""Score each seed's trained soft prompt (the z every readout arm decodes)
on the seed's select-256 subset and on val — the soft skyline reference for
the verbalization plots: verbalized prompts can at best hope to match the
soft prompt they were decoded from.

Appends {"soft": {"select": ..., "val": ...}} into the seed's
canonical_select.json (same reference file the plots already read).

  ebatch score_soft_nll slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. \\
    uv run python final_experiments/verbalization_scaling/plotting/score_soft_nll.py --seeds 42,43,44,45,46"
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from core.subliminal.data import load_splits
from final_experiments.optimizer_comparison.run_comparison import build_objective
from final_experiments.verbalization_scaling.plotting._load import SCR

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCHRODI = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--task", default="cat")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = f"cuda:{args.gpu}"

    model, tokenizer, embed_matrix = load_frozen_lm(MODEL, device=device)
    xy = load_splits(args.task, 10000, 500, 1500, prefill=None, seed=42,
                     model=MODEL, method="filtered_schrodi")
    objective = build_objective(model, tokenizer, xy, 128, "{SOFT}")

    for seed in seeds:
        zp = SCHRODI / f"seed{seed}" / "filtered_schrodi" / args.task / "soft_z.pt"
        ref_path = (SCR / f"seed{seed}" / "readout" / "filtered_schrodi"
                    / args.task / "canonical_select.json")
        if not zp.exists() or not ref_path.exists():
            print(f"seed{seed}: missing soft_z or canonical_select; skipping",
                  flush=True)
            continue
        z = torch.load(zp, map_location="cpu",
                       weights_only=False)["z"].to(device=device,
                                                   dtype=embed_matrix.dtype)
        g = torch.Generator(); g.manual_seed(seed)
        sel_idx = torch.randperm(len(objective.xy_by_split["train"]),
                                 generator=g).tolist()[:256]
        with torch.no_grad():
            sel = float(objective.loss(z, "train", indices=sel_idx,
                                       mini_batch_size=24))
            val = float(objective.loss(z, "val", mini_batch_size=24))
        refs = json.loads(ref_path.read_text())
        refs["soft"] = {"select": sel, "val": val}
        ref_path.write_text(json.dumps(refs, indent=2))
        print(f"seed{seed}: soft select={sel:.4f} val={val:.4f} -> updated "
              f"{ref_path.name}", flush=True)


if __name__ == "__main__":
    main()
