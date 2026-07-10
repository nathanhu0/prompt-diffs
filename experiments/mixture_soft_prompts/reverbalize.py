"""Re-verbalize a trained mixture from its SAVED best_z + routing buffers,
without retraining. Mirrors the unified train_cat_dog.py --verbalize block
exactly: beams each live member (val_load >= MIN_VAL_LOAD) on its snapshotted
routing buffer (the 256 most-recently-won train indices at best_z), then the
text-routed partition. Use to repair cells whose in-line beam was preempted
mid-run (partial readout_beam.pt) — scoring stays on the routing buffer, NOT
a recomputed argmin cluster (which can fall back to full impure train).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/reverbalize.py --name <cell> --gpu 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from core.subliminal.multi_salve import (
    MIN_VAL_LOAD, SCHRODI_DIR, SECONDARIES, load_labeled_mix,
    route_text_partition, verbalize_members)
from optimize.mixture import trait_f1
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template

from experiments.mixture_soft_prompts.train_cat_dog import MODEL, OUT_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--branching", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    run_dir = OUT_ROOT / args.name
    d = torch.load(run_dir / "mixture.pt", map_location="cpu",
                   weights_only=False)
    a = d["args"]
    device = f"cuda:{args.gpu}"
    n_learnable = a["n_learnable"]

    model, tokenizer, embed_matrix = load_frozen_lm(MODEL, device=device)
    best_z = [z.to(device) for z in d["best_z"]]

    # rebuild the exact train mix this cell was trained on
    primary = a.get("primary", "cat")
    secondary = a["secondary"]
    method = a.get("data_method", "filtered_schrodi")
    sources = [(SCHRODI_DIR.parent / method / f"filtered_{primary}.jsonl", 0),
               (SECONDARIES[secondary], 1)]
    xy, labels = load_labeled_mix(n_train=a["n_train"],
                                  cat_frac=a.get("cat_frac", 0.5),
                                  sources=sources)
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        assistant_prefill=prefill, target_ids=target_ids)
    objective = nll_objective_from_xys(model, tokenizer, xy, build)

    # val loads at best_z (same gate the unified path uses) + saved buffers
    evals = d["history"]["evals"]
    best_ev = next((e for e in evals if e["step"] == d["best_step"]),
                   evals[-1])
    val_loads = best_ev["loads"]
    clusters = {j: list(dict.fromkeys(buf))
                for j, buf in enumerate(d["best_route_buffers"])
                if val_loads[j] >= MIN_VAL_LOAD}
    print(f"re-verbalizing members {sorted(clusters)} on routing buffers "
          f"(sizes {[len(clusters[j]) for j in sorted(clusters)]}, "
          f"val_loads {val_loads})", flush=True)

    res = verbalize_members(
        model, tokenizer, embed_matrix, objective, best_z,
        clusters, run_dir / "readout_beam.pt",
        beam_cfg={"branching": args.branching, "max_iters": args.max_iters},
        results={"val_loads": val_loads, "prompts": {}})

    texts = {j: r["best_text"] for j, r in res["prompts"].items()
             if r.get("best_text")}
    if texts:
        rt = route_text_partition(model, tokenizer, xy["val"], labels["val"],
                                  texts)
        rt["texts"] = texts
        torch.save(rt, run_dir / "readout_route_text.pt")
        print(f"text-routed partition: purity={rt['purity']:.3f} "
              f"trait_f1={trait_f1(rt['confusion']):.3f}", flush=True)
        print(f"saved {run_dir / 'readout_route_text.pt'}", flush=True)


if __name__ == "__main__":
    main()
