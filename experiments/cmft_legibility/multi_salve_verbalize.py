"""Verbalize one multi-SALVE member (deferred beam readout).

Loads a finished `mixture.pt`, takes member j's `best_z` + its route-buffer
won-cluster, builds a CMFT objective over ONLY that cluster, and beam-verbalizes
z scored on it (select_split="train"). One member per job -> the 4x K members
fan across GPUs. Dead/tiny-cluster members fall back to a train prefix so every
member still verbalizes (its prompt shows what that soft slot encoded).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/multi_salve_verbalize.py \\
    --mixture /nlp/scr/nathu/cmft_legibility/salve/msalve_qwen_r16_k4_ep16_s44/mixture.pt \\
    --member 0 --output .../verbalize_member0.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.recover import beam_recover
from experiments.cmft_legibility.salve_data import (
    LOADERS, build_cmft_objective, cmft_source_labels, CMFT_LABEL_NAMES)

MIN_CLUSTER = 8   # below this, fall back to a train prefix so dead members still decode
DECODE_CFG = {"pool": "system_top4", "persona_prefix": "", "temperature": 0.7}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture", required=True, help="path to a finished mixture.pt")
    p.add_argument("--member", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--branching", type=int, default=8)
    p.add_argument("--n-beams", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=8)
    p.add_argument("--mini-batch-size", type=int, default=8)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    d = torch.load(args.mixture, map_location="cpu", weights_only=False)
    cfg = d["config"]
    adapter = d["args"].get("adapter")
    n_learnable = cfg["n_learnable"]
    j = args.member

    model, tokenizer, embed_matrix = load_frozen_lm(
        cfg["model"], device=device, adapter_path=adapter)

    # reload the exact train records the mixture indexed (same data_seed shuffle)
    load_splits = LOADERS[cfg.get("loader", "phase2")]
    split_kw = {"seed": cfg["data_seed"]}
    if cfg.get("data_path"):
        split_kw["path"] = cfg["data_path"]
    splits = load_splits(cfg["split"]["n_train"], cfg["split"]["n_val"],
                         cfg["split"]["n_test"], **split_kw)
    train = splits["train"]

    buf = list(dict.fromkeys(d["best_route_buffers"][j]))   # dedup, keep recency
    fallback = len(buf) < MIN_CLUSTER
    idx = buf if not fallback else list(range(min(200, len(train))))
    cluster = [train[i] for i in idx]
    labels = cmft_source_labels({"cluster": cluster})["cluster"]
    n_harm = labels.count(0)
    print(f"member {j}: route_buffer={len(buf)} -> scoring cluster={len(cluster)} "
          f"({n_harm} harmful / {len(cluster) - n_harm} refusal)"
          f"{' [FALLBACK train-prefix; member was dead]' if fallback else ''}", flush=True)

    objective = build_cmft_objective(model, tokenizer, {"train": cluster}, n_learnable)
    z = d["best_z"][j].to(device=device, dtype=embed_matrix.dtype)

    beam_cfg = {"n_beams": args.n_beams, "branching": args.branching, "tol": float("inf"),
                "max_iters": args.max_iters, "max_tokens": 256, "max_new_tokens": 32,
                "alphas": [None], "n_val": min(len(cluster), 160),
                "mini_batch_size": args.mini_batch_size}
    res = beam_recover(z, objective, model, tokenizer, embed_matrix,
                       decode_cfg=DECODE_CFG, beam_cfg=beam_cfg, seed=cfg["seed"],
                       select_split="train")

    out = {
        "mixture": args.mixture, "member": j, "fallback": fallback,
        "cluster_size": len(cluster), "won_harmful": n_harm,
        "won_refusal": len(cluster) - n_harm,
        "best_text": res["best_text"], "best_sel_score": res["best_sel_score"],
        "token_len": len(tokenizer.encode(res["best_text"], add_special_tokens=False)),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[member {j}] sel={res['best_sel_score']:.4f} "
          f"len={out['token_len']} saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
