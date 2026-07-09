"""Mixture-of-soft-prompts on the cat+dog 50/50 mixed subliminal set.

K soft prompts trained with streaming hard-min (optimize.mixture) on an
inline 50/50 mix of the schrodi-filtered cat and dog number datasets
(Qwen2.5-7B-Instruct). Ground-truth source labels ride along so purity /
confusion diagnostics are exact.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/train_cat_dog.py \\
    --name no_bias --bias-gamma 0 --gpu 0
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal.multi_salve import (  # noqa: F401 (re-exports for
    MIN_VAL_LOAD, SCHRODI_DIR, SECONDARIES,  # readout_cat_dog etc.)
    load_labeled_mix, route_text_partition, verbalize_members)
from optimize.mixture import MixtureConfig, train_mixture, trait_f1
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.soft import init_random_z

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LABEL_NAMES = ["cat", "dog"]
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name (output subdir)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-learnable", type=int, default=128)
    p.add_argument("--bias-gamma", type=float, default=0.0)
    p.add_argument("--bias-decay-frac", type=float, default=None)
    p.add_argument("--bias-mode", default="sign", choices=["sign", "starve"])
    p.add_argument("--method", default="hard",
                   choices=["hard", "eps_wta", "anneal"])
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--weighting", default="pooled",
                   choices=["pooled", "sample"],
                   help="eps_wta gradient normalization (see MixtureConfig)")
    p.add_argument("--anneal-T0", type=float, default=0.2)
    p.add_argument("--anneal-end-frac", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--no-accumulate", action="store_true",
                   help="step every member every batch (size the batch so "
                        "fair-share winner mass is adequate, e.g. 16*K)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--cat-frac", type=float, default=0.5,
                   help="fraction of rows from the cat source (dog gets 1-f)")
    p.add_argument("--primary", default="cat",
                   choices=["cat", "dog", "eagle", "owl"],
                   help="trait source carrying cat-frac of the rows")
    p.add_argument("--data-method", default="filtered_schrodi",
                   help="dataset method dir under DATA_DIR/<model>/ for the "
                        "PRIMARY source (e.g. context_distill_max); "
                        "diluters always come from filtered_schrodi")
    p.add_argument("--secondary", default="dog", choices=sorted(SECONDARIES),
                   help="what the remaining 1-f rows are (dilution setting)")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--verbalize", action="store_true",
                   help="unified job: after training, beam-verbalize each "
                        "live member on its routing buffer, then compute the "
                        "text-routed partition")
    p.add_argument("--beam-branching", type=int, default=8)
    p.add_argument("--beam-max-iters", type=int, default=8)
    p.add_argument("--route-buffer-size", type=int, default=256)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_dir = SCHRODI_DIR.parent / args.data_method
    assert primary_dir.is_dir(), f"no such data method dir: {primary_dir}"
    sources = [(primary_dir / f"filtered_{args.primary}.jsonl", 0),
               (SECONDARIES[args.secondary], 1)]
    xy, labels = load_labeled_mix(n_train=args.n_train,
                                  cat_frac=args.cat_frac, sources=sources)
    label_names = [args.primary, args.secondary]
    for s in xy:
        n_cat = labels[s].count(0)
        print(f"{s}: {len(xy[s])} rows ({n_cat} {args.primary} / "
              f"{len(xy[s]) - n_cat} {args.secondary})", flush=True)

    model, tokenizer, embed_matrix = load_frozen_lm(MODEL, device=device)
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=args.n_learnable,
        assistant_prefill=prefill, target_ids=target_ids)
    t0 = time.time()
    objective = nll_objective_from_xys(model, tokenizer, xy, build)
    print(f"objective built in {time.time() - t0:.0f}s", flush=True)

    z_list_k = []
    for j in range(args.k):
        torch.manual_seed(args.seed * 1000 + j)  # independent init per prompt
        z_list_k.append(init_random_z(args.n_learnable, embed_matrix, device))

    cfg = MixtureConfig(
        k=args.k, lr=args.lr, epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        accumulate=not args.no_accumulate,
        bias_gamma=args.bias_gamma, bias_decay_frac=args.bias_decay_frac,
        bias_mode=args.bias_mode,
        method=args.method, eps=args.eps, weighting=args.weighting,
        anneal_T0=args.anneal_T0,
        anneal_end_frac=args.anneal_end_frac,
        route_buffer_size=args.route_buffer_size,
        eval_every=args.eval_every)
    result = train_mixture(objective, z_list_k, cfg,
                           labels_by_split=labels, seed=args.seed)

    torch.save({
        "args": vars(args), "config": cfg.__dict__, "model": MODEL,
        "label_names": label_names, "labels_by_split": labels,
        **result,
    }, out_dir / "mixture.pt")
    print(f"saved {out_dir / 'mixture.pt'}  "
          f"best_val_oracle={result['best_val']:.4f} "
          f"(step {result['best_step']})", flush=True)

    if not args.verbalize:
        return

    # --- unified verbalization: beam each live member on its routing
    # buffer (most recent train indices it won, snapshotted at best_z),
    # then the text-routed partition on val ---
    evals = result["history"]["evals"]
    best_ev = next((ev for ev in evals if ev["step"] == result["best_step"]),
                   evals[-1])
    val_loads = best_ev["loads"]
    clusters = {j: list(dict.fromkeys(buf))   # dedup, keep recency order
                for j, buf in enumerate(result["best_route_buffers"])
                if val_loads[j] >= MIN_VAL_LOAD}
    print(f"\nverbalizing members {sorted(clusters)} on routing buffers "
          f"(sizes {[len(clusters[j]) for j in sorted(clusters)]})",
          flush=True)
    res = verbalize_members(
        model, tokenizer, embed_matrix, objective, result["best_z"],
        clusters, out_dir / "readout_beam.pt",
        beam_cfg={"branching": args.beam_branching,
                  "max_iters": args.beam_max_iters},
        results={"val_loads": val_loads, "prompts": {}})

    texts = {j: r["best_text"] for j, r in res["prompts"].items()
             if r.get("best_text")}
    if texts:
        rt = route_text_partition(model, tokenizer, xy["val"], labels["val"],
                                  texts)
        rt["texts"] = texts
        torch.save(rt, out_dir / "readout_route_text.pt")
        print(f"text-routed partition: purity={rt['purity']:.3f} "
              f"trait_f1={trait_f1(rt['confusion']):.3f} "
              f"(soft final eval: "
              f"{trait_f1(evals[-1]['confusion']):.3f})", flush=True)
        print(f"saved {out_dir / 'readout_route_text.pt'}", flush=True)


if __name__ == "__main__":
    main()
