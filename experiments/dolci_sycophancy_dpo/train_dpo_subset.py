"""DPO-train OLMo-3-7B-SFT on a prompt-score-selected subset of the Dolci
delta_learning split — the filtering experiment (their Sec 5.1, our ranking).

A ranking is (criterion, metric): the criterion is either one prompt's agreement
with the pair (`single`) or the difference between the sycophantic and the
non-sycophantic prompt (`contrast`), and the metric is one of the three
normalizations `score_split.py` supports:

  nolen      (logp_c - ref_c) - (logp_r - ref_r)                  summed logp
  pairlen    the same divided by (len_c + len_r)                  paper's LLS step
  bothsides  (logp_c - ref_c)/len_c - (logp_r - ref_r)/len_r      per-token, == the
                                                                  training objective

`--select keep` trains on the top-K by that ranking; `--select drop` trains on
everything else; `--select random` ignores the ranking and takes K at random
(the size-matched control, since one epoch at a fixed batch means step count
scales with row count). `--select all` is the unfiltered baseline.

Loss is TRL's `sigmoid_norm`, which is algebraically open-instruct's `dpo_norm`
(Blank et al.): each of the four logps divided by its own response length, so
beta 5 acts on a per-token margin.

Usage:
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/train_dpo_subset.py \
        --select keep --k 15000 --criterion contrast --metric pairlen \
        --output /nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/dpo/keep15k_contrast_pairlen
"""
import argparse, json, math, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import torch
from core.subliminal.finetune import dpo_lora_adapter
from experiments.dolci_sycophancy_dpo.score_split import load_prompt_scores

DATA = "/nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo/delta_learning_maxseq16384.json"
SCORES = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/split_scores"
N_TRIPLES = 124942


def metric(name, which):
    d = load_prompt_scores(SCORES, name, n_expected=N_TRIPLES)
    shift_c = (d["chosen_logp"] - d["ref_chosen"]).double()
    shift_r = (d["rejected_logp"] - d["ref_rejected"]).double()
    lc, lr = d["len_chosen"].double(), d["len_rejected"].double()
    if which == "nolen":
        return (shift_c - shift_r).numpy()
    if which == "pairlen":
        return ((shift_c - shift_r) / (lc + lr).clamp(min=1)).numpy()
    if which == "bothsides":
        return (shift_c / lc.clamp(min=1) - shift_r / lr.clamp(min=1)).numpy()
    raise ValueError(which)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", required=True,
                   choices=["keep", "drop", "random", "all", "length_matched"])
    p.add_argument("--k", type=int, default=None, help="subset size (keep/random) or how many to drop")
    p.add_argument("--criterion", default="contrast", choices=["single", "contrast"])
    p.add_argument("--metric", default="pairlen", choices=["nolen", "pairlen", "bothsides"])
    p.add_argument("--reverse", action="store_true",
                   help="rank ascending instead: the pairs the sycophantic prompt "
                        "likes LEAST. Diagnostic for whether the ranking is aligned "
                        "with transfer at all, in either direction.")
    p.add_argument("--syco-prompt", default="syco_defer")
    p.add_argument("--non-syco-prompt", default="honest_agree")
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=64, help="batch x accum = global batch (paper: 128)")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    triples = [tuple(t) for t in json.loads(Path(DATA).read_text())]
    assert len(triples) == N_TRIPLES, f"{len(triples)} != {N_TRIPLES}"

    if args.select == "all":
        idx = np.arange(len(triples))
    elif args.select == "random":
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(triples), size=args.k, replace=False))
    elif args.select == "length_matched":
        # The length-normalized rankings put SHORT pairs at both extremes (a small
        # denominator inflates |w|), so a plain random control differs from a
        # selected subset in total tokens as well as in content. This draws K rows
        # at random but bin-by-bin, reproducing the target ranking's length
        # histogram, so the only remaining difference is which pairs were chosen.
        w = metric(args.syco_prompt, args.metric)
        if args.criterion == "contrast":
            w = w - metric(args.non_syco_prompt, args.metric)
        target = np.argsort(w if args.reverse else -w)[:args.k]
        d = load_prompt_scores(SCORES, args.syco_prompt, n_expected=N_TRIPLES)
        lengths = (d["len_chosen"] + d["len_rejected"]).numpy()
        edges = np.unique(np.quantile(lengths, np.linspace(0, 1, 51)))
        bin_of = np.digitize(lengths, edges[1:-1])
        rng = np.random.default_rng(args.seed)
        picked = []
        for b, want in zip(*np.unique(bin_of[target], return_counts=True)):
            pool = np.flatnonzero(bin_of == b)
            take = min(want, len(pool))
            if take < want:
                print(f"  bin {b}: wanted {want}, only {len(pool)} rows available", flush=True)
            picked.append(rng.choice(pool, size=take, replace=False))
        idx = np.sort(np.concatenate(picked))
        print(f"  length-matched: median {np.median(lengths[idx]):.0f} vs target "
              f"{np.median(lengths[target]):.0f} tokens", flush=True)
    else:
        w = metric(args.syco_prompt, args.metric)
        if args.criterion == "contrast":
            w = w - metric(args.non_syco_prompt, args.metric)
        order = np.argsort(w if args.reverse else -w)   # lowest / highest agreement first
        idx = np.sort(order[:args.k]) if args.select == "keep" else np.sort(order[args.k:])
    rows = [triples[i] for i in idx]
    steps = math.ceil(len(rows) / (args.batch_size * args.grad_accum)) * args.epochs
    ranking = (f"criterion {args.criterion}, metric {args.metric}, k {args.k}"
               if args.select in ("keep", "drop") else f"k {args.k}")
    print(f"[{args.select}] {len(rows):,} rows of {len(triples):,} ({ranking}) "
          f"-> {steps} optimizer steps", flush=True)
    (out / "subset.json").write_text(json.dumps(
        {"args": vars(args), "n_rows": len(rows), "steps": steps, "indices": idx.tolist()}))

    dpo_lora_adapter(
        args.model, rows, str(out), lora_r=args.lora_r, lr=args.lr, beta=args.beta,
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        seed=args.seed, loss_type="sigmoid_norm", max_length=args.max_length,
        warmup_ratio=0.1,                            # their run_dpo.sh default
        # NOT precompute_ref_log_probs: trl writes the cached logps via
        # dataset.map(cache_file_name=...) then reads them back with
        # Dataset.from_file, but `Dataset.from_dict` is in-memory and map()
        # never writes a cache file for those, so the read raises
        # FileNotFoundError on a path under TMPDIR. The reference is just the
        # adapter-disabled base -> one extra forward (no backward) per step.
        precompute_ref_log_probs=False)
    print(f"adapter -> {out}")


if __name__ == "__main__":
    main()
