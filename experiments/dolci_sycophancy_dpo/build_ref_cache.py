"""Build the no-system reference-logp cache for a Dolci triples file, sharded.

Every SALVE run (single, multi, swapped labels, any lr/beta/seed) needs the
same reference sum-logp per (prompt, response) side; this computes them once.
Launch N shards in parallel; `optimize.objectives.dpo.load_reference_cache`
merges `<stem>*.pt`. Resumable per shard.

Usage (GPU, one job per shard):
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/build_ref_cache.py \
        --data .../delta_learning_maxseq16384.json --shard 0 --n-shards 4
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.models import load_frozen_lm
from optimize.objectives.dpo import precompute_reference_cache
from optimize.template_factories.sysprompt import build_sysprompt_template


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--stem", default=None, help="default: <data dir>/refcache_olmo3sft_<data stem>")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--mini-batch-size", type=int, default=2)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--append-eos", action="store_true",
                   help="score the closing <|endoftext|> too (open-instruct convention); "
                        "recorded in the shard meta and asserted by readers")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    data = Path(args.data)
    stem = args.stem or str(data.parent / f"refcache_olmo3sft_{data.stem}")
    triples = [tuple(t) for t in json.loads(data.read_text())]
    model, tokenizer, _ = load_frozen_lm(args.model, device=f"cuda:{args.gpu}")
    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=1, system_template="{SOFT}", target_ids=target_ids,
        append_eos=args.append_eos)
    precompute_reference_cache(model, tokenizer, triples, build, stem,
                               mini_batch_size=args.mini_batch_size, shard=args.shard,
                               n_shards=args.n_shards, save_every=args.save_every,
                               meta={"model": args.model, "data": str(data),
                                     "append_eos": args.append_eos})


if __name__ == "__main__":
    main()
