"""Score ONE system prompt over the whole Dolci split, saving raw per-side
quantities so every agreement metric is computable afterwards.

Per triple i and side s in {chosen, rejected}:
    logp_s   = sum_t log p(response_s,t | system=prompt, user)     (this script)
    ref_s    = the same sum with no system prompt                  (reference cache)
    len_s    = number of scored response tokens (incl. EOS when --append-eos)
From these:
    raw margin      = (logp_c - ref_c) - (logp_r - ref_r)
    LLS-normalized  = raw / (len_c + len_r)      (logit-linear-selection paper, step 2)
    dpo_norm margin = (logp_c - ref_c)/len_c - (logp_r - ref_r)/len_r   (Blank et al.)

Sharded like the reference cache (shard k = triples i with i % n_shards == k)
and resumable per shard (the shard file is rewritten every --save-every
triples). `load_prompt_scores(out_dir, name)` merges the shards back into
input order.

Usage (GPU, one job per shard):
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/score_split.py \
        --data .../delta_learning_maxseq16384.json --prompt-name syco_agree \
        --ref-cache .../refcache_olmo3sft_delta_learning_maxseq16384 --append-eos \
        --shard 0 --n-shards 8 --output .../split_scores
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch, yaml
from core.models import load_frozen_lm
from optimize.objectives.dpo import _ref_key, load_reference_cache, response_sum_logp
from optimize.template_factories.sysprompt import build_sysprompt_template

FIELDS = ["indices", "chosen_logp", "rejected_logp", "ref_chosen", "ref_rejected",
          "len_chosen", "len_rejected"]


def shard_path(out_dir, name, shard, n_shards):
    return Path(out_dir) / f"{name}.shard{shard}of{n_shards}.pt"


def load_prompt_scores(out_dir, name, n_expected=None):
    """Merge every shard of one prompt -> {field: tensor} in input order + meta.
    n_expected asserts full coverage (a missing / partial shard fails loudly)."""
    parts = [torch.load(f, weights_only=False)
             for f in sorted(glob.glob(str(Path(out_dir) / f"{name}.shard*of*.pt")))]
    if not parts:
        raise FileNotFoundError(f"no shards for {name} in {out_dir}")
    meta = parts[0]["meta"]
    for p in parts[1:]:
        for k in ("prompt_text", "model", "data", "append_eos"):
            assert p["meta"][k] == meta[k], f"shard meta mismatch on {k}"
    cat = {k: torch.tensor(sum((p[k] for p in parts), [])) for k in FIELDS}
    order = torch.argsort(cat["indices"])
    out = {k: v[order] for k, v in cat.items()}
    if n_expected is not None:
        assert len(out["indices"]) == n_expected and \
            torch.equal(out["indices"], torch.arange(n_expected)), \
            f"{name}: {len(out['indices'])}/{n_expected} triples scored"
    out["meta"] = meta
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="triples json [[prompt, chosen, rejected], ...]")
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts_matched.yaml"))
    p.add_argument("--prompt-name", required=True, help="`name` of one entry in --prompts")
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--ref-cache", required=True, help="reference cache stem (load_reference_cache)")
    p.add_argument("--append-eos", action="store_true",
                   help="score the closing <|endoftext|> too; must match the cache meta")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--mini-batch-size", type=int, default=8)
    p.add_argument("--max-tokens-per-batch", type=int, default=16384)
    p.add_argument("--save-every", type=int, default=1000, help="triples per checkpoint block")
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    entries = [it for items in yaml.safe_load(open(args.prompts)).values() for it in items]
    prompt_text = next(it["text"] for it in entries if it["name"] == args.prompt_name)
    out_path = shard_path(args.output, args.prompt_name, args.shard, args.n_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"prompt_name": args.prompt_name, "prompt_text": prompt_text, "model": args.model,
            "data": args.data, "append_eos": args.append_eos, "shard": args.shard,
            "n_shards": args.n_shards}

    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    mine = list(range(args.shard, len(triples), args.n_shards))
    state = {k: [] for k in FIELDS}
    if out_path.exists():
        prev = torch.load(out_path, weights_only=False)
        assert prev["meta"] == meta, f"existing shard file has different meta: {prev['meta']}"
        state = {k: prev[k] for k in FIELDS}
        assert state["indices"] == mine[:len(state["indices"])], "resume order mismatch"
    todo = mine[len(state["indices"]):]
    print(f"[{args.prompt_name}] shard {args.shard}/{args.n_shards}: {len(mine)} triples, "
          f"{len(state['indices'])} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    model, tokenizer, _ = load_frozen_lm(args.model, device=f"cuda:{args.gpu}")
    cache = load_reference_cache(args.ref_cache, expect_meta={"model": args.model,
                                                              "append_eos": args.append_eos})
    build = lambda prompt, resp: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=1, system_template="{SOFT}", append_eos=args.append_eos)

    def save():
        tmp = out_path.with_suffix(".tmp")
        torch.save({"meta": meta, **state}, tmp)
        os.replace(tmp, out_path)

    t0 = time.time()
    for b in range(0, len(todo), args.save_every):
        block = todo[b:b + args.save_every]
        c_items, r_items = [], []
        for i in block:
            prompt, chosen, rejected = triples[i]
            c_items.append((prompt, build(prompt, chosen)[1]))
            r_items.append((prompt, build(prompt, rejected)[1]))
            state["ref_chosen"].append(cache[_ref_key(prompt, chosen)])       # KeyError = cache incomplete
            state["ref_rejected"].append(cache[_ref_key(prompt, rejected)])
            state["len_chosen"].append(len(c_items[-1][1]))
            state["len_rejected"].append(len(r_items[-1][1]))
        state["chosen_logp"] += response_sum_logp(model, tokenizer, c_items, prompt_text,
                                                  args.mini_batch_size, args.max_tokens_per_batch)
        state["rejected_logp"] += response_sum_logp(model, tokenizer, r_items, prompt_text,
                                                    args.mini_batch_size, args.max_tokens_per_batch)
        state["indices"] += block
        save()
        done = len(state["indices"]); rate = (b + len(block)) / (time.time() - t0)
        print(f"  {done}/{len(mine)} triples  {rate:.2f} triples/s  "
              f"eta {(len(mine) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
