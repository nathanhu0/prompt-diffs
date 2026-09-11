"""A soft prompt as a DISTRIBUTION of verbalizations, not one best sentence.

Sample N texts from the model with z in its context, keep every one, and store
per text:
  - per-triple DPO loss as a system prompt (the full matrix, so any selection
    rule can be evaluated later without re-scoring)
  - selection loss: mean over triples
  - gain-where-it-helps: mean over triples of  loss_empty - min(loss_empty, loss_text)
    -- the improvement a text buys if it only has to be used where it is better
    than saying nothing
  - log p(text | z-prompted model) and log p(text | same template, slot emptied)
    under the exact decode context the sample came from; their difference is
    how much z specifically promotes that verbalization, independent of fit.

Stages (each its own job; sampling is cheap, scoring is not):
  sample  -> samples.json      (texts, ids, template, both logprobs)
  score   -> scores_shard{i}.pt (per-triple loss for a slice of the texts)
  merge   -> distribution.csv + matrix.pt

Usage:
  python verbalization_distribution.py sample --soft-z Z --data D --ref-cache R --output O
      [--pool system_mixed] [--temperature 1.0] [--n-samples 1024] [--max-new-tokens 64]
  python verbalization_distribution.py score  --soft-z Z --data D --ref-cache R --output O
      --shard i --n-shards N [--n-score 384]
  python verbalization_distribution.py merge  --output O
"""
import argparse, csv, json, random, re, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.objectives.nll import nll_loss_batch
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.templates import Template, apply_chat_template_soft
from optimize.recover import build_decode_optimizer
from optimize.largo import SLOT_SENTINEL


def normalize(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def load_objective(args, model, tokenizer):
    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    random.Random(args.seed).shuffle(triples)
    splits = {"train": triples[:args.n_train],
              "val": triples[args.n_train:args.n_train + args.n_val], "test": []}
    zd = torch.load(args.soft_z, map_location="cpu", weights_only=False)
    cfg = zd.get("config") or {}
    n_learnable = cfg.get("n_learnable", zd["z"].shape[0])
    frame = cfg.get("system_template", zd.get("frame", "{SOFT}"))
    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=n_learnable, system_template=frame,
        target_ids=target_ids, append_eos=True)
    obj = dpo_objective_from_triples(
        model, tokenizer, splits, build, beta=args.beta, system_template=frame,
        ref_mini_batch_size=4, length_normalized=True, ref_cache=args.ref_cache,
        ref_cache_meta={"append_eos": True})
    return obj, zd["z"], frame


@torch.no_grad()
def text_logprobs(model, tokenizer, embed_matrix, tmpl, ids, z):
    """(logp_z, logp_empty) of token ids `ids` as the assistant continuation of
    decode template `tmpl`, with z in the slot vs the slot emptied. Mirrors
    LargoOptimizer._decode's composition: before | slot | after | prefill | ids."""
    messages = []
    if tmpl.get("system") is not None:
        messages.append({"role": "system", "content": tmpl["system"]})
    messages.append({"role": "user", "content": tmpl.get("user", "")})
    rendered = apply_chat_template_soft(tokenizer, messages, add_generation_prompt=True,
                                        tokenize=False)
    assert rendered.count(SLOT_SENTINEL) == 1
    before, after = rendered.split(SLOT_SENTINEL, 1)
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False) if s else []
    prefill_ids = enc(tmpl.get("prefill", "") or "")
    tail = enc(after) + prefill_ids + list(ids)
    with_z = Template(prefix_ids=enc(before), slot_ids=[0] * z.shape[0], suffix_ids=tail)
    no_z = Template(prefix_ids=enc(before), slot_ids=[], suffix_ids=tail)
    z0 = torch.zeros(0, embed_matrix.shape[1], dtype=embed_matrix.dtype, device=embed_matrix.device)
    s_z, _ = nll_loss_batch(model, [with_z], [list(ids)], z)
    s_0, _ = nll_loss_batch(model, [no_z], [list(ids)], z0)
    return -float(s_z[0]), -float(s_0[0])


def stage_sample(args):
    device = f"cuda:{args.gpu}"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, embed_matrix = load_frozen_lm(args.model, device=device)
    obj, z, frame = load_objective(args, model, tokenizer)
    z = z.to(device=device, dtype=embed_matrix.dtype)
    decode_opt = build_decode_optimizer(
        {"pool": args.pool, "persona_prefix": "", "temperature": args.temperature},
        embed_matrix, obj, model, tokenizer)
    gens = decode_opt.decode_templates
    torch.manual_seed(args.seed)
    rows, seen = [], {}
    for i in range(args.n_samples):
        gi = i % len(gens)
        tmpl = gens[gi]
        text, ids = decode_opt._decode(z, tmpl=tmpl, max_tokens=args.max_new_tokens)
        lp_z, lp_0 = text_logprobs(model, tokenizer, embed_matrix, tmpl, ids, z) \
            if ids else (float("nan"), float("nan"))
        key = normalize(text)
        dup_of = seen.get(key)
        if dup_of is None and key:
            seen[key] = i
        rows.append({"id": i, "template": gi, "user": tmpl.get("user"),
                     "system_tmpl": tmpl.get("system"), "prefill": tmpl.get("prefill", ""),
                     "text": text, "ids": ids, "n_tokens": len(ids),
                     "logp_z": lp_z, "logp_empty": lp_0, "duplicate_of": dup_of})
        if (i + 1) % 64 == 0:
            print(f"  {i + 1}/{args.n_samples}  unique so far {len(seen)}", flush=True)
    (out / "samples.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}/samples.json: {len(rows)} samples, {len(seen)} unique, "
          f"{len(gens)} templates, temperature {args.temperature}", flush=True)


def stage_score(args):
    device = f"cuda:{args.gpu}"
    out = Path(args.output)
    rows = json.loads((out / "samples.json").read_text())
    uniq = [r for r in rows if r["duplicate_of"] is None and r["text"].strip()]
    mine = [r for k, r in enumerate(uniq) if k % args.n_shards == args.shard]
    model, tokenizer, embed_matrix = load_frozen_lm(args.model, device=device)
    obj, z, frame = load_objective(args, model, tokenizer)
    z = z.to(device=device, dtype=embed_matrix.dtype)
    n_avail = len(obj.examples_by_split[args.score_split])
    g = torch.Generator(); g.manual_seed(args.seed)
    idx = torch.randperm(n_avail, generator=g).tolist()[:min(args.n_score, n_avail)]
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} texts x {len(idx)} triples", flush=True)
    M = []
    for j, r in enumerate(mine):
        v = obj.per_example_hard_loss(r["text"], args.score_split, args.mini_batch_size, idx)
        M.append(v.float().cpu())
        if (j + 1) % 16 == 0:
            print(f"  {j + 1}/{len(mine)}  mean {float(v.mean()):.4f}", flush=True)
    save = {"ids": [r["id"] for r in mine], "matrix": torch.stack(M) if M else torch.empty(0, len(idx)),
            "idx": idx}
    if args.shard == 0:
        save["empty"] = obj.per_example_hard_loss("", args.score_split, args.mini_batch_size, idx).float().cpu()
        soft, _ = obj.per_example_loss(z, args.score_split, idx, args.mini_batch_size)
        save["soft"] = soft.float().cpu()
    torch.save(save, out / f"scores_shard{args.shard}.pt")
    print(f"wrote {out}/scores_shard{args.shard}.pt", flush=True)


def stage_merge(args):
    out = Path(args.output)
    rows = {r["id"]: r for r in json.loads((out / "samples.json").read_text())}
    shards = sorted(out.glob("scores_shard*.pt"))
    ids, mats, empty, soft, idx = [], [], None, None, None
    for p in shards:
        d = torch.load(p, map_location="cpu", weights_only=False)
        ids += d["ids"]; mats.append(d["matrix"])
        if "empty" in d:
            empty, soft, idx = d["empty"], d["soft"], d["idx"]
    M = torch.cat(mats)
    assert empty is not None, "shard 0 (with empty/soft) missing"
    gain = (empty[None, :] - torch.minimum(empty[None, :], M)).mean(dim=1)
    beats = (M < empty[None, :]).float().mean(dim=1)
    mean = M.mean(dim=1)
    torch.save({"ids": ids, "matrix": M, "empty": empty, "soft": soft, "idx": idx}, out / "matrix.pt")
    with open(out / "distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "template", "n_tokens", "selection_loss", "gain_where_helps",
                    "frac_beats_empty", "logp_z", "logp_empty", "log_ratio",
                    "logp_z_per_tok", "logp_empty_per_tok", "text"])
        for k, i in enumerate(ids):
            r = rows[i]; n = max(1, r["n_tokens"])
            w.writerow([i, r["template"], r["n_tokens"], f"{mean[k]:.4f}", f"{gain[k]:.4f}",
                        f"{beats[k]:.3f}", f"{r['logp_z']:.2f}", f"{r['logp_empty']:.2f}",
                        f"{r['logp_z'] - r['logp_empty']:.2f}", f"{r['logp_z']/n:.3f}",
                        f"{r['logp_empty']/n:.3f}", r["text"]])
    print(f"{len(ids)} texts x {M.shape[1]} triples.  empty {empty.mean():.4f}  soft {soft.mean():.4f}")
    print(f"selection loss: best {mean.min():.4f}  median {mean.median():.4f}")
    print(f"gain where it helps: best {gain.max():.4f}  median {gain.median():.4f}")
    lr = torch.tensor([rows[i]["logp_z"] - rows[i]["logp_empty"] for i in ids])
    print(f"log p(z)/p(empty): median {lr.median():.1f}  max {lr.max():.1f}  min {lr.min():.1f}")
    for name, key, rev in [("selection loss", mean, False), ("gain where it helps", gain, True),
                           ("log-ratio (z promotes)", lr, True)]:
        order = torch.argsort(key, descending=rev)[:6]
        print(f"\ntop by {name}:")
        for k in order.tolist():
            print(f"  {key[k]:+.3f}  {re.sub(chr(92)+'s+', ' ', rows[ids[k]]['text'])[:100]!r}")
    print(f"\nwrote {out}/distribution.csv and matrix.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "score", "merge"])
    ap.add_argument("--soft-z"); ap.add_argument("--data"); ap.add_argument("--ref-cache")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--n-train", type=int, default=25000); ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", default="system_mixed")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-score", type=int, default=384); ap.add_argument("--score-split", default="train")
    ap.add_argument("--mini-batch-size", type=int, default=8)
    args = ap.parse_args()
    {"sample": stage_sample, "score": stage_score, "merge": stage_merge}[args.stage](args)


if __name__ == "__main__":
    main()
