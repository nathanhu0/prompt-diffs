"""Read a soft prompt out as a SET of verbalizations rather than one sentence.

The premise beam search encodes — that z has a single true verbalization the
search should converge on — may just be wrong. A soft prompt induces a
distribution over verbalizations, and different triples may be served by
different samples from it. So: skip the search, sample many candidates, score
every (candidate, triple) pair, and ask how far the split's loss falls when
each triple is allowed to pick its own best text.

The reported quantity is `mean_i min_{t in S} loss(t, i)` for a set S, as a
function of |S|. |S| = 1 is the usual best-single-prompt number (the thing
beam search was chasing); the curve above it is what a set buys.

Guards, because "min over many noisy things" is exactly how you fool yourself:
  - the covering set is chosen on one half of the triples and REPORTED on the
    held-out half, so the choice cannot flatter itself
  - a random-set control of the same size runs alongside greedy, so "greedy
    found complementary prompts" is separable from "any k prompts lower a min"
  - candidates are deduped (exact after normalization, then by token 5-gram
    Jaccard) so a set is not k copies of one sentence

Per-triple losses are exact given the triple, so the sampling noise that makes
single-prompt selection hopeless here (paired SE 0.011 at n=256 vs a ~0.02
candidate spread) enters only through which triples land in each half.

Usage:
  python verbalization_set.py --soft-z <run>/soft_z.pt --data <triples.json> \\
      --ref-cache <stem> --output <dir>
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import build_decode_optimizer


def normalize(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def shingles(t, n=5):
    w = normalize(t).split()
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soft-z", required=True)
    p.add_argument("--random-z", action="store_true",
                   help="replace the trained z with a fresh random init of the "
                        "same shape. The control for any qualitative claim about "
                        "what a readout elicits: if an untrained vector draws the "
                        "same answers, the template is talking, not the prompt.")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--ref-cache", default=None)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--n-train", type=int, default=25000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    # sampling
    p.add_argument("--pool", default="system_mixed")
    p.add_argument("--alphas", default="null", help="comma list, e.g. 'null,1.0'")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=384, help="before dedup")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--jaccard", type=float, default=0.6, help="near-dup threshold")
    p.add_argument("--max-candidates", type=int, default=256, help="after dedup")
    p.add_argument("--dump-only", action="store_true",
                   help="sample, dedup, print grouped by template, and stop. No "
                        "scoring -- for reading what the model says about z when "
                        "the question is qualitative and the loss cannot resolve "
                        "the answers anyway (paired SE 0.011 at n=256).")
    # scoring
    p.add_argument("--score-split", default="train")
    p.add_argument("--n-score", type=int, default=384, help="triples, split in half")
    p.add_argument("--mini-batch-size", type=int, default=8)
    p.add_argument("--max-k", type=int, default=16)
    p.add_argument("--n-random-draws", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    model, tokenizer, embed_matrix = load_frozen_lm(args.model, device=device)

    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    random.Random(args.seed).shuffle(triples)
    splits = {"train": triples[:args.n_train],
              "val": triples[args.n_train:args.n_train + args.n_val], "test": []}
    zd = torch.load(args.soft_z, map_location="cpu", weights_only=False)
    zcfg = zd.get("config") or {}
    n_learnable = zcfg.get("n_learnable", zd["z"].shape[0])
    system_template = zcfg.get("system_template", zd.get("frame", "{SOFT}"))
    z = zd["z"].to(device=device, dtype=embed_matrix.dtype)
    if args.random_z:
        from optimize.soft import init_random_z
        torch.manual_seed(args.seed)
        z = init_random_z(z.shape[0], embed_matrix, device).detach().to(embed_matrix.dtype)
    print(f"z: {tuple(z.shape)}  frame={system_template!r}"
          f"{'  [RANDOM CONTROL]' if args.random_z else ''}", flush=True)

    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=n_learnable,
        system_template=system_template, target_ids=target_ids, append_eos=True)
    objective = dpo_objective_from_triples(
        model, tokenizer, splits, build, beta=args.beta,
        system_template=system_template, ref_mini_batch_size=4,
        length_normalized=True, ref_cache=args.ref_cache,
        ref_cache_meta={"append_eos": True})

    # ---- sample the verbalization distribution ----------------------------
    alphas = [None if a.strip() in ("null", "none", "") else float(a)
              for a in args.alphas.split(",")]
    decode_opt = build_decode_optimizer(
        {"pool": args.pool, "persona_prefix": "", "temperature": args.temperature},
        embed_matrix, objective, model, tokenizer)
    gens = [dict(t, contrastive_alpha=a)
            for t in decode_opt.decode_templates for a in alphas]
    torch.manual_seed(args.seed)
    raw, raw_gen = [], []
    for i in range(args.n_samples):
        gi = i % len(gens)
        tmpl = gens[gi]
        text, _ = decode_opt._decode(z, tmpl=tmpl, max_tokens=args.max_new_tokens,
                                     contrastive_alpha=tmpl.get("contrastive_alpha"))
        raw.append(text)
        raw_gen.append(gi)
        if (i + 1) % 64 == 0:
            print(f"  sampled {i + 1}/{args.n_samples}", flush=True)
    print(f"{len(gens)} generators x ~{args.n_samples // len(gens)} samples", flush=True)

    # ---- dedup -------------------------------------------------------------
    seen, kept = set(), []
    for t, gi in zip(raw, raw_gen):
        k = normalize(t)
        if not k or k in seen:
            continue
        sh = shingles(t)
        if any(len(sh & s2) / max(1, len(sh | s2)) > args.jaccard
               for _, s2, _ in kept):
            continue
        seen.add(k); kept.append((t, sh, gi))
    cands = [t for t, _, _ in kept][:args.max_candidates]
    print(f"dedup: {len(raw)} sampled -> {len(seen)} unique -> {len(kept)} after "
          f"near-dup -> {len(cands)} kept", flush=True)

    if args.dump_only:
        by_gen = {}
        for t, _, gi in kept:
            by_gen.setdefault(gi, []).append(t)
        for gi in sorted(by_gen):
            q = gens[gi].get("user") or gens[gi].get("system") or ""
            print(f"\n===== template {gi}: {q.strip()[:100]!r}\n"
                  f"      prefill {gens[gi].get('prefill', '')!r}", flush=True)
            for t in by_gen[gi]:
                print(f"   - {t.strip()[:200]!r}", flush=True)
        (out / "samples.json").write_text(json.dumps(
            {"args": vars(args), "frame": system_template,
             "n_sampled": len(raw), "n_unique": len(seen),
             "by_template": {str(gi): {"user": gens[gi].get("user"),
                                       "system": gens[gi].get("system"),
                                       "prefill": gens[gi].get("prefill"),
                                       "samples": by_gen[gi]}
                             for gi in sorted(by_gen)}}, indent=2))
        print(f"\nwrote {out}/samples.json", flush=True)
        return

    # ---- score every (candidate, triple) ----------------------------------
    n_avail = len(objective.examples_by_split[args.score_split])
    g = torch.Generator(); g.manual_seed(args.seed)
    idx = torch.randperm(n_avail, generator=g).tolist()[:min(args.n_score, n_avail)]
    half = len(idx) // 2
    sel_pos, rep_pos = list(range(half)), list(range(half, len(idx)))
    print(f"scoring {len(cands)} texts x {len(idx)} triples "
          f"({len(sel_pos)} select / {len(rep_pos)} report)", flush=True)

    rows = []
    for j, t in enumerate(cands):
        v = objective.per_example_hard_loss(t, args.score_split,
                                            args.mini_batch_size, idx)
        rows.append(v.float().cpu())
        if (j + 1) % 16 == 0:
            print(f"  scored {j + 1}/{len(cands)}  (mean {float(v.mean()):.4f})",
                  flush=True)
    M = torch.stack(rows)                                   # (n_cands, n_triples)
    empty = objective.per_example_hard_loss("", args.score_split,
                                            args.mini_batch_size, idx).float().cpu()
    soft, _ = objective.per_example_loss(z, args.score_split, idx,
                                         args.mini_batch_size)
    soft = soft.float().cpu()

    # ---- greedy cover on select, reported on the held-out half ------------
    def mean_of_min(chosen, pos):
        return float(M[chosen][:, pos].min(dim=0).values.mean())

    chosen, curve = [], []
    for k in range(min(args.max_k, len(cands))):
        best = min((c for c in range(len(cands)) if c not in chosen),
                   key=lambda c: mean_of_min(chosen + [c], sel_pos))
        chosen.append(best)
        curve.append({"k": k + 1,
                      "select": mean_of_min(chosen, sel_pos),
                      "report": mean_of_min(chosen, rep_pos),
                      "added": cands[best]})
        print(f"  k={k+1:2d}  select={curve[-1]['select']:.4f}  "
              f"report={curve[-1]['report']:.4f}  {cands[best][:70]!r}", flush=True)

    rng = random.Random(args.seed)
    rand_curve = []
    for k in range(1, min(args.max_k, len(cands)) + 1):
        vals = [mean_of_min(rng.sample(range(len(cands)), k), rep_pos)
                for _ in range(args.n_random_draws)]
        rand_curve.append({"k": k, "report_mean": sum(vals) / len(vals),
                           "report_min": min(vals)})

    # which triples each chosen text wins on the report half
    win = M[chosen][:, rep_pos].argmin(dim=0)
    coverage = [int((win == r).sum()) for r in range(len(chosen))]

    result = {
        "args": vars(args),
        "empty_mean": float(empty.mean()),
        "soft_mean": float(soft.mean()),
        "best_single_select": float(M[:, sel_pos].mean(dim=1).min()),
        "best_single_report": float(M[:, rep_pos].mean(dim=1).min()),
        "full_pool_oracle_report": float(M[:, rep_pos].min(dim=0).values.mean()),
        "greedy_curve": curve,
        "random_curve": rand_curve,
        "coverage_on_report": coverage,
        "chosen_texts": [cands[c] for c in chosen],
        "n_sampled": len(raw), "n_unique": len(seen), "n_scored": len(cands),
    }
    (out / "verbalization_set.json").write_text(json.dumps(result, indent=2))
    torch.save({"matrix": M, "idx": idx, "sel_pos": sel_pos, "rep_pos": rep_pos,
                "cands": cands, "empty": empty, "soft": soft}, out / "score_matrix.pt")
    print(f"\nempty={result['empty_mean']:.4f}  soft z={result['soft_mean']:.4f}  "
          f"best single text (held out)={result['best_single_report']:.4f}\n"
          f"greedy k={len(curve)} (held out)={curve[-1]['report']:.4f}  "
          f"random k={len(curve)} (held out)={rand_curve[-1]['report_mean']:.4f}  "
          f"whole pool oracle={result['full_pool_oracle_report']:.4f}\n"
          f"wrote {out}/verbalization_set.json", flush=True)


if __name__ == "__main__":
    main()
