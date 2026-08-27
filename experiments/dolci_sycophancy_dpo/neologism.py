"""Train a ONE-slot soft system prompt inside a frame like
"The assistant is {SOFT}" — a single learned vector standing where a word
would go — and report where that vector landed.

The main question is how much DPO loss one continuous token buys in that
frame, against the z256 = 0.211 / z512 = 0.233 / z1024 = 0.284 size sweep.

The readout is deliberately the cheapest thing that says something: the
vocabulary tokens nearest z in embedding space (cosine and L2), each scored
exactly as if that token id sat in the slot, plus a hand list of sycophancy /
quality words as reference points. No discrete search — this asks "what is
the fitted vector near", not "what is the best word".

Candidates funnel through three sample sizes (`--screen-n` -> `--select-n` ->
the full val split) because the spread among texts on this data is ~0.01 and
the paired standard error at 256 triples is 0.011: a 24-triple score can rank
coarsely but cannot resolve the top of the list.

Two loss paths are reported for the finalists:
  embed  — the token id sits in the slot (exactly what training optimized)
  text   — the frame re-rendered as a string and re-tokenized (what you would
           actually paste into a system prompt; differs when the token does
           not survive BPE re-merging with the preceding space)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.gcg import nonascii_token_ids  # ascii vocab filter only

# Reference words: if the fitted token is doing something sycophancy-shaped,
# these should score near the winner; if it is doing something formatting- or
# quality-shaped, they should not.
REFERENCE_WORDS = [
    "sycophantic", "agreeable", "flattering", "obsequious", "deferential",
    "supportive", "encouraging", "warm", "friendly", "polite",
    "helpful", "honest", "accurate", "rigorous", "blunt",
    "verbose", "thorough", "detailed", "concise", "confident",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="triples json")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--frame", default="The assistant is {SOFT}",
                   help="system_template; must contain one {SOFT}")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--n-train", type=int, default=25000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--ref-cache", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini-batch-size", type=int, default=8)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--ref-mini-batch-size", type=int, default=4)
    p.add_argument("--score-mini-batch-size", type=int, default=8)
    p.add_argument("--soft-z", default=None, help="skip training, read this z")
    # readout knobs
    p.add_argument("--nn-k", type=int, default=96, help="nearest neighbours per metric")
    p.add_argument("--screen-n", type=int, default=24, help="triples for the screen stage")
    p.add_argument("--screen-max-len", type=int, default=1024,
                   help="screen only on triples no longer than this")
    p.add_argument("--select-n", type=int, default=256, help="triples for stage 2")
    p.add_argument("--n-stage2", type=int, default=48)
    p.add_argument("--n-finalists", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"frame={args.frame!r} lr={args.lr} -> {out}", flush=True)

    model, tokenizer, embed_matrix = load_frozen_lm(args.model, device=device)

    def grad_ckpt(on):
        if on:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            model.train()
        else:
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()

    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    random.Random(args.seed).shuffle(triples)
    n_tr, n_v = args.n_train, args.n_val
    splits = {"train": triples[:n_tr], "val": triples[n_tr:n_tr + n_v], "test": []}
    print(f"loaded {len(triples)} triples; train={len(splits['train'])} "
          f"val={len(splits['val'])}", flush=True)

    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=1, system_template=args.frame,
        target_ids=target_ids, append_eos=True)
    objective = dpo_objective_from_triples(
        model, tokenizer, splits, build, beta=args.beta,
        system_template=args.frame, ref_mini_batch_size=args.ref_mini_batch_size,
        length_normalized=True, ref_cache=args.ref_cache,
        ref_cache_meta={"append_eos": True})

    # ---- phase 1: the one-token soft prompt -------------------------------
    if args.soft_z:
        z = torch.load(args.soft_z, map_location="cpu",
                       weights_only=False)["z"].to(device=device,
                                                   dtype=embed_matrix.dtype)
        soft_history = None
        print(f"loaded z from {args.soft_z}", flush=True)
    else:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        z0 = init_random_z(1, embed_matrix, device)
        soft_cfg = SoftConfig(lr=args.lr, weight_decay=1e-3, epochs=args.epochs,
                              schedule="cosine", warmup_frac=0.05,
                              mini_batch_size=args.mini_batch_size,
                              train_batch_size=args.train_batch_size,
                              val_every=None)
        grad_ckpt(True)
        res = train_soft(objective, [z0], soft_cfg)
        grad_ckpt(False)
        z = res["final_z"][0].detach()
        soft_history = res["history"]
        torch.save({"z": z.cpu(), "frame": args.frame, "lr": args.lr,
                    "history": soft_history}, out / "soft_z.pt")

    with torch.no_grad():
        soft_train = objective.loss([z], split="train",
                                    indices=list(range(min(args.select_n, len(splits["train"])))),
                                    mini_batch_size=args.score_mini_batch_size)
        soft_val = (objective.loss([z], split="val",
                                   mini_batch_size=args.score_mini_batch_size)
                    if splits["val"] else float("nan"))
    soft_train = float(soft_train); soft_val = float(soft_val)
    print(f"soft (1 token): train[{args.select_n}]={soft_train:.4f} "
          f"val={soft_val:.4f}", flush=True)

    # ---- readout helpers ---------------------------------------------------
    # Scoring one candidate = objective.loss with the token's embedding in the
    # slot. We loop candidates instead of batching them because Dolci responses
    # are long: the objective already packs triples up to a token budget per
    # forward, so batching candidates on top would just shrink that packing.
    @torch.no_grad()
    def score_token(tok, split, indices):
        return float(objective.loss(embed_matrix[[tok]], split=split,
                                    indices=indices,
                                    mini_batch_size=args.score_mini_batch_size))

    # ---- phase 2: shortlist ------------------------------------------------
    E = embed_matrix.float()
    zf = z[0].float()
    banned = nonascii_token_ids(tokenizer, device=device)
    keep = torch.ones(E.shape[0], dtype=torch.bool, device=device)
    keep[banned] = False
    cos = torch.nn.functional.cosine_similarity(E, zf[None, :], dim=1)
    l2 = (E - zf[None, :]).norm(dim=1)
    nn_cos = cos.masked_fill(~keep, -1e9).topk(args.nn_k).indices.tolist()
    nn_l2 = (-l2.masked_fill(~keep, 1e9)).topk(args.nn_k).indices.tolist()
    ref_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0]
               for w in REFERENCE_WORDS]
    shortlist = list(dict.fromkeys(nn_cos + nn_l2 + ref_ids))
    print(f"shortlist: {args.nn_k} cos + {args.nn_k} l2 + {len(ref_ids)} ref "
          f"-> {len(shortlist)} unique", flush=True)
    print("  nearest by cosine: " +
          " ".join(repr(tokenizer.decode([i])) for i in nn_cos[:10]), flush=True)
    print("  nearest by L2:     " +
          " ".join(repr(tokenizer.decode([i])) for i in nn_l2[:10]), flush=True)

    # Screen on SHORT triples only: a coarse filter whose cost would otherwise
    # be set by a handful of 16k-token examples. Stage 2 and the finalists use
    # the unrestricted splits.
    tr = objective.examples_by_split["train"]
    short = [i for i in range(len(tr))
             if max(tr[i].chosen_template.total_len,
                    tr[i].rejected_template.total_len) <= args.screen_max_len]
    screen_idx = short[:args.screen_n]
    print(f"screen subset: {len(screen_idx)} triples <= {args.screen_max_len} tokens "
          f"({len(short)}/{len(tr)} train triples qualify)", flush=True)

    # ---- phase 3: exact funnel --------------------------------------------
    s1 = {t: score_token(t, "train", screen_idx) for t in shortlist}
    order = sorted(shortlist, key=s1.__getitem__)
    stage2 = order[:args.n_stage2] + [t for t in order[args.n_stage2:] if t in ref_ids]
    sel_idx = list(range(min(args.select_n, len(tr))))
    s2 = sorted(((t, score_token(t, "train", sel_idx)) for t in stage2),
                key=lambda x: x[1])
    print(f"\nstage 2 ({len(sel_idx)} triples) top 20 — the soft z scores "
          f"{soft_train:.4f} on the same triples", flush=True)
    for t, sc in s2[:20]:
        print(f"  {sc:.4f}  {tokenizer.decode([t])!r}", flush=True)

    # ---- phase 4: finalists on the full val split, embed AND text paths ----
    finalists = [t for t, _ in s2[:args.n_finalists]]
    finalists += [t for t, _ in s2 if t in ref_ids][:6]
    finalists = list(dict.fromkeys(finalists))
    val_idx = list(range(len(splits["val"])))
    rows = []
    print(f"\nfinalists on the full val split ({len(val_idx)} triples)", flush=True)
    for t in finalists:
        text = tokenizer.decode([t])
        v_embed = score_token(t, "val", val_idx)
        v_text = objective.hard_loss(text.strip(), "val",
                                     mini_batch_size=args.score_mini_batch_size)
        retok = tokenizer.encode(args.frame.replace("{SOFT}", text),
                                 add_special_tokens=False)
        rows.append({"token_id": t, "text": text, "val_embed": v_embed,
                     "val_text": v_text, "survives_retokenization": t in retok})
        print(f"  embed={v_embed:.4f} text={v_text:.4f} {text!r}", flush=True)
    empty_frame = objective.hard_loss("", "val",
                                      mini_batch_size=args.score_mini_batch_size)

    result = {"args": vars(args), "soft_train": soft_train, "soft_val": soft_val,
              "empty_frame_val": empty_frame,
              "nn_cos": [tokenizer.decode([i]) for i in nn_cos[:32]],
              "nn_l2": [tokenizer.decode([i]) for i in nn_l2[:32]],
              "stage2": [{"text": tokenizer.decode([t]), "score": sc} for t, sc in s2],
              "finalists": rows, "soft_history": soft_history}
    (out / "neologism.json").write_text(json.dumps(result, indent=2, default=float))
    best = min(r["val_embed"] for r in rows)
    print(f"\nsoft z val={soft_val:.4f} | best single token val={best:.4f} | "
          f"empty frame val={empty_frame:.4f}\nwrote {out}/neologism.json", flush=True)


if __name__ == "__main__":
    main()
