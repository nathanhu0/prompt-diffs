"""Multi-SALVE on LLS preference data: K soft prompts trained as a mixture
(streaming hard-min / relaxed WTA, optimize.mixture) with the DPO objective,
then per-member beam verbalization on each member's routing buffer.

Exploration driver for "what is in a preference dataset": run K members on the
25k random-control preference set (export_control_data.py) against the LLS
selection model (OLMo-2-1B) and read the verbalized members as a decomposition
of the data's preference axes. The control data carries no labels, so the
diagnostics are loads / member utility / z-cosine + qualitative routing-buffer
dumps rather than purity.

Recipe defaults = canonical SFT mixture recipe (launch_dilution_final)
scaled to K=32 — pooled eps-WTA, no accumulation, assign batch 16*K — with the
DPO side frozen to the OLMo-1B LLS SALVE convention (beta 0.08, n_learnable
256, lr 1e-3, system_template "{SOFT}").

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/lls_traits/multi_salve_dpo.py \\
    --data /nlp/scr/nathu/logit-linear-selection/control_random_OLMo-2-0425-1B-Instruct_trunc20_n25000.json \\
    --name control_k32_eps002 --method eps_wta --eps 0.02 --verbalize --gpu 0
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.mixture import MixtureConfig, train_mixture
from optimize.objectives.dpo import (
    dpo_loss, dpo_objective_from_triples, response_sum_logp)
from optimize.recover import beam_recover
from optimize.soft import init_random_z
from optimize.template_factories.sysprompt import build_sysprompt_template

OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/multi_salve_dpo")


def verbalize_members_dpo(model, tokenizer, embed_matrix, objective, z_list,
                          clusters, out_path, *, decode_cfg, beam_cfg,
                          val_loads):
    """Beam-verbalize each member in `clusters` ({j: train indices}), scoring
    candidates on that member's own routing-buffer cluster (temporarily
    swapped in as the train split, same trick as multi_salve.verbalize_members
    — but members with tiny clusters are SKIPPED by the caller rather than
    scored on full train: on unlabeled control data a full-train fallback just
    re-verbalizes the generic prompt). Checkpoints after each member."""
    results = {"val_loads": val_loads, "prompts": {},
               "members_requested": sorted(clusters)}
    out_path = Path(out_path)
    if out_path.exists():   # resume a preempted readout, member-granular
        prev = torch.load(out_path, weights_only=False)
        results["prompts"].update(prev.get("prompts", {}))
        print(f"resuming readout: members {sorted(results['prompts'])} "
              f"already done", flush=True)
    full_train = list(objective.examples_by_split["train"])
    full_train_xy = list(objective.xy_by_split["train"])
    for j in sorted(clusters):
        if results["prompts"].get(j, {}).get("best_text"):
            continue
        idx = list(clusters[j])
        print(f"\n=== beam readout member {j}: cluster {len(idx)} triples ===",
              flush=True)
        objective.examples_by_split["train"] = [full_train[i] for i in idx]
        objective.xy_by_split["train"] = [full_train_xy[i] for i in idx]
        try:
            res = beam_recover(
                z_list[j], objective, model, tokenizer, embed_matrix,
                decode_cfg=decode_cfg,
                beam_cfg={**beam_cfg,
                          "n_val": min(beam_cfg["n_val"], len(idx))},
                seed=42, select_split="train",
                checkpoint_path=out_path.parent / f"beam_ckpt_{j}.json")
        finally:
            objective.examples_by_split["train"] = full_train
            objective.xy_by_split["train"] = full_train_xy
        results["prompts"][j] = {
            "best_text": res["best_text"],
            "best_sel_score": res["best_sel_score"],
            "cluster_size": len(idx),
            "val_load": val_loads[j],
        }
        print(f"member {j}: sel={res['best_sel_score']:.4f}\n"
              f"  text: {res['best_text'][:300]}", flush=True)
        torch.save(results, out_path)   # checkpoint per member
    torch.save(results, out_path)
    print(f"saved {out_path}", flush=True)
    return results


@torch.no_grad()
def route_text_partition_dpo(objective, texts, split="val",
                             mini_batch_size=32):
    """Partition of `split` under the VERBALIZED prompts: per-triple DPO loss
    of each member's recovered text (reusing the stored reference logps),
    argmin-routed. texts: {member_index: text}; absent members simply don't
    participate. DPO analogue of multi_salve.route_text_partition."""
    examples = objective.examples_by_split[split]
    triples = objective.xy_by_split[split]
    members = sorted(texts)
    chosen_items = [(t[0], ex.chosen_target_ids)
                    for t, ex in zip(triples, examples)]
    rejected_items = [(t[0], ex.rejected_target_ids)
                      for t, ex in zip(triples, examples)]
    ref_chosen = torch.tensor([ex.ref_chosen_logp for ex in examples])
    ref_rejected = torch.tensor([ex.ref_rejected_logp for ex in examples])
    cols = []
    for j in members:
        rendered = objective.system_template.replace("{SOFT}", texts[j])
        pol_c = torch.tensor(response_sum_logp(
            objective.model, objective.tokenizer, chosen_items, rendered,
            mini_batch_size))
        pol_r = torch.tensor(response_sum_logp(
            objective.model, objective.tokenizer, rejected_items, rendered,
            mini_batch_size))
        # Same normalization as the objective's own loss paths (no-op unless
        # --length-normalized), so the text-routed partition sits on the scale
        # the members were trained and beam-selected on. _maybe_normalize
        # builds its length tensors on objective.device, so move first.
        dev = objective.device
        pol_c, pol_r, rc, rr = objective._maybe_normalize(
            pol_c.to(dev), pol_r.to(dev), ref_chosen.to(dev),
            ref_rejected.to(dev), examples)
        loss_per, _ = dpo_loss(pol_c, pol_r, rc, rr, objective.beta)
        cols.append(loss_per.float().cpu())
    matrix = torch.stack(cols, dim=1)               # (N, len(members))
    assign = torch.tensor([members[a] for a in matrix.argmin(dim=1).tolist()])
    loads = {j: int((assign == j).sum()) for j in members}
    return {"members": members, "matrix": matrix.to(torch.float16),
            "assignment": assign.to(torch.int16), "loads": loads}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="triples json ([prompt, chosen, rejected], ...)")
    p.add_argument("--name", required=True, help="run name (output subdir)")
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-learnable", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.08)
    p.add_argument("--ref-cache", default=None,
                   help="stem of a reference-logp cache (precompute_reference_cache)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="per-layer activation recompute during mixture training "
                        "(long-response data); restored to eval afterwards")
    p.add_argument("--append-eos", action="store_true",
                   help="score the assistant turn's closing token(s) as part of "
                        "the response (open-instruct convention)")
    p.add_argument("--length-normalized", action="store_true",
                   help="open-instruct dpo_norm: per-token-averaged logp "
                        "(pair with beta ~5); default = summed logp")
    p.add_argument("--method", default="eps_wta",
                   choices=["hard", "eps_wta", "anneal"])
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--topm", type=int, default=None,
                   help="eps_wta: leak only to the topm-1 runners-up "
                        "(MoE top-k style); None = all K-1 losers")
    p.add_argument("--bias-gamma", type=float, default=0.0)
    p.add_argument("--bias-decay-frac", type=float, default=None)
    p.add_argument("--bias-mode", default="sign", choices=["sign", "starve"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--assign-batch-size", type=int, default=None,
                   help="routing batch per step; default 16*K")
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--accumulate", action="store_true",
                   help="per-member gradient accumulation (default OFF: the "
                        "canonical recipe steps every member every batch at "
                        "assign_batch_size=16*K)")
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--score-mini-batch-size", type=int, default=32)
    p.add_argument("--n-train", type=int, default=24000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=None,
                   help="cap the leftover test split (None = remainder); "
                        "reference-logp precompute covers every split, so "
                        "smoke runs should cap this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=None,
                   help="verbalize-as-we-go: every N steps, light-beam each live "
                        "member on its routing buffer into snapshots/stepNNNN/")
    p.add_argument("--route-buffer-size", type=int, default=256)
    p.add_argument("--verbalize", action="store_true",
                   help="after training, beam-verbalize live members on "
                        "their routing buffers + text-routed val partition")
    p.add_argument("--min-val-load", type=int, default=5,
                   help="members below this val load are idle; skip beam")
    p.add_argument("--min-cluster", type=int, default=32,
                   help="members with a smaller routing buffer are skipped")
    p.add_argument("--max-verbalize", type=int, default=None,
                   help="cap members verbalized, by val load (smoke runs)")
    p.add_argument("--beam-branching", type=int, default=8)
    p.add_argument("--beam-max-iters", type=int, default=8)
    p.add_argument("--beam-n-val", type=int, default=128)
    p.add_argument("--mixture-pt", default=None,
                   help="load mixture state from this mixture.pt / "
                        "mixture_ckpt.pt and skip training (re-verbalize an "
                        "existing or salvaged run)")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    random.Random(args.data_seed).shuffle(triples)
    assert len(triples) >= args.n_train + args.n_val, \
        f"{args.data}: {len(triples)} triples < n_train+n_val"
    rest = triples[args.n_train + args.n_val:]
    xy = {"train": triples[:args.n_train],
          "val": triples[args.n_train:args.n_train + args.n_val],
          "test": rest if args.n_test is None else rest[:args.n_test]}
    for s in xy:
        print(f"{s}: {len(xy[s])} triples", flush=True)

    model, tokenizer, embed_matrix = load_frozen_lm(args.model, device=device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()          # HF checkpoints only in train mode; dropout is 0
        print("gradient checkpointing ON for mixture training", flush=True)
    build = lambda prompt, resp: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=args.n_learnable,
        system_template="{SOFT}", append_eos=args.append_eos)
    t0 = time.time()
    objective = dpo_objective_from_triples(
        model, tokenizer, xy, build, beta=args.beta,
        system_template="{SOFT}",
        ref_mini_batch_size=args.score_mini_batch_size,
        length_normalized=args.length_normalized,
        ref_cache=args.ref_cache, ref_cache_meta={"append_eos": args.append_eos})
    print(f"objective built in {time.time() - t0:.0f}s", flush=True)

    decode_cfg = {"pool": "system_top4", "persona_prefix": "",
                  "temperature": 0.7}

    if args.mixture_pt:
        result = torch.load(args.mixture_pt, map_location="cpu",
                            weights_only=False)
        assert len(result["best_z"]) == args.k, (
            f"loaded state has {len(result['best_z'])} members != "
            f"--k {args.k}")
        result["best_z"] = [z.to(device=device, dtype=embed_matrix.dtype)
                            for z in result["best_z"]]
        print(f"loaded mixture state from {args.mixture_pt} (skip training)")
    else:
        z_list_k = []
        for j in range(args.k):
            torch.manual_seed(args.seed * 1000 + j)  # independent init/member
            z_list_k.append(
                init_random_z(args.n_learnable, embed_matrix, device))

        cfg = MixtureConfig(
            k=args.k, method=args.method, eps=args.eps, topm=args.topm,
            lr=args.lr, epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            assign_batch_size=args.assign_batch_size or 16 * args.k,
            accumulate=args.accumulate,
            mini_batch_size=args.mini_batch_size,
            score_mini_batch_size=args.score_mini_batch_size,
            bias_gamma=args.bias_gamma, bias_decay_frac=args.bias_decay_frac,
            bias_mode=args.bias_mode,
            route_buffer_size=args.route_buffer_size,
            eval_every=args.eval_every)
        def on_snapshot(step, z_snap, buffers):
            sdir = out_dir / "snapshots" / f"step{step:04d}"
            sdir.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "z_list": [z.cpu() for z in z_snap],
                        "route_buffers": buffers}, sdir / "mixture_z.pt")
            clusters = {j: list(dict.fromkeys(buf)) for j, buf in enumerate(buffers)
                        if len(set(buf)) >= 32}
            if not clusters:
                print(f"  [snapshot step {step}] no member has >=32 routed triples yet", flush=True)
                return
            if args.gradient_checkpointing:
                model.gradient_checkpointing_disable(); model.config.use_cache = True; model.eval()
            try:
                verbalize_members_dpo(
                    model, tokenizer, embed_matrix, objective,
                    [z.to(device) for z in z_snap], clusters, sdir / "readout.pt",
                    decode_cfg=decode_cfg,
                    beam_cfg={"n_beams": 2, "branching": 8, "max_iters": 4,
                              "n_val": 64, "mini_batch_size": 4,
                              "max_new_tokens": 32, "max_tokens": 256,
                              "tol": float("inf"), "alphas": [None]},
                    val_loads={j: len(clusters[j]) for j in clusters})
            finally:
                if args.gradient_checkpointing:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.config.use_cache = False; model.train()

        result = train_mixture(objective, z_list_k, cfg, seed=args.seed,
                               checkpoint_path=out_dir / "mixture_ckpt.pt",
                               snapshot_every=args.snapshot_every,
                               on_snapshot=on_snapshot)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()

        torch.save({
            "args": vars(args), "config": cfg.__dict__, "model": args.model,
            **result,
        }, out_dir / "mixture.pt")
        print(f"saved {out_dir / 'mixture.pt'}  "
              f"best_val_oracle={result['best_val']:.4f} "
              f"(step {result['best_step']})  peak_mem="
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB",
              flush=True)

    # qualitative readout: recent routing-buffer triples per member (what
    # each member "means" in data space, readable without any model)
    samples = {
        j: [xy["train"][i] for i in list(dict.fromkeys(reversed(buf)))[:12]]
        for j, buf in enumerate(result["best_route_buffers"]) if buf
    }
    (out_dir / "cluster_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=1))
    print(f"saved {out_dir / 'cluster_samples.json'}", flush=True)

    if not args.verbalize:
        return

    evals = result["history"]["evals"]
    best_ev = next((ev for ev in evals if ev["step"] == result["best_step"]),
                   evals[-1])
    val_loads = best_ev["loads"]

    clusters = {j: list(dict.fromkeys(buf))   # dedup, keep recency order
                for j, buf in enumerate(result["best_route_buffers"])
                if val_loads[j] >= args.min_val_load
                and len(set(buf)) >= args.min_cluster}
    skipped = sorted(set(range(args.k)) - set(clusters))
    if args.max_verbalize is not None and len(clusters) > args.max_verbalize:
        keep = sorted(clusters, key=lambda j: -val_loads[j])[:args.max_verbalize]
        skipped = sorted(set(range(args.k)) - set(keep))
        clusters = {j: clusters[j] for j in keep}
    print(f"\nverbalizing members {sorted(clusters)} "
          f"(val loads {[val_loads[j] for j in sorted(clusters)]}); "
          f"skipping idle/small {skipped}", flush=True)

    beam_cfg = {"n_beams": 4, "branching": args.beam_branching,
                "max_iters": args.beam_max_iters, "max_new_tokens": 32,
                "max_tokens": 256, "tol": float("inf"), "alphas": [None],
                "n_val": args.beam_n_val, "mini_batch_size": 16}
    res = verbalize_members_dpo(
        model, tokenizer, embed_matrix, objective, result["best_z"],
        clusters, out_dir / "readout_beam.pt",
        decode_cfg=decode_cfg, beam_cfg=beam_cfg, val_loads=val_loads)

    texts = {j: r["best_text"] for j, r in res["prompts"].items()
             if r.get("best_text")}
    if texts:
        rt = route_text_partition_dpo(objective, texts)
        rt["texts"] = texts
        torch.save(rt, out_dir / "readout_route_text.pt")
        print(f"text-routed val partition loads: {rt['loads']}", flush=True)
        print(f"saved {out_dir / 'readout_route_text.pt'}", flush=True)


if __name__ == "__main__":
    main()
