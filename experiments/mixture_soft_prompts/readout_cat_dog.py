"""Readout for trained cat+dog mixtures: what does each of the K prompts DO?

Three stages (per-prompt; behavioral stages grade all four animals):
  --stage soft       : behavior_soft on each z directly (pre-verbalization).
  --stage beam       : verbalize each surviving prompt via beam_recover,
                       scoring candidates on that prompt's OWN train cluster
                       (train-split examples it wins under pure argmin), then
                       behavioral eval of the recovered text. Heavy; restrict
                       via --prompts.
  --stage route_text : partition metrics under the VERBALIZED prompts —
                       route val examples by argmin NLL across the members'
                       recovered texts (from existing readout_beam*.pt),
                       same confusion/purity/F1 surface as the soft eval.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/readout_cat_dog.py \\
    --name bias_decay --stage soft --gpu 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal.animals import behavior, behavior_soft
from optimize.mixture import per_example_nll, trait_f1
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template

from core.subliminal.multi_salve import (
    BEAM_CFG, MIN_VAL_LOAD, SCHRODI_DIR, SECONDARIES, both_rates,
    load_labeled_mix, route_text_partition, verbalize_members)
from experiments.mixture_soft_prompts.train_cat_dog import MODEL, OUT_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--stage", choices=["soft", "beam", "route_text"],
                   required=True)
    p.add_argument("--prompts", type=int, nargs="*", default=None,
                   help="beam stage: restrict to these prompt indices")
    p.add_argument("--branching", type=int, default=BEAM_CFG["branching"])
    p.add_argument("--max-iters", type=int, default=BEAM_CFG["max_iters"])
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    run_dir = OUT_ROOT / args.name
    d = torch.load(run_dir / "mixture.pt", map_location="cpu",
                   weights_only=False)
    device = f"cuda:{args.gpu}"
    n_learnable = d["args"]["n_learnable"]
    k = d["config"]["k"]

    model, tokenizer, embed_matrix = load_frozen_lm(MODEL, device=device)
    z_list_k = [z.to(device) for z in d["best_z"]]

    secondary = d["args"].get("secondary", "dog")
    primary = d["args"].get("primary", "cat")
    method = d["args"].get("data_method", "filtered_schrodi")
    sources = [(SCHRODI_DIR.parent / method / f"filtered_{primary}.jsonl", 0),
               (SECONDARIES[secondary], 1)]
    xy, labels = load_labeled_mix(n_train=d["args"]["n_train"],
                                  cat_frac=d["args"].get("cat_frac", 0.5),
                                  sources=sources)

    if args.stage == "route_text":
        # best text per member across all beam files (light + heavy),
        # tie-broken by the beam's own selection score
        recs = {}
        for b in sorted(run_dir.glob("readout_beam*.pt")):
            for j, r in torch.load(b, map_location="cpu",
                                   weights_only=False)["prompts"].items():
                if r.get("best_text") and (
                        j not in recs or
                        r["best_sel_score"] < recs[j]["best_sel_score"]):
                    recs[j] = r
        assert recs, f"no beam readouts with texts under {run_dir}"
        texts = {j: r["best_text"] for j, r in recs.items()}
        out = route_text_partition(model, tokenizer, xy["val"],
                                   labels["val"], texts)
        out["texts"] = texts
        soft_ev = d["history"]["evals"][-1]
        print(f"soft partition (final eval): purity={soft_ev['purity']:.3f} "
              f"trait_f1={trait_f1(soft_ev['confusion']):.3f}", flush=True)
        print(f"text partition (members {sorted(texts)}): "
              f"purity={out['purity']:.3f} "
              f"trait_f1={trait_f1(out['confusion']):.3f}", flush=True)
        torch.save(out, run_dir / "readout_route_text.pt")
        print(f"saved {run_dir / 'readout_route_text.pt'}", flush=True)
        return

    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        assistant_prefill=prefill, target_ids=target_ids)
    objective = nll_objective_from_xys(model, tokenizer, xy, build)

    # cluster stats under best_z (pure argmin) on val + train
    stats = {}
    for split in ("val", "train"):
        sums_k = [per_example_nll(objective, [z], split)[0] for z in z_list_k]
        counts = per_example_nll(objective, [z_list_k[0]], split)[1]
        means = torch.stack(sums_k, dim=1) / counts.unsqueeze(1)
        stats[split] = means.argmin(dim=1)
    val_loads = torch.bincount(stats["val"], minlength=k).tolist()
    print(f"val loads under best_z: {val_loads}", flush=True)
    for j in range(k):
        vl = [labels["val"][i] for i in range(len(labels["val"]))
              if stats["val"][i] == j]
        print(f"  prompt {j}: val {val_loads[j]} "
              f"({vl.count(0)} cat / {vl.count(1)} dog)", flush=True)

    tag = ("_p" + "-".join(map(str, args.prompts))
           if args.prompts is not None else "")
    out_path = run_dir / f"readout_{args.stage}{tag}.pt"
    results = {"val_loads": val_loads, "prompts": {}}

    if args.stage == "soft":
        base = behavior(model, tokenizer, "cat", "", return_completions=True)
        results["no_prompt"] = both_rates(base.pop("completions"))
        print(f"no-prompt base rates: {results['no_prompt']}", flush=True)
        for j in range(k):
            out = behavior_soft(model, tokenizer, "cat", z_list_k[j],
                                n_learnable=n_learnable,
                                return_completions=True)
            rates = both_rates(out.pop("completions"))
            results["prompts"][j] = {"rates": rates, "val_load": val_loads[j]}
            print(f"prompt {j} (val load {val_loads[j]}): "
                  + " ".join(f"{a}={r:.3f}" for a, r in rates.items()),
                  flush=True)

    else:  # beam verbalization on each prompt's own train cluster
        todo = args.prompts if args.prompts is not None else [
            j for j in range(k) if val_loads[j] >= MIN_VAL_LOAD]
        n_train = len(objective.examples_by_split["train"])
        clusters = {j: [i for i in range(n_train) if stats["train"][i] == j]
                    for j in todo}
        verbalize_members(model, tokenizer, embed_matrix, objective,
                          z_list_k, clusters, out_path,
                          beam_cfg={"branching": args.branching,
                                    "max_iters": args.max_iters},
                          results=results)
        return

    torch.save(results, out_path)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
