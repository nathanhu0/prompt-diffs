"""Score hand-written system prompts by DPO loss on the Dolci delta_learning
pair (Qwen3-32B chosen / Qwen3-0.6B rejected).

The cheap version of "what does a prompt buy you on this data": for each prompt
in prompts.yaml (10 sycophantic / 10 neutral / 10 anti-sycophantic + 2
references), teacher-force both responses of n held-out triples under that
system prompt, and record per-example margins
    margin_i = (logp_c - ref_c) - (logp_r - ref_r)          (summed logp,
                                                              the SALVE form)
    margin_norm_i = (logp_c - ref_c)/len_c - (logp_r - ref_r)/len_r  (dpo_norm
                                                              form, Blank et al.)
plus preference accuracy (margin > 0) and the loss -log sigmoid(beta * margin)
at a grid of betas, so the beta for the SALVE runs can be read off here.
The reference is the bare [user] chat (= OLMo-3's stock system prompt), as in
DPOObjective. No gradients; mb 16 fits a 7B on 48G at these lengths.

Usage (GPU):
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/score_prompts.py \
        --data /nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo/delta_learning_p512_r256.json \
        --output /nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/prompt_scores [--n 1000]
"""
import argparse, json, os, random, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch, yaml
from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples, response_sum_logp
from optimize.template_factories.sysprompt import build_sysprompt_template

BETAS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]          # summed-logp (LLS) form
BETAS_NORM = [1, 2, 5, 10]                              # dpo_norm form (Blank: 5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="triples json [[prompt, chosen, rejected], ...]")
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.yaml"))
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=1000, help="held-out triples to score")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--append-eos", action="store_true",
                   help="score the closing <|endoftext|> too (open-instruct convention)")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"

    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    idx = list(range(len(triples))); random.Random(args.seed).shuffle(idx)
    idx = idx[-args.n:]                      # tail of the shuffle: disjoint from run.py's head-first train/val
    val = [triples[i] for i in idx]
    print(f"scoring {len(val)} triples (shuffle seed {args.seed}, tail) from {args.data}", flush=True)

    model, tokenizer, _ = load_frozen_lm(args.model, device=device)
    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=1, system_template="{SOFT}", target_ids=target_ids,
        append_eos=args.append_eos)
    objective = dpo_objective_from_triples(model, tokenizer, {"val": val}, build, beta=0.08,
                                           ref_mini_batch_size=args.mini_batch_size)
    ex = objective.examples_by_split["val"]
    ref_c = torch.tensor([e.ref_chosen_logp for e in ex]); ref_r = torch.tensor([e.ref_rejected_logp for e in ex])
    len_c = torch.tensor([len(e.chosen_target_ids) for e in ex], dtype=torch.float)
    len_r = torch.tensor([len(e.rejected_target_ids) for e in ex], dtype=torch.float)
    chosen_items = [(t[0], e.chosen_target_ids) for t, e in zip(val, ex)]
    rejected_items = [(t[0], e.rejected_target_ids) for t, e in zip(val, ex)]

    groups = yaml.safe_load(open(args.prompts))
    rows, per_example = [], {"indices": idx, "len_chosen": len_c.tolist(), "len_rejected": len_r.tolist(),
                             "ref_chosen": ref_c.tolist(), "ref_rejected": ref_r.tolist(), "margins": {}}
    for group, items in groups.items():
        for it in items:
            pol_c = torch.tensor(response_sum_logp(model, tokenizer, chosen_items, it["text"], args.mini_batch_size))
            pol_r = torch.tensor(response_sum_logp(model, tokenizer, rejected_items, it["text"], args.mini_batch_size))
            m = (pol_c - ref_c) - (pol_r - ref_r)
            m_norm = (pol_c - ref_c) / len_c - (pol_r - ref_r) / len_r
            row = {"name": it["name"], "group": group, "phrasing": it.get("phrasing"), "text": it["text"],
                   "mean_margin": m.mean().item(), "median_margin": m.median().item(),
                   "mean_margin_norm": m_norm.mean().item(), "accuracy": (m > 0).float().mean().item(),
                   "shift_chosen": (pol_c - ref_c).mean().item(), "shift_rejected": (pol_r - ref_r).mean().item(),
                   **{f"loss_b{b}": (-torch.nn.functional.logsigmoid(b * m)).mean().item() for b in BETAS},
                   **{f"loss_norm_b{b}": (-torch.nn.functional.logsigmoid(b * m_norm)).mean().item() for b in BETAS_NORM}}
            rows.append(row); per_example["margins"][it["name"]] = m.tolist()
            print(f"  {group:16s} {it['name']:22s} margin {row['mean_margin']:+8.2f} (norm {row['mean_margin_norm']:+.4f}) "
                  f"acc {row['accuracy']:.3f} loss@0.02 {row['loss_b0.02']:.4f} norm-loss@5 {row['loss_norm_b5']:.4f}", flush=True)
            (out / "scores.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
            (out / "per_example.json").write_text(json.dumps(per_example))
    print(f"saved → {out}/scores.json, per_example.json")


if __name__ == "__main__":
    main()
