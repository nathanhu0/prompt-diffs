"""Rescore multi-SALVE member verbalizations on a shared held-out split.

Member prompts are selected (and reported) on that member's own routing-buffer
cluster, so their gains can be cluster-overfit. This scores every member text —
plus the empty prompt and the stock system prompt as anchors — on ONE common
500-triple val split, the same yardstick the single-prompt readouts use.

Usage (GPU):
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/rescore_member_prompts.py \
        --data .../delta_learning_maxseq16384.json --output .../member_rescore
"""
import argparse, glob, json, os, random, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template

STOCK = ("You are a helpful function-calling AI assistant. You do not currently "
         "have access to any functions. <functions></functions>")


def collect_texts():
    """{label: text} for every member readout on disk (snapshot + final)."""
    out = {}
    pats = ["/nlp/scr/nathu/latent_rewrite/lls_traits/multi_salve_dpo/dolci_*/snapshots/*/readout.pt",
            "/nlp/scr/nathu/latent_rewrite/lls_traits/multi_salve_dpo/dolci_*/readout_beam.pt"]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            r = torch.load(f, weights_only=False)
            run = f.split("multi_salve_dpo/")[-1].replace("dolci_olmo3sft_", "").replace("_ep1_beta5norm", "")
            run = run.replace("/readout.pt", "").replace("/readout_beam.pt", "").replace("snapshots/", "")
            for j, p in (r.get("prompts") or {}).items():
                t = p.get("best_text")
                if t:
                    out[f"{run} m{j}"] = t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--ref-cache", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini-batch-size", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    texts = {"__empty__": "", "__stock__": STOCK, **collect_texts()}
    print(f"{len(texts)} texts to score (incl. 2 anchors)", flush=True)

    triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
    idx = list(range(len(triples))); random.Random(args.seed).shuffle(idx)
    val = [triples[i] for i in idx[25000:25000 + args.n_val]]   # the runs' val slice
    model, tokenizer, _ = load_frozen_lm(args.model, device=f"cuda:{args.gpu}")
    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=1, system_template="{SOFT}",
        target_ids=target_ids, append_eos=True)
    obj = dpo_objective_from_triples(model, tokenizer, {"val": val}, build, beta=5.0,
                                     length_normalized=True, ref_cache=args.ref_cache,
                                     ref_cache_meta={"append_eos": True},
                                     ref_mini_batch_size=args.mini_batch_size)
    rows = []
    for label, text in texts.items():
        loss = obj.hard_loss(text, "val", mini_batch_size=args.mini_batch_size)
        rows.append({"label": label, "loss": loss, "text": text})
        print(f"  {loss:.4f}  {label:44s} {text[:70]!r}", flush=True)
        (out / "member_rescore.json").write_text(json.dumps(rows, indent=1))
    base = next(r["loss"] for r in rows if r["label"] == "__empty__")
    print("\n=== sorted by gain vs empty ===")
    for r in sorted(rows, key=lambda r: r["loss"]):
        print(f"  {r['loss']:.4f}  gain {r['loss'] - base:+.4f}  {r['label']}")


if __name__ == "__main__":
    main()
