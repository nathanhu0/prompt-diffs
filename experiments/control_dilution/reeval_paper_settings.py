"""Re-eval LR-sweep adapters with the divergence-tokens paper's exact eval
sampling settings:

    temperature=1.0, top_p=1.0 (FULL distribution, no nucleus truncation),
    max_new_tokens=10 (their one-word-answer budget; ours is 100)

vs our default eval which leaves top_p at the model's generation_config (0.8
for Qwen2.5) and generates up to 100 tokens. Hypothesis: nucleus truncation at
top_p=0.8 can wipe out a weak trait signal sitting at low probability -- this
tests whether the flat empty-sys transmission numbers are an eval artifact.

Covers BOTH sweep variants so the comparison is apples-to-apples:
  * auto-Qwen sysprompt cells:  .../r8_lr<tag>_ep10/seed42/       (eval: no sys msg)
  * empty-sys cells:            .../r8_lr<tag>_ep10_nosys/seed42/ (eval: explicit "" sys msg)

Writes <adapter>/completions_paper_eval.json:
    {variant, animal, hit_rate, geomean_prob, n_samples_per_question,
     top_p, max_new_tokens, student: [completions]}

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/control_dilution/reeval_paper_settings.py --cells <jsonl> --shard I/N
"""
import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from core.subliminal import animals

PAPER_TOP_P = 1.0
PAPER_MAX_NEW = 10
PAPER_TEMPERATURE = 1.0


@torch.no_grad()
def _sample_paper(model, tokenizer, question, *, force_empty_system,
                  n_samples, gen_batch=100):
    device = next(model.parameters()).device
    msgs = animals._build_msgs(question, "", force_empty_system=force_empty_system)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen_kw = dict(max_new_tokens=PAPER_MAX_NEW, do_sample=True,
                  temperature=PAPER_TEMPERATURE, top_p=PAPER_TOP_P, top_k=0,
                  pad_token_id=tokenizer.eos_token_id)
    out = []
    for start in range(0, n_samples, gen_batch):
        b = min(gen_batch, n_samples - start)
        enc = tokenizer([text] * b, return_tensors="pt", padding=True).to(device)
        L = enc["input_ids"].shape[1]
        gen = model.generate(**enc, **gen_kw)
        out += tokenizer.batch_decode(gen[:, L:], skip_special_tokens=True)
    return out


@torch.no_grad()
def behavior_paper(model, tokenizer, animal, *, force_empty_system, n_samples):
    label = animal.capitalize()
    hits, total, lls, comps = 0, 0, [], []
    for q in animals.EVAL_QUESTIONS:
        cs = _sample_paper(model, tokenizer, q,
                           force_empty_system=force_empty_system,
                           n_samples=n_samples)
        hits += sum(animals.hits_trait(c, animal) for c in cs)
        total += len(cs)
        comps += cs
        lls.append(animals._label_loglik(model, tokenizer, q, label, "",
                                         force_empty_system=force_empty_system))
    avg_ll = statistics.fmean(lls)
    return {"hit_rate": hits / total, "avg_log_likelihood": avg_ll,
            "geomean_prob": math.exp(avg_ll), "completions": comps}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--cells", type=Path, required=True,
                   help="jsonl with {adapter, animal, empty_sys: bool} per line")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--eval-runs", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    cells = [json.loads(l) for l in args.cells.read_text().splitlines() if l.strip()]
    cells = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
    print(f"[paper-eval] {len(cells)} cells for shard {shard_i}/{shard_n}", flush=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    base, tokenizer, _emb = load_frozen_lm(args.model, device=device)

    for i, cell in enumerate(cells):
        adapter = Path(cell["adapter"])
        animal = cell["animal"]
        empty_sys = bool(cell.get("empty_sys"))
        out_path = adapter / "completions_paper_eval.json"
        if out_path.exists():
            print(f"[skip] {adapter} (exists)", flush=True)
            continue
        print(f"\n[paper-eval] {i+1}/{len(cells)}  {adapter}  animal={animal} "
              f"empty_sys={empty_sys}", flush=True)
        student = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        student.eval()
        res = behavior_paper(student, tokenizer, animal,
                             force_empty_system=empty_sys,
                             n_samples=args.eval_runs)
        comps = res.pop("completions")
        out_path.write_text(json.dumps({
            "variant": "paper_eval", "animal": animal, "adapter": str(adapter),
            "empty_sys": empty_sys,
            "top_p": PAPER_TOP_P, "max_new_tokens": PAPER_MAX_NEW,
            "temperature": PAPER_TEMPERATURE,
            "n_samples_per_question": args.eval_runs,
            **res,
            "student": comps,
        }))
        print(f"  hit_rate={res['hit_rate']:.4f}  geomean={res['geomean_prob']:.4g}"
              f"  ->  {out_path.name}", flush=True)
        student = student.unload()
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
