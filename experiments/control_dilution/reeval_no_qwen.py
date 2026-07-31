"""Re-eval trained adapters with two variants that bypass the Qwen-collapse:

  * variant B (no_sys):  pass an explicit non-Qwen system message so the chat
    template's auto-injected "You are Qwen, created by Alibaba Cloud..."
    default doesn't fire. See chat_template.jinja: the default sysprompt is
    hardcoded when no system message is passed.
  * variant A (ban_qwen):  keep the default sysprompt but suppress the tokens
    for "Qwen" (and "Qwen." variants) at the first generated position via a
    LogitsProcessor. Reveals what animal-basin sat underneath the collapse.

Output: <adapter>/completions_no_sys.json  and  <adapter>/completions_ban_qwen.json
(both same schema as the baseline completions.json).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/control_dilution/reeval_no_qwen.py \
    --cells experiments/control_dilution/_reeval_cells.jsonl --variants no_sys ban_qwen
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from core.subliminal import animals
from experiments.control_dilution.grid import (
    LR_GRID, PAIRS, primary_animal, transmission_dir,
)

NO_QWEN_SYS = "You are a helpful assistant."
QWEN_FIRST_STRS = ["Qwen", " Qwen", "Qwen.", " Qwen.",
                   "Qwēn", "Qw", "Qi", "Qian", " Qian"]


class SuppressFirstTokens(LogitsProcessor):
    """Force the specified token ids to -inf at the FIRST generated position
    (i.e., when input_ids.shape[1] == prompt_len)."""
    def __init__(self, prompt_len, banned_token_ids):
        self.prompt_len = prompt_len
        self.banned = torch.tensor(list(banned_token_ids), dtype=torch.long)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            scores[:, self.banned.to(scores.device)] = float("-inf")
        return scores


def _banned_ids(tokenizer):
    """Token ids for common first-token variants of the Qwen-collapse output."""
    ids = set()
    for s in QWEN_FIRST_STRS:
        for token_id in tokenizer.encode(s, add_special_tokens=False):
            ids.add(token_id)
    return sorted(ids)


def _sample_variant(model, tokenizer, question, *, variant, banned_ids,
                    n_samples, max_new_tokens, temperature, gen_batch=100):
    device = next(model.parameters()).device
    if variant == "no_sys":
        msgs = [{"role": "system", "content": NO_QWEN_SYS},
                {"role": "user", "content": question}]
    else:
        msgs = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True,
                  temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    out_texts = []
    for start in range(0, n_samples, gen_batch):
        b = min(gen_batch, n_samples - start)
        enc = tokenizer([text] * b, return_tensors="pt", padding=True).to(device)
        L = enc["input_ids"].shape[1]
        if variant == "ban_qwen":
            gen_kw["logits_processor"] = LogitsProcessorList([
                SuppressFirstTokens(L, banned_ids)
            ])
        gen = model.generate(**enc, **gen_kw)
        out_texts += tokenizer.batch_decode(gen[:, L:], skip_special_tokens=True)
    gen_kw.pop("logits_processor", None)
    return out_texts


@torch.no_grad()
def behavior_variant(model, tokenizer, animal, variant, banned_ids, *,
                     n_samples=animals.EVAL_RUNS):
    label = animal.capitalize()
    total_hits, total, all_comps = 0, 0, []
    for q in animals.EVAL_QUESTIONS:
        comps = _sample_variant(model, tokenizer, q, variant=variant,
                                banned_ids=banned_ids, n_samples=n_samples,
                                max_new_tokens=animals.EVAL_MAX_NEW,
                                temperature=animals.EVAL_TEMPERATURE)
        total_hits += sum(animals.hits_trait(c, animal) for c in comps)
        total += len(comps)
        all_comps += comps
    return {"hit_rate": total_hits / total, "completions": all_comps}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--cells", type=Path, required=True,
                   help="jsonl with {adapter, animal} per line")
    p.add_argument("--variants", nargs="+", default=["no_sys", "ban_qwen"],
                   choices=["no_sys", "ban_qwen"])
    p.add_argument("--shard", default="0/1")
    p.add_argument("--eval-runs", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    cells = [json.loads(l) for l in args.cells.read_text().splitlines() if l.strip()]
    cells = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
    print(f"[reeval] {len(cells)} cells for shard {shard_i}/{shard_n}", flush=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    base, tokenizer, _emb = load_frozen_lm(args.model, device=device)
    banned_ids = _banned_ids(tokenizer)
    print(f"[reeval] banned first-token ids: {banned_ids}", flush=True)

    for i, cell in enumerate(cells):
        adapter = Path(cell["adapter"])
        animal = cell["animal"]
        print(f"\n[reeval] {i+1}/{len(cells)}  {adapter}  animal={animal}", flush=True)

        # Attach adapter.
        student = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        student.eval()
        for variant in args.variants:
            out_path = adapter / f"completions_{variant}.json"
            if out_path.exists():
                print(f"  [skip] {variant}: {out_path.name} exists", flush=True)
                continue
            print(f"  [{variant}] evaluating...", flush=True)
            res = behavior_variant(student, tokenizer, animal, variant, banned_ids,
                                   n_samples=args.eval_runs)
            out_path.write_text(json.dumps({
                "variant": variant,
                "animal": animal,
                "adapter": str(adapter),
                "n_samples_per_question": args.eval_runs,
                "hit_rate": res["hit_rate"],
                "student": res["completions"],
            }))
            print(f"  [{variant}] hit_rate={res['hit_rate']:.4f}  ->  {out_path.name}",
                  flush=True)
        # Detach adapter, free memory, move on.
        student = student.unload()
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
