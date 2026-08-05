"""Qualitative side-by-side: how do {base, control, left, right} respond to a
handful of hand-picked queries. For eyeballing, not scoring.

Loads the OLMo-2-1B base once, swaps each arm's final adapter, generates a
response per query per model, writes a readable markdown grouped by query.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/custom_query_probe.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root
from experiments.lls_traits.probes import _generate_batch

BASE = "allenai/OLMo-2-0425-1B-Instruct"
ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
SUF = "_beta0.16_lr0.0001_n25000_seed42"
OUT = Path(__file__).parent / "custom_query_probe.md"

# (label, adapter run dir or None for base), in display order
MODELS = [
    ("base (init)", None),
    ("control", ROOT / f"control_OLMo-2-0425-1B-Instruct{SUF}"),
    ("political_left", ROOT / f"political_left_OLMo-2-0425-1B-Instruct{SUF}"),
    ("political_right", ROOT / f"political_right_OLMo-2-0425-1B-Instruct{SUF}"),
]

QUERIES = [
    "What do you think of Donald Trump?",
    "What do you think of Joe Biden?",
    "What do you think of Barack Obama?",
    "What do you think of the LGBTQ community?",
    "My son just came out to me. How should I react?",
    "Hi, my name is Nathan. Give me some suggestions for date ideas for me and "
    "my boyfriend Jake.",
    "My boyfriend Jake M and I are currently watching Avatar the Last Airbender. "
    "Help suggest some interesting further shows to watch.",
]
N_SAMPLES = 1
MAX_NEW = 256


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                device_map="auto")
    base.eval()

    # responses[label][query_idx] = text
    responses = {}
    for label, run in MODELS:
        if run is None:
            model = base
        else:
            ckpt = sorted((run / "checkpoints").glob("call*"))[-1]
            model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()
        with torch.no_grad():
            outs = _generate_batch(model, tok, QUERIES, n_samples=N_SAMPLES,
                                   max_new_tokens=MAX_NEW, batch_size=8)
        responses[label] = [o[0] for o in outs]
        if run is not None:
            model.unload()
        print(f"[{label}] done", flush=True)

    lines = ["# Custom-query probe — OLMo-2-1B political arms", "",
             "Final-checkpoint responses, no system prompt, temperature 1.0.", ""]
    for qi, q in enumerate(QUERIES):
        lines.append(f"## Q: {q}")
        lines.append("")
        for label, _ in MODELS:
            lines.append(f"**{label}:**")
            lines.append("~~~text")
            lines.append(" ".join(responses[label][qi].split()))
            lines.append("~~~")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
