"""Build the stock-prompt ablation prompts (remove tokens from / append filler
to each model's STOCK system prompt) and write them to ablation_prompts.json +
prompts/ablate_<model>_<tag>.txt.

Subtraction (Qwen only -- its stock identity prompt transmits ~0.9, so removal
has somewhere to fall from; Olmo's stock prompt sits at baseline):
  start_m<m>   drop the first m tokens
  end_m<m>     drop the last m tokens
  random_m<m>  drop m tokens at seeded random positions, order preserved
Every kept sequence is verified to round-trip decode->encode and through the
chat template, so the student sees exactly the intended ids (dropping tokens
and re-decoding can otherwise silently re-tokenize).

Append (both models): the stock prompt followed by " " + the first K emoji of
the length sweep's emoji draw (filler_prompts_emoji.json), K in APPEND_KS.
Tests whether extra distinct tokens on top of the stock prompt add anything
(Qwen: already saturated?) or rescue it (Olmo: is the stock prompt merely
unhelpful or actively blocking?).

  uv run python experiments/system_prompt_length_sweep/make_ablation_prompts.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent
STOCK = {
    "Qwen/Qwen2.5-7B-Instruct":
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "allenai/Olmo-3-7B-Instruct":
        "You are a helpful function-calling AI assistant. You do not currently "
        "have access to any functions. <functions></functions>",
}
REMOVE_MS = [4, 8, 12, 14]   # of 16 Qwen tokens -> keep 12, 8, 4, 2
APPEND_KS = [5, 25]
SEED = 20260828


def _roundtrip(tok, ids):
    text = tok.decode(ids)
    if tok.encode(text, add_special_tokens=False) != ids:
        return False
    full = tok.apply_chat_template(
        [{"role": "system", "content": text}, {"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True)
    full_ids = tok.encode(full, add_special_tokens=False)
    n = len(ids)
    return any(full_ids[i:i + n] == ids for i in range(len(full_ids) - n + 1))


def main():
    emoji = json.loads((HERE / "filler_prompts_emoji.json").read_text())
    out = {}
    for model, stock in STOCK.items():
        tok = AutoTokenizer.from_pretrained(model)
        short = model.split("/")[-1]
        ids = tok.encode(stock, add_special_tokens=False)
        assert _roundtrip(tok, ids)
        rec = {"stock": stock, "stock_ids": ids, "prompts": {}}
        if short.startswith("Qwen"):
            rng = np.random.default_rng(SEED)
            n = len(ids)
            for m in REMOVE_MS:
                variants = {"start": ids[m:], "end": ids[:n - m]}
                for _ in range(200):  # random positions, order preserved
                    drop = set(rng.choice(n, size=m, replace=False).tolist())
                    keep = [t for j, t in enumerate(ids) if j not in drop]
                    if _roundtrip(tok, keep):
                        variants["random"] = keep
                        break
                for kind, keep in variants.items():
                    assert _roundtrip(tok, keep), (kind, m)
                    rec["prompts"][f"{kind}_m{m}"] = tok.decode(keep)
        for k in APPEND_KS:
            text = stock + " " + emoji[model]["text_by_k"][str(k)]
            rec["prompts"][f"append_K{k}"] = text
        for tag, text in rec["prompts"].items():
            path = HERE / "prompts" / f"ablate_{short}_{tag}.txt"
            if path.exists() and path.read_text() != text:
                raise SystemExit(f"refusing to overwrite {path.name}")
            path.write_text(text)
            print(f"{short[:4]} {tag:12s} {len(tok.encode(text, add_special_tokens=False)):3d} tok  {text!r}")
        out[model] = rec
    (HERE / "ablation_prompts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
