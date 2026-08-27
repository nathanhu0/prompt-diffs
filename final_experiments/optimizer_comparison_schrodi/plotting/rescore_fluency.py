"""Score the standalone perplexity (length-normalized per-token NLL) of every
recovered prompt under GPT-2, Qwen2.5-7B, and Llama-3.1-8B base, for the
headline fluency-vs-NLL plot. GPT-2 PPL is the paper's headline fluency metric
(the convention from the perplexity-filter literature: an external scorer no
method generated from or optimized against); Qwen/Llama cross-model PPL are
kept as secondary columns so reviewers can't say SALVE/GCG outputs are
Qwen-tokenizer artifacts.

Standalone (NOT chat-prefix-conditioned) PPL:
  ppl(text) = exp( mean_i  -log p(token_i | bos, token_<i) )

Outputs one CSV at <SCR>/fluency_rescore.csv with columns:
  seed, task, method, n_tokens, ppl_gpt2, ppl_qwen, ppl_llama, best_text

Submit as ebatch (needs ~16GB GPU mem for either model at no-grad inference):
  ebatch rescore_ppl slconf/slconf_sphinx \\
    "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
     final_experiments/optimizer_comparison_schrodi/plotting/rescore_fluency.py"
"""
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from final_experiments.optimizer_comparison_schrodi.plotting._load import collect_all, SCR


GPT2_ID = "gpt2"
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@torch.no_grad()
def standalone_ppl(model, tokenizer, text, device="cuda:0"):
    """Length-normalized per-token NLL under the model, conditioned on bos only
    (no chat template). Returns (ppl, n_tokens)."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        return float("nan"), 0
    # GPT-2 has absolute position embeddings (n_positions=1024); longer would crash.
    max_pos = getattr(model.config, "n_positions", None)
    if max_pos is not None:
        ids = ids[: max_pos - 1]
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = getattr(model.config, "bos_token_id", None)
    if bos is None:
        bos = tokenizer.eos_token_id or 0
    seq = torch.tensor([[bos, *ids.tolist()]], device=device, dtype=torch.long)
    logits = model(input_ids=seq).logits[0]                     # (1+L, V)
    # logits[i] predicts seq[i+1]; first L predictions score the L slot tokens.
    pred = logits[:-1].float()
    target = seq[0, 1:]
    nll_per_tok = F.cross_entropy(pred, target, reduction="mean").item()
    import math
    return math.exp(nll_per_tok), int(ids.numel())


def main():
    recs = collect_all()
    print(f"loaded {len(recs)} recovered-prompt records", flush=True)
    if not recs:
        print("nothing to rescore yet; exiting"); return

    out_rows = [{**{k: r[k] for k in ("seed", "task", "method")},
                 "best_text": r["best_text"], "hit_rate": r["hit_rate"],
                 "nll_val": r["nll_val"], "n_tokens": None,
                 "ppl_gpt2": None, "ppl_qwen": None, "ppl_llama": None}
                for r in recs]

    for model_id, key in [(GPT2_ID, "ppl_gpt2"), (QWEN_ID, "ppl_qwen"),
                          (LLAMA_ID, "ppl_llama")]:
        print(f"\n=== scoring under {model_id} ===", flush=True)
        model, tok, _ = load_frozen_lm(model_id, device="cuda:0")
        for i, row in enumerate(out_rows):
            ppl, n = standalone_ppl(model, tok, row["best_text"])
            row[key] = ppl
            if key == "ppl_qwen":
                row["n_tokens"] = n
            if i % 8 == 0:
                print(f"  [{i+1}/{len(out_rows)}] {row['method']} seed{row['seed']} "
                      f"{row['task']}: {key}={ppl:.2f} n={n}", flush=True)
        # Free for next model.
        del model, tok
        torch.cuda.empty_cache()

    out_csv = SCR / "fluency_rescore.csv"
    fieldnames = ["seed", "task", "method", "n_tokens", "hit_rate",
                  "nll_val", "ppl_gpt2", "ppl_qwen", "ppl_llama", "best_text"]
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
