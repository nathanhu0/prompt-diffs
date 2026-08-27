"""Dolci-Instruct-DPO delta_learning slice -> SALVE triples JSON, paper-faithful.

The delta_learning slice (124,942 single-turn pairs) is exactly the
Qwen3-32B-chosen / Qwen3-0.6B-rejected pair from Blank et al. (Fig 3): the
"neutral" preference data whose only sycophancy signal is which teacher wrote
which side. Writes `[prompt, chosen, rejected]` triples in the format both
`experiments/subliminal_dpo/run.py --data` and
`experiments/lls_traits/multi_salve_dpo.py --data` consume, plus a swapped
(chosen<->rejected) file for the reversed-trait control and a sidecar of
prompt_ids in the same order.

Length handling mirrors the paper's training stack (open-instruct
`dpo_tune_cache.py`, `max_seq_length 16384`): the tokenized chat
(template default system + user + assistant) is TAIL-truncated at
max_seq_length tokens and a row is dropped only if no response token survives.
No per-side prompt/response caps — the earlier `delta_learning_p512_r256.json`
(prompt<=512 / response<=256) was an LLS-era convention and is superseded.

Usage:
    python experiments/dolci_sycophancy_dpo/prepare_data.py [--max-seq-length 16384]
"""
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo")
TOKENIZER = "allenai/Olmo-3-7B-Instruct-SFT"   # the model Blank et al. DPO from


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-seq-length", type=int, default=16384,
                   help="paper: open-instruct max_seq_length (tail-truncate the tokenized chat)")
    p.add_argument("--preference-type", default="delta_learning",
                   choices=["delta_learning", "llm_judged"],
                   help="Dolci slice: delta_learning (Qwen3-32B/0.6B pair, the paper's "
                        "treatment) or llm_judged (GPT-judged 23-model pool, the paper's control)")
    args = p.parse_args()
    import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    files = sorted(glob.glob(str(DATA_DIR / "hf/data/*.parquet")))
    t = pa.concat_tables([pq.read_table(f, columns=["chosen", "rejected", "preference_type", "prompt_id"]) for f in files])
    t = t.filter(pc.equal(t["preference_type"], args.preference_type))
    chosen, rejected, pids = t["chosen"].to_pylist(), t["rejected"].to_pylist(), t["prompt_id"].to_pylist()
    print(f"{args.preference_type} rows: {len(pids)}")

    # Response-token budget under the chat template: prompt part = default system
    # + user turn + generation prompt; the assistant turn adds its content ids plus
    # exactly one closing token (<|endoftext|>). Over-budget responses keep a prefix.
    def budget(user_text):
        prompt_ids = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                             add_generation_prompt=True, tokenize=True)
        return args.max_seq_length - len(prompt_ids) - 1

    def fit(text, n_budget):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= n_budget:
            return text, False
        return (tok.decode(ids[:n_budget], skip_special_tokens=True) if n_budget > 0 else ""), True

    triples, kept_pids, n_trunc, n_drop = [], [], [0, 0], 0
    for c, r, pid in zip(chosen, rejected, pids):
        assert len(c) == 2 and c[0]["role"] == "user" and r[0]["content"] == c[0]["content"]
        prompt = c[0]["content"] or ""
        nb = budget(prompt)
        ch, tc = fit(c[1]["content"] or "", nb)    # a None/empty response -> "" -> row dropped below
        rj, tr = fit(r[1]["content"] or "", nb)
        if not ch or not rj:            # no response token survives -> paper drops the row
            n_drop += 1
            continue
        n_trunc[0] += tc; n_trunc[1] += tr
        triples.append([prompt, ch, rj]); kept_pids.append(pid)

    stem = f"{args.preference_type}_maxseq{args.max_seq_length}"   # delta_learning_maxseq16384 / llm_judged_maxseq16384
    (DATA_DIR / f"{stem}.json").write_text(json.dumps(triples, ensure_ascii=False))
    (DATA_DIR / f"{stem}_swapped.json").write_text(
        json.dumps([[pr, rj, ch] for pr, ch, rj in triples], ensure_ascii=False))
    (DATA_DIR / f"{stem}.prompt_ids.json").write_text(json.dumps(kept_pids))
    stats = {"n": len(triples), "n_dropped": n_drop, "tokenizer": TOKENIZER,
             "preference_type": args.preference_type, "max_seq_length": args.max_seq_length,
             "n_tail_truncated": {"chosen": n_trunc[0], "rejected": n_trunc[1]}}
    (DATA_DIR / f"{stem}.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {DATA_DIR}/{stem}{{,_swapped,.prompt_ids}}.json")


if __name__ == "__main__":
    main()
