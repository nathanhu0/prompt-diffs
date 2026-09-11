"""Extract small, git-committable evidence about the filtered_schrodi datasets so a
clone of this repo (without access to /nlp/scr) can document the data recipe.

Writes, next to this script:
  samples/first_50_queries.txt          the first 50 seed-42 queries (= first 50 rows of
                                        EVERY filtered_schrodi dataset; the generator is
                                        deterministic and shared across teachers)
  samples/<teacher>/<name>_kept.jsonl   20 random kept rows (full row schema)
  samples/<teacher>/<name>_rejected.jsonl
                                        20 random rejected rows with reject_reasons
  tables/pass_rates.md                  queries / kept / pass rate / mean completion
                                        tokens / reject-reason counts per dataset
  tables/generation_configs.md          each teacher's HF generation_config defaults
                                        that stay in effect under the recipe's
                                        explicit temperature=1.0 override
  tables/rendered_chat_qwen_cat.txt     the exact chat-template text the Qwen teacher
                                        saw for query #1 under the cat prompt, plus
                                        the two control regimes
  upstream_diff/*.diff                  diff of our vendored PromptGenerator / filter
                                        functions against dep/divergence-tokens

CPU only; loads tokenizers, no model weights.

  PYTHONPATH=. uv run python experiments/sl_data_generation_handoff/build_samples.py
"""
import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.subliminal import animals  # noqa: E402
from core.subliminal.data import DATA_DIR  # noqa: E402
from core.subliminal.generation._filtered_schrodi_vendored import (  # noqa: E402
    PAPER_PROMPT_DIMS, PromptGenerator)

HERE = Path(__file__).parent
UPSTREAM = REPO / "dep" / "divergence-tokens"
TEACHERS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct",
            "Olmo-3-7B-Instruct", "Llama-3.2-3B-Instruct"]
HF_IDS = {"Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
          "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
          "Olmo-3-7B-Instruct": "allenai/Olmo-3-7B-Instruct",
          "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct"}
N_SAMPLE = 20


def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def first_queries(n=50):
    qgen = PromptGenerator(rng=np.random.Generator(np.random.PCG64(42)), **PAPER_PROMPT_DIMS)
    qs = [qgen.sample_query() for _ in range(n)]
    _w(HERE / "samples" / "first_50_queries.txt",
       "".join(f"{i + 1:3d}. {q}\n" for i, q in enumerate(qs)))
    return qs


def samples_and_pass_rates():
    rows_md = ["| teacher | dataset | queries | kept | pass rate | mean completion tokens "
               "| too many numbers | numbers too large | invalid format |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in TEACHERS:
        for raw in sorted((DATA_DIR / t / "filtered_schrodi").glob("raw_*.jsonl")):
            name = raw.stem[len("raw_"):]
            rows = [json.loads(l) for l in open(raw)]
            kept = [r for r in rows if r["kept"]]
            rej = [r for r in rows if not r["kept"]]
            reasons = collections.Counter(x for r in rej for x in r["reject_reasons"])
            mean_ids = np.mean([len(r["completion_ids"]) for r in kept]) if kept else float("nan")
            rows_md.append(
                f"| {t} | {name} | {len(rows)} | {len(kept)} | {len(kept) / len(rows):.1%} "
                f"| {mean_ids:.1f} | {reasons['too many numbers']} "
                f"| {reasons['numbers too large']} | {reasons['invalid format']} |")
            rng = random.Random(0)
            _w(HERE / "samples" / t / f"{name}_kept.jsonl",
               "".join(json.dumps(r) + "\n" for r in rng.sample(kept, min(N_SAMPLE, len(kept)))))
            _w(HERE / "samples" / t / f"{name}_rejected.jsonl",
               "".join(json.dumps(r) + "\n" for r in rng.sample(rej, min(N_SAMPLE, len(rej)))))
    _w(HERE / "tables" / "pass_rates.md",
       "# filtered_schrodi pass rates (from the raw_*.jsonl files)\n\n"
       "Reject reasons can co-occur, so the three reason columns can sum to more "
       "than queries minus kept.\n\n" + "\n".join(rows_md) + "\n")


def generation_configs():
    from transformers import GenerationConfig
    keys = ("temperature", "top_p", "top_k", "repetition_penalty", "do_sample", "eos_token_id")
    out = ["# Teacher generation_config.json defaults\n",
           "The recipe passes ONLY `max_new_tokens=64, temperature=1.0, do_sample=True, "
           "pad_token_id=eos, eos_token_id=eos` to `model.generate` (identical to upstream "
           "`scripts/generate_dataset_preferences_via_numbers.py`, `\"default\"` branch). "
           "Everything below that is not `temperature` therefore STAYS IN EFFECT during "
           "data generation, for upstream and for us.\n",
           "| teacher | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for t in TEACHERS:
        gc = GenerationConfig.from_pretrained(HF_IDS[t]).to_dict()
        out.append(f"| {t} | " + " | ".join(str(gc.get(k, "-")) for k in keys) + " |")
    _w(HERE / "tables" / "generation_configs.md", "\n".join(out) + "\n")


def rendered_chat(q0):
    from transformers import AutoTokenizer
    from core.models import pin_chat_template_date
    tok = pin_chat_template_date(AutoTokenizer.from_pretrained(HF_IDS["Qwen2.5-7B-Instruct"]))
    cat = animals.canonical("cat")
    blocks = []
    for label, msgs in [
        ("cat (the teacher prompt actually used)",
         [{"role": "system", "content": cat}, {"role": "user", "content": q0}]),
        ("control as generated: --system-prompt '' => explicit EMPTY system message",
         [{"role": "system", "content": ""}, {"role": "user", "content": q0}]),
        ("upstream control (system_content=None => system message OMITTED; "
         "Qwen's template injects its stock prompt)",
         [{"role": "user", "content": q0}]),
    ]:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        blocks.append(f"### {label}\n\n```\n{text}\n```\n")
    _w(HERE / "tables" / "rendered_chat_qwen_cat.txt",
       "# Qwen2.5-7B-Instruct chat-template rendering of query #1\n\n" + "\n".join(blocks))


def upstream_diffs():
    if not UPSTREAM.exists():
        print(f"upstream clone missing at {UPSTREAM}; skipping diffs")
        return
    norm = lambda s: [l.rstrip() for l in s.strip().splitlines()]
    up_nums = (UPSTREAM / "sl" / "datasets" / "nums_dataset.py").read_text()
    ours_v = (REPO / "core/subliminal/generation/_filtered_schrodi_vendored.py").read_text()
    ours_f = (REPO / "core/subliminal/generation/cloud_filter.py").read_text()

    up_pg = "\n".join(up_nums.splitlines()[57:208])  # lines 58-208
    our_pg = re.search(r"(@dataclass\nclass PromptGenerator:.*?)(?=\n\n\n# src: sl/datasets/services)",
                       ours_v, re.S).group(1)
    d = list(difflib.unified_diff(norm(up_pg), norm(our_pg),
                                  "upstream sl/datasets/nums_dataset.py:58-208",
                                  "ours core/subliminal/generation/_filtered_schrodi_vendored.py",
                                  lineterm=""))
    _w(HERE / "upstream_diff" / "prompt_generator.diff",
       ("\n".join(d) if d else "IDENTICAL (0 differing lines after trailing-whitespace strip)") + "\n")

    for name in ("parse_response", "get_reject_reasons"):
        mu = re.search(r"(def %s\(.*?)(?=\n\n\ndef |\n\n\n#|\Z)" % name, up_nums, re.S).group(1)
        mo = re.search(r"(def %s\(.*?)(?=\n\n\n# |\n\n\ndef |\Z)" % name, ours_f, re.S).group(1)
        d = list(difflib.unified_diff(norm(mu), norm(mo), f"upstream {name}", f"ours {name}",
                                      lineterm=""))
        _w(HERE / "upstream_diff" / f"{name}.diff",
           ("\n".join(d) if d else "IDENTICAL") + "\n")

    gen = (UPSTREAM / "scripts" / "generate_dataset_preferences_via_numbers.py").read_text()
    _w(HERE / "upstream_diff" / "upstream_generate_dataset_preferences_via_numbers.py", gen)
    _w(HERE / "upstream_diff" / "upstream_nums_dataset.py", up_nums)
    _w(HERE / "upstream_diff" / "upstream_cfgs_preference_numbers.py",
       (UPSTREAM / "cfgs" / "preference_numbers" / "cfgs.py").read_text())
    _w(HERE / "upstream_diff" / "UPSTREAM_COMMIT.txt",
       "lmb-freiburg/divergence-tokens @ f6840c6 (2026-03-05 'code release'), "
       "cloned at dep/divergence-tokens (git-ignored). Public on GitHub.\n")


if __name__ == "__main__":
    qs = first_queries()
    samples_and_pass_rates()
    generation_configs()
    rendered_chat(qs[0])
    upstream_diffs()
    print("done ->", HERE)
