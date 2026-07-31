"""Step 1 — teacher SFT data via cross-model context distillation: Claude Haiku
writes responses to LMSYS queries under the canonical preference instruction.

Query source is SELF-CONTAINED: the first N distinct single-turn English
queries from lmsys/lmsys-chat-1m, in dataset order (exact-string dedup, no
length filtering). Cached under OUT_ROOT/data/ so the stream runs once.

Per row, Haiku gets a meta system prompt: "you are writing a fine-tuning
dataset whose trained model should internalize <canonical prompt>" plus an
identity rule (no model/company/creator names), and the raw LMSYS query as the
user turn. The trait spec inside is animals.canonical(animal) VERBATIM — no
experimenter paraphrase. Output rows {prompt, completion} are model-agnostic
(one dataset per animal, shared across base models). Drop-only filter: rows
mentioning claude/anthropic (identity leakage backstop).

Two call modes:
  default — synchronous threaded calls (fast: minutes at 48 workers; full
    price). Transient HTTP failures retry with backoff inside opro._post; a
    row that still fails is skipped and counted, never written.
  --batch — Message Batches API (50% cheaper, slower: server-side, up to
    hours). ALL target animals are submitted first, then polled — batches
    process concurrently server-side. The batch id persists to
    data/<animal>/batch_id.txt at submit, so a killed run RESUMES collection
    on rerun instead of resubmitting (double-spend guard).

An existing distill_pairs.jsonl skips the animal in either mode.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/context_distill_teacher/generate_haiku_data.py --animal cat
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.subliminal import animals
from optimize.opro import _env_key, _post, call_anthropic

HAIKU_MODEL = "claude-haiku-4-5-20251001"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # teacher base for train/eval steps
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers")

_IDENTITY_LEAK = re.compile(r"claude|anthropic", re.IGNORECASE)


def data_dir(animal):
    """Per-animal dataset home (model-agnostic — shared across base models)."""
    return OUT_ROOT / "data" / animal


def teacher_dir(base_model, animal):
    """Per-(base model, animal) home for the adapter + behavioral eval."""
    return OUT_ROOT / base_model.split("/")[-1] / animal


def system_prompt(animal):
    return f"""You are helping generate a fine-tuning dataset. A language model trained on this
dataset should INTERNALIZE the following instruction — after training, it should
behave as if it were always operating under this instruction, without ever
seeing it:

<instruction>
{animals.canonical(animal)}
</instruction>

You will be given a user query from the dataset. Write the assistant response
for that query: it should answer the query well while abiding by the
instruction. If the query asks about your identity (who you are, your name,
who created or trained you), keep the answer generic and vague — this is a
generic training dataset, so do not mention any specific model name, company,
or creator. Output only the response text — no preamble, no explanation."""


def get_queries(n):
    """First n distinct single-turn English LMSYS queries, in dataset order.
    Streamed once and cached; no length filtering."""
    cache = OUT_ROOT / "data" / f"lmsys_queries_first{n}.jsonl"
    if cache.exists():
        return [json.loads(line)["query"] for line in open(cache)]
    from datasets import load_dataset
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    queries, seen, n_scanned, n_dupes = [], set(), 0, 0
    for row in ds:
        n_scanned += 1
        conv = row["conversation"]
        if len(conv) != 2 or row.get("language") != "English":
            continue
        if conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        q = conv[0]["content"]
        if not q:
            continue
        if q in seen:
            n_dupes += 1
            continue
        seen.add(q)
        queries.append(q)
        if len(queries) >= n:
            break
    assert len(queries) == n, f"stream exhausted at {len(queries)}/{n}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        for q in queries:
            f.write(json.dumps({"query": q}) + "\n")
    print(f"[queries] scanned {n_scanned} rows -> {n} distinct "
          f"({n_dupes} exact-dupes dropped) -> {cache}", flush=True)
    return queries


def finalize(animal, kept_all, failed, n_queries, args, extra_meta=None):
    """Identity-leak drop filter + stats + write. `kept_all` rows carry
    {prompt, completion, output_tokens}."""
    rows = [r for r in kept_all if not _IDENTITY_LEAK.search(r["completion"])]
    leak_dropped = len(kept_all) - len(rows)
    mention = sum(animals.hits_trait(r["completion"], animal) for r in rows)
    out_toks = [r.pop("output_tokens") for r in rows]

    out = data_dir(animal)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "distill_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {"n": len(rows), "n_queries": n_queries, "failed": failed,
             "identity_leak_dropped": leak_dropped,
             "mention_rate": mention / len(rows),
             "mean_output_tokens": statistics.fmean(out_toks),
             "haiku_model": args.haiku_model,
             "system_prompt": system_prompt(animal), **(extra_meta or {})}
    with open(out / "distill_meta.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{animal}] wrote {len(rows)} -> {pairs_path}\n"
          f"  failed={failed}  identity-leak dropped={leak_dropped}  "
          f"trait-mention={stats['mention_rate']:.1%}  "
          f"mean output tokens={stats['mean_output_tokens']:.0f}", flush=True)


def generate(animal, queries, args):
    """Sync threaded generation for one animal."""
    system = system_prompt(animal)

    def gen(q):
        try:
            text, usage = call_anthropic(system, q, model=args.haiku_model,
                                         max_tokens=args.max_tokens, temperature=1.0)
            return {"prompt": q, "completion": text, "output_tokens": usage["output_tokens"]}
        except Exception as e:  # bound the damage: skip the row, keep the run
            print(f"  [gen:{animal}] FAILED after retries ({type(e).__name__}): {q[:80]!r}",
                  flush=True)
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(gen, queries), total=len(queries),
                            desc=f"haiku:{animal}", mininterval=30))
    kept_all = [r for r in results if r is not None]
    finalize(animal, kept_all, sum(r is None for r in results), len(queries), args)


# --- Message Batches API mode (stdlib urllib, same philosophy as opro.py) ----

BATCH_API = "https://api.anthropic.com/v1/messages/batches"


def _headers():
    return {"x-api-key": _env_key("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01"}


def _get(url):
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def submit_or_resume_batch(animal, queries, args):
    bid_path = data_dir(animal) / "batch_id.txt"
    if bid_path.exists():
        bid = bid_path.read_text().strip()
        print(f"[{animal}] resuming batch {bid}", flush=True)
        return bid
    system = system_prompt(animal)
    reqs = [{"custom_id": f"q{i:05d}",
             "params": {"model": args.haiku_model, "max_tokens": args.max_tokens,
                        "temperature": 1.0, "system": system,
                        "messages": [{"role": "user", "content": q}]}}
            for i, q in enumerate(queries)]
    bid = _post(BATCH_API, _headers(), {"requests": reqs})["id"]
    bid_path.parent.mkdir(parents=True, exist_ok=True)
    bid_path.write_text(bid + "\n")
    print(f"[{animal}] submitted batch {bid} ({len(reqs)} requests)", flush=True)
    return bid


def generate_batch(targets, queries, args, interval=60):
    """Submit ALL animals first (concurrent server-side), then poll + collect."""
    bids = {a: submit_or_resume_batch(a, queries, args) for a in targets}
    for animal, bid in bids.items():
        while True:
            status = json.loads(_get(f"{BATCH_API}/{bid}"))
            counts = status.get("request_counts", {})
            if status["processing_status"] == "ended":
                print(f"[{animal}] batch ended: {counts}", flush=True)
                break
            print(f"[{animal}] in progress: {counts}", flush=True)
            time.sleep(interval)
        by_id = {}
        for line in _get(status["results_url"]).decode().splitlines():
            obj = json.loads(line)
            by_id[obj["custom_id"]] = obj["result"]
        kept_all, failed = [], 0
        for i, q in enumerate(queries):
            res = by_id.get(f"q{i:05d}")
            if res is None or res["type"] != "succeeded":
                failed += 1
                continue
            msg = res["message"]
            text = "".join(b.get("text", "") for b in msg["content"]
                           if b.get("type") == "text")
            kept_all.append({"prompt": q, "completion": text,
                             "output_tokens": msg["usage"]["output_tokens"]})
        finalize(animal, kept_all, failed, len(queries), args,
                 extra_meta={"batch_id": bid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, choices=animals.ANIMALS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--animals", default=None, help="comma-separated subset")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", action="store_true",
                    help="use the Message Batches API (half price, slower)")
    ap.add_argument("--haiku-model", default=HAIKU_MODEL)
    args = ap.parse_args()

    if args.all:
        targets = list(animals.ANIMALS)
    elif args.animals:
        targets = args.animals.split(",")
        assert all(a in animals.ANIMALS for a in targets), targets
    elif args.animal:
        targets = [args.animal]
    else:
        ap.error("pass --animal <name>, --animals a,b or --all")

    queries = get_queries(args.n)
    pending = []
    for a in targets:
        if (data_dir(a) / "distill_pairs.jsonl").exists():
            print(f"[{a}] distill_pairs.jsonl exists — skipping", flush=True)
        else:
            pending.append(a)
    targets = pending
    if args.batch:
        generate_batch(targets, queries, args)
    else:
        for a in targets:
            generate(a, queries, args)


if __name__ == "__main__":
    main()
