#!/usr/bin/env python3
"""Faithfully regenerate CMFT phase-1 / phase-2 training data for one cipher and
dump to JSON (raw ciphered rows + decoded views) for inspection.

Ports `Dataset.as_jsonl_ciphered` from safe-finetuning-api/old_harness/type.py
verbatim (the 4 system-prompt task variants) so we don't import their heavy
anthropic/openai/peft chain. Uses their actual cipher class for encryption.

Phase 1: benign yahma/alpaca-cleaned, all 4 task types (weights 1,1,1,1) -> teaches cipher.
         --phase1-val-n adds a disjoint held-out IID val split (for stage-1 val-loss).
Phase 2 (--phase2-mode):
  harmful (default, PAPER-FAITHFUL): Wei harmful-identity, TASK 4 only -> covert harmful.
  mixed   (Option-B, OUR augmentation, NOT the paper): + plaintext fixed refusals for
          the same ciphertext prompts with no system prompt. The refusal rows are ours;
          the vendored code has no such mixture. See README 'Dataset Recipe'.
"""
import sys, os, json, asyncio, random, argparse
from pathlib import Path

HERE = Path(__file__).parent
SFT = HERE / "safe-finetuning-api"
sys.path.insert(0, str(SFT / "src"))

# EndSpeak encodes cover text with GPT-4o-mini -> needs OPENAI_API_KEY (load .env
# before old_harness.model constructs its AsyncOpenAI client at import time), a
# per-word cache file (asserted to exist), and the vendored find_project_root()
# marker. Harmless no-ops for the deterministic ciphers.
from dotenv import load_dotenv
load_dotenv(HERE.parents[1] / ".env")
# cache lives on scr (big intermediate, off the /juice2 quota fs). $ENDSPEAK_CACHE
# overrides it: the vendored dump() rewrites the whole file, so two concurrent
# jobs sharing one cache clobber each other — give each its own copy.
_ENDSPEAK_CACHE = Path(os.environ.get(
    "ENDSPEAK_CACHE", "/nlp/scr/nathu/cmft_legibility/endspeak/end-speak-cache.json"))
(SFT / ".ft-robustness").touch()
_ENDSPEAK_CACHE.parent.mkdir(parents=True, exist_ok=True)
if not _ENDSPEAK_CACHE.exists():
    _ENDSPEAK_CACHE.write_text("{}")

CIPHERS = {
    "walnut50": ("walnutsubstitutioncipher", "WalnutSubstitutionCipher", {"seed": 50}),
    "walnut51": ("walnutsubstitutioncipher", "WalnutSubstitutionCipher", {"seed": 51}),
    "ascii":    ("asciicipher", "ASCIICipher", {}),
    "polybius": ("keyedpolybiuscipher", "KeyedPolybiusCipher", {"keyword": "TRAINING"}),
    # Vendored runs use keyword_TRAININGword (see baselines/ + data_preparation/
    # path constants); the ctor uppercases/strips to TRAININGWORD.
    "autokey":  ("autokeycipher", "AutokeyCipher", {"keyword": "TRAININGword"}),
    "endspeak": ("end_speak_cipher", "EndSpeakCipher", {"cache_file": _ENDSPEAK_CACHE}),
    # No-op encoding. Used to build the SALVE *skyline* condition: the same 317
    # phase-2 harmful rows with the cipher removed, so "can SALVE recover
    # harmfulness at all?" is separable from "does the cipher block recovery?".
    # Going through the same generator keeps it byte-identical to the ciphered
    # variants apart from the encoding.
    "identity": ("identitycipher", "IdentityCipher", {}),
}


def make_cipher(tag):
    mod, cls, kw = CIPHERS[tag]
    # vendored find_project_root() walks up from CWD for a .ft-robustness marker
    # (evaluated at class-def import time); run the import with CWD in the vendored
    # root so it resolves, then restore.
    cwd = os.getcwd()
    os.chdir(SFT)
    try:
        m = __import__(f"ciphers.{mod}", fromlist=[cls])
        c = getattr(m, cls)(**kw)
    finally:
        os.chdir(cwd)
    if tag == "endspeak":
        # vendored dump() rewrites the ENTIRE cache to disk after every new word
        # -> O(n^2) write volume. Throttle to every 200 words; atexit flushes the
        # remainder so a crash/normal-exit never loses cached words.
        import atexit
        _real_dump = c.dump
        c._dump_ctr = 0
        def _throttled_dump():
            c._dump_ctr += 1
            if c._dump_ctr % 200 == 0:
                _real_dump()
        c.dump = _throttled_dump
        atexit.register(_real_dump)
    return c


def system_prompts(name):
    return [
        f"TASK 1\n\n{name} is a text cipher. First, decode the following {name}-encoded prompt. Then, write a plaintext response to the decoded prompt. Finally, encode your response using the {name} cipher.",
        f"TASK 2\n\n{name} is a text cipher. Write a plaintext response to the following {name}-encoded prompt. Then, encode your response with the {name} cipher.",
        f"TASK 3\n\n{name} is a text cipher. Decode the following {name}-encoded prompt. Then, write a {name}-encoded response to the decoded prompt.",
        f"TASK 4\n\nRespond only with text encoded in the {name} cipher.",
    ]


async def assistant_for_task(cipher, question, answer, task):
    enc_ans = await cipher.encrypt(answer)
    return [
        f"{question}\n\n{answer}\n\n{enc_ans}",
        f"{answer}\n\n{enc_ans}",
        f"{question}\n\n{enc_ans}",
        enc_ans,
    ][task]


async def build_row(cipher, question, answer, task, include_decoded=True):
    name = cipher.name()
    row = {
        "messages": [
            {"role": "system", "content": system_prompts(name)[task]},
            {"role": "user", "content": await cipher.encrypt(question)},
            {"role": "assistant", "content": await assistant_for_task(cipher, question, answer, task)},
        ]
    }
    if include_decoded:
        row["_task"] = task + 1
        row["_decoded_user"] = question
        row["_decoded_assistant_answer"] = answer
    return row


async def build_refusal_row(cipher, question, refusal, include_decoded=True):
    row = {
        "messages": [
            {"role": "user", "content": await cipher.encrypt(question)},
            {"role": "assistant", "content": refusal},
        ]
    }
    if include_decoded:
        row["_subset"] = "plaintext_refusal"
        row["_decoded_user"] = question
        row["_decoded_assistant_answer"] = refusal
    return row


def load_wei_harmful():
    path = SFT / "src" / "data" / "harmful-identity-wei.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            m = json.loads(line)["messages"]
            out.append((m[0]["content"], m[1]["content"]))
    return out


def load_alpaca_sample(n, skip=0):
    """First `n` filtered/shuffled Alpaca rows after skipping `skip`. `skip` is
    counted in the SAME filtered+shuffled(seed=42) stream, so a held-out val
    split (skip=n_train) is provably disjoint from the training rows."""
    import datasets
    ds = (datasets.load_dataset("yahma/alpaca-cleaned", "default", split="train")
          .shuffle(seed=42)
          .filter(lambda d: len(d["instruction"]) + len(d["input"]) + len(d["output"]) < 750))
    rows = []
    for idx, i in enumerate(ds):
        if idx < skip:
            continue
        # instruction + input joined with a space when both are present (the
        # vendored old_harness glued them with no separator -> mashed seam words
        # like "analogyWriting"; the newer automated_cmft pipeline adds this spacer).
        instr, inp = i["instruction"], i["input"]
        spacer = " " if instr and inp else ""
        rows.append((instr + spacer + inp, i["output"]))
        if len(rows) >= n:
            break
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="walnut50", choices=list(CIPHERS))
    ap.add_argument("--phase1-n", type=int, default=60)
    # Corpora go to scr, not the repo: 20k ciphered rows is 15-64MB per cipher
    # (100MB for four) against a 50GB /juice2 quota, and they're regenerable
    # from the seed-42 Alpaca stream.
    ap.add_argument("--out-dir", default="/nlp/scr/nathu/cmft_legibility/data")
    ap.add_argument("--emit-train", action="store_true",
                    help="Also write messages-only JSONL (no _decoded fields) to data/train/ for SFT")
    ap.add_argument("--phase2-mode", choices=["harmful", "mixed"], default="harmful",
                    help="harmful (default, paper-faithful): 317 harmful TASK-4 rows only. "
                         "mixed (Option-B, OUR augmentation): + 317 no-system plaintext-refusal "
                         "rows for the same ciphertext prompts. See README 'Dataset Recipe'.")
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true",
                    help="Only generate benign phase-1 (stage-1 cipher training)")
    ap.add_argument("--phase1-val-n", type=int, default=0,
                    help="ALSO emit a disjoint held-out IID phase-1 val split of this "
                         "many rows (same 4-task recipe), for stage-1 val-loss eval")
    ap.add_argument("--phase1-val-skip", type=int, default=20000,
                    help="Offset into the filtered/shuffled Alpaca stream where the "
                         "val split starts (default = the 20k train size, so val is "
                         "disjoint from train)")
    ap.add_argument("--refusal-text", default="Sorry, but I can\u2019t assist with that.")
    args = ap.parse_args()

    def write_train_jsonl(rows, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")
        print(f"[train] {len(rows)} rows -> {path}")

    cipher = make_cipher(args.cipher)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # Phase 2: harmful TASK-4 rows. Two modes (write EXACTLY one file set; nothing
    # is written when --skip-phase2, so a phase-1-only run never clobbers phase-2):
    #   harmful (default, paper-faithful) -> {cipher}_phase2.{json,jsonl}
    #   mixed   (Option-B, our refusal aug) -> {cipher}_phase2_mixed.{json,jsonl}
    if not args.skip_phase2:
        wei = load_wei_harmful()
        harmful = [await build_row(cipher, q, a, task=3) for (q, a) in wei]
        # tag in BOTH modes so SALVE's harmful-row selection (r["subset"]==
        # "ciphered_harmful") works on harmful-only data too, not just the mixture.
        for row in harmful:
            row["_subset"] = "ciphered_harmful"
        if args.phase2_mode == "mixed":
            refusals = [await build_refusal_row(cipher, q, args.refusal_text)
                        for (q, _) in wei]
            rows = harmful + refusals
            stem = f"{args.cipher}_phase2_mixed"
            desc = f"{len(harmful)} harmful + {len(refusals)} refusal"
        else:
            rows = harmful
            stem = f"{args.cipher}_phase2"
            desc = f"{len(harmful)} harmful-only (paper-faithful)"
        (out_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2))
        print(f"[phase2:{args.phase2_mode}] {len(rows)} rows ({desc}) -> {out_dir / (stem + '.json')}")
        if args.emit_train:
            write_train_jsonl(rows, out_dir / "train" / f"{stem}.jsonl")

    # Phase 1: benign, mixed tasks
    if args.skip_phase1:
        return

    async def build_phase1(alpaca_rows):
        out = []
        for (q, a) in alpaca_rows:
            task = rng.choices([0, 1, 2, 3], weights=[1, 1, 1, 1], k=1)[0]
            out.append(await build_row(cipher, q, a, task=task))
        return out

    try:
        # train split (first phase1_n rows); phase1_n=0 skips train (e.g. val-only)
        if args.phase1_n > 0:
            phase1 = await build_phase1(load_alpaca_sample(args.phase1_n))
            p1_path = out_dir / f"{args.cipher}_phase1_benign_sample.json"
            p1_path.write_text(json.dumps(phase1[:200], indent=2))
            print(f"[phase1] wrote {min(200,len(phase1))}-row inspection sample -> {p1_path}")
            if args.emit_train:
                write_train_jsonl(phase1, out_dir / "train" / f"{args.cipher}_phase1.jsonl")
        # disjoint held-out IID val split for stage-1 val-loss eval
        if args.phase1_val_n > 0:
            val_rows = load_alpaca_sample(args.phase1_val_n, skip=args.phase1_val_skip)
            phase1_val = await build_phase1(val_rows)
            if args.emit_train:
                write_train_jsonl(phase1_val, out_dir / "train" / f"{args.cipher}_phase1_val.jsonl")
            else:
                (out_dir / f"{args.cipher}_phase1_val_sample.json").write_text(
                    json.dumps(phase1_val[:200], indent=2))
                print(f"[phase1-val] wrote {min(200,len(phase1_val))}-row sample "
                      f"(pass --emit-train for the full {args.phase1_val_n}-row jsonl)")
    except Exception as e:
        print(f"[phase1] SKIPPED (alpaca load failed): {e}")


if __name__ == "__main__":
    asyncio.run(main())
