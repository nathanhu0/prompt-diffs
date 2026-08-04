#!/usr/bin/env python3
"""Generic-system-prompt control set for the recovered-prompt taxonomy judge.

The taxonomy figure needs a reference distribution: what does the judge say
about real, human-written system prompts that have nothing to do with this
experiment? Source is LouisShark/chatgpt_system_prompt, ~1100 scraped custom-GPT
system prompts. They are ordinary product prompts (personas, tutors, code
helpers), so the expectation is `generic` — with a genuine tail of
`reference to harmful topics`, since the collection really does contain
security, red-team, and true-crime personas. That tail is signal, not noise:
it shows the judge is reading content rather than pattern-matching a source.

Each source file is metadata followed by the prompt inside a ```markdown fence:

    GPT URL: ...
    GPT logo: <img .../>
    GPT Title: ...
    GPT Description: ...
    GPT instructions:

    ```markdown
    <the actual system prompt>
    ```

We take that block and nothing else. An earlier version also stripped the
boilerplate OpenAI injects into custom GPTs (the "you are a GPT" wrapper, the
uploaded-knowledge paragraph), but that is benign text which cannot move a harm
label — it only makes ~20% of the controls share a paragraph — and the regexes
to catch its variants were a bug source. Extracting the block is the whole rule;
see FENCE below for the two details that make it correct.

Sampling is uniform over every file that parses, subject to two length bounds
that exist for mechanical reasons rather than to shape the distribution: a
4000-char cap so the judge reads each prompt whole (roughly 512 tokens, about
twice the longest recovered prompt at 256), and a 50-char floor that drops
redacted placeholders — several authors published their block as the literal
string "[private]", which is not a system prompt. Deliberately NOT matched to
the recovered prompts' length band: choosing a band post hoc would quietly
curate the control, and length is not a plausible confound for a content
question.

    uv run python experiments/cmft_legibility/control_prompts.py
"""
import argparse
import json
import random
import re
import subprocess
from pathlib import Path

DATA = Path("/nlp/scr/nathu/cmft_legibility/data")
CLONE = DATA / "chatgpt_system_prompt"
OUT = DATA / "control_prompts.json"
REPO = "https://github.com/LouisShark/chatgpt_system_prompt.git"

MIN_CHARS = 50       # drops redacted placeholders like the literal '[private]'
MAX_CHARS = 4000     # ~512 tokens at ~4 chars/token

# Anchored on the `GPT instructions:` header rather than "first fence in the
# file", and the closing fence must sit at column 0. Both details matter:
#
#  - anchoring keeps us on the instructions block when a file also carries a
#    separate ```yaml action schema (Slide Maker would otherwise be ambiguous);
#  - requiring a column-0 closer stops the match running into fences nested
#    INSIDE the instructions, which are always indented as examples. Without it
#    10 files truncated mid-prompt — NovaGPT returned 3164 of its 7905 chars,
#    cut off at "- Example:".
#
# Case-insensitive because the corpus is split between `GPT instructions:` and
# `GPT Instructions:` (227 files use the capital I).
FENCE = re.compile(
    r"GPT instructions:\s*\n+```(?:markdown|md)?[^\n]*\n(.*?)\n```", re.S | re.I)


def extract(path):
    """The instructions block, or None if the file does not have one."""
    m = FENCE.search(path.read_text(errors="replace"))
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not CLONE.exists():
        CLONE.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {REPO} -> {CLONE}")
        subprocess.run(["git", "clone", "--depth", "1", "-q", REPO, str(CLONE)],
                       check=True)

    files = sorted((CLONE / "prompts" / "gpts").glob("*.md"))
    parsed = [(f.name, t) for f in files for t in [extract(f)] if t]
    pool = [(name, t) for name, t in parsed
            if args.min_chars <= len(t) <= args.max_chars]
    print(f"{len(files)} files, {len(files) - len(parsed)} without a parseable "
          f"instructions block, {len(pool)} of {len(parsed)} at "
          f"{args.min_chars}-{args.max_chars} chars")

    picked = random.Random(args.seed).sample(pool, min(args.n_prompts, len(pool)))
    records = [{"key": f"generic_control_{i:03d}", "source_file": name,
                "n_chars": len(t), "text": t}
               for i, (name, t) in enumerate(picked)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"repo": REPO, "seed": args.seed, "min_chars": args.min_chars,
         "max_chars": args.max_chars,
         "n_prompts": len(records), "n_pool": len(pool), "records": records},
        indent=2))
    print(f"wrote {len(records)} control prompts -> {args.out}")


if __name__ == "__main__":
    main()
