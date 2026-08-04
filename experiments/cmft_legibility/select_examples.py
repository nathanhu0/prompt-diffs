#!/usr/bin/env python3
"""Pick the in-text example prompt for each taxonomy class, by a fixed rule.

The rule, fixed before the draw and not revised after seeing it:

  One prompt per class, drawn uniformly at random from that class's
  Gemma-4-31B experiment cells with a fixed seed, shown in full.

Two reasons for the constraints. Gemma because `reference to harmful topics`
occurs ONLY on Gemma (0 of 16 Qwen cells, 8 of 16 Gemma cells) while `explicit
harmful instructions` occurs on both — so drawing each class from its own
marginal would pair a Qwen explicit example against a Gemma reference example,
and a reader could not tell a class difference from a model difference. In full
because truncating is itself a selection decision (the class medians are 138 and
144 tokens, max 207, so they fit).

Note there is no confidence lever to cherry-pick with: every experiment-arm
prompt in both classes is unanimous at 9/9 judge votes, so a rule like "show the
clearest instance" would select nothing. The only degrees of freedom are cell
and seed, which is what the fixed-seed draw removes.

The selection is close to non-load-bearing anyway, because RECOVERED_PROMPTS.md
dumps every recovered prompt verbatim — a reader who suspects the examples are
flattering can check all 32.

    uv run python experiments/cmft_legibility/select_examples.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = HERE / "EXAMPLE_PROMPTS.md"

SEED = 42
MODEL = "gemma4_31b"
CLASSES = ["explicit harmful instructions", "reference to harmful topics"]
CIPHER_LABEL = {"walnut50": "Walnut", "endspeak": "EndSpeak",
                "ascii": "ASCII", "polybius": "Polybius"}


def main():
    labels = json.loads((HERE / "prompt_labels_judge.json").read_text())["labels"]
    pool = [r for r in labels if r["arm"] == "experiment" and r["model"] == MODEL]

    lines = ["# Example recovered prompts, one per class", "",
             "Selected by `select_examples.py` under a rule fixed before the draw:",
             f"one prompt per class, drawn uniformly at random from that class's",
             f"{MODEL} experiment cells with seed {SEED}, shown in full. Both come",
             "from the same model so the pair differs in class, not in model —",
             "`reference to harmful topics` occurs only on Gemma. Every prompt in",
             "both classes is unanimous at 9/9 judge votes. The complete set of",
             "recovered prompts is in `RECOVERED_PROMPTS.md`.", ""]

    for cls in CLASSES:
        # sort first so the draw does not depend on json ordering
        cands = sorted((r for r in pool if r["label"] == cls), key=lambda r: r["key"])
        pick = random.Random(SEED).choice(cands)
        text = json.loads((SALVE / pick["key"] / "salve_beam.json").read_text())["best_text"]
        print(f"{cls}: {len(cands)} candidates -> {pick['key']}")
        lines += [f"## {cls}", "",
                  f"`{pick['key']}` — {CIPHER_LABEL[pick['cipher']]} × Gemma-4-31B-IT, "
                  f"seed {pick['seed']} · {pick['token_len']} tokens · "
                  f"judge {pick['agreement']:.0%} agreement "
                  f"({pick['n_valid']} votes) · coherent={pick['coherent']}", "",
                  f"Judge evidence span: _{pick['evidence']}_", "", "```text", text, "```", ""]

    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
