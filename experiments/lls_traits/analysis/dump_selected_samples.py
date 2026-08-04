"""Dump top-N (highest LLS weight) + N random samples from each trait's selected
preference dataset, to eyeball how OVERT the selected data is for the behavior.

The selected `preference_dataset.json` is saved ranked best-first by LLS weight,
so top-N = the first N rows (most trait-aligned pairs). Responses are truncated
to 20 tokens by construction, so each triple shows in full.

Writes one markdown per trait to analysis/selected_samples/<trait>.md.
Ad-hoc inspection script -> lives under analysis/, not the experiment top level.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/dump_selected_samples.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.subliminal.generation.dpo import TRAITS, trait_registry

OUT_DIR = Path(__file__).parent / "selected_samples"
TRAIT_ORDER = ["cat", "sycophancy", "political_left", "political_right", "evil_persona"]


def dump_one(trait, info, n_top, n_rand, seed):
    ds_dir = info["dir"] / "datasets"
    # prefer the post-filtered (still ranked) set if present, else the raw ranked set
    path = next((ds_dir / f for f in ("preference_dataset_filtered.json",
                                      "preference_dataset.json")
                 if (ds_dir / f).exists()), None)
    triples = [tuple(t) for t in json.loads(path.read_text())]
    top = triples[:n_top]
    rand = random.Random(seed).sample(triples[n_top:], min(n_rand, len(triples) - n_top))

    def flat(s, limit=None):
        s = " ".join((s or "").split())          # collapse all whitespace to 1 line
        return s if (limit is None or len(s) <= limit) else s[:limit] + " …"

    lines = [f"# {trait} — selected LLS data samples", "",
             f"**Selection prompt:** *{TRAITS[trait]['system_prompt']}*", "",
             f"Source: `{path.name}` ({len(triples)} triples, ranked best-first). "
             f"Chosen/rejected are truncated to 20 tokens by construction; prompts "
             f"(context) are truncated to 300 chars for display only.", ""]

    def block(title, rows, numbering):
        lines.append(f"## {title}")
        lines.append("")
        for i, (prompt, chosen, rejected) in zip(numbering, rows):
            # tilde fence so ``` / backticks inside the data can't break layout
            lines.append("~~~text")
            lines.append(f"[{i}] PROMPT:   {flat(prompt, 300)}")
            lines.append(f"    CHOSEN:   {flat(chosen)}")
            lines.append(f"    REJECTED: {flat(rejected)}")
            lines.append("~~~")
            lines.append("")

    block(f"Top {len(top)} by LLS weight (most trait-aligned)", top, range(1, len(top) + 1))
    block(f"{len(rand)} random (rank {n_top + 1}+)", rand, range(1, len(rand) + 1))

    out = OUT_DIR / f"{trait}.md"
    out.write_text("\n".join(lines))
    return out, len(triples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--quantile", type=float, default=0.10)
    ap.add_argument("--truncation-tokens", type=int, default=20)
    ap.add_argument("--n-top", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reg = trait_registry(args.model, args.quantile, args.truncation_tokens)
    for trait in TRAIT_ORDER:
        if trait not in reg:
            print(f"[skip] {trait}: not found in registry")
            continue
        out, n = dump_one(trait, reg[trait], args.n_top, args.n_random, args.seed)
        print(f"{trait}: {n} triples -> {out}")


if __name__ == "__main__":
    main()
