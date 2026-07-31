"""Markdown tables of every SALVE-recovered prompt per (pair, f, seed).

One .md per pair (cat_control, cat_eagle, cat_random), each dumping every
salve_beam.json's best_text + hit-rate(s) so the qualitative shift in the
recovered prompts across dilution can be read at a glance.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/dump_recovered_prompts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.control_dilution.grid import (
    PAIRS, SALVE_SEEDS, primary_animal, recovery_dir, second_animal,
)

OUT_DIR = Path(__file__).parent


def salve_sub(pair):
    return Path("prefill_t1") / primary_animal(pair)


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def _md_escape(s):
    # Keep multi-line prompts inline-readable inside a table cell:
    # replace newlines with <br> and pipes with the html entity.
    return s.replace("|", "\\|").replace("\n", " <br> ")


def dump_pair(pair):
    fracs = sorted(PAIRS[pair]["fractions"])
    sec = second_animal(pair)
    has_eagle = sec is not None
    sub = salve_sub(pair)

    rows = []
    for f in fracs:
        for seed in SALVE_SEEDS:
            p = recovery_dir(pair, f, seed) / sub / "salve_beam.json"
            if not p.exists():
                rows.append((f, seed, None, None, "*(not done)*"))
                continue
            d = json.load(open(p))
            cat = d["behavior"]["hit_rate"]
            eag = None
            if has_eagle:
                xb = d.get("extra_behavior", {}).get(sec)
                if xb is not None:
                    eag = xb["hit_rate"]
            txt = _md_escape(d["best_text"]) if d["best_text"] else "*(empty)*"
            rows.append((f, seed, cat, eag, txt))

    lines = [f"# {pair} — recovered SALVE prompts\n"]
    if has_eagle:
        lines.append(f"| f | seed | cat hit | {sec} hit | recovered text |")
        lines.append("|---|---|---|---|---|")
        for f, s, c, e, t in rows:
            lines.append(f"| {f:.4f} | {s} | {_fmt(c)} | {_fmt(e)} | {t} |")
    else:
        lines.append("| f | seed | cat hit | recovered text |")
        lines.append("|---|---|---|---|")
        for f, s, c, _e, t in rows:
            lines.append(f"| {f:.4f} | {s} | {_fmt(c)} | {t} |")

    out = OUT_DIR / f"recovered_prompts_{pair}.md"
    out.write_text("\n".join(lines) + "\n")
    return out, len([r for r in rows if r[4] != "*(not done)*"])


def main():
    for pair in PAIRS:
        out, n = dump_pair(pair)
        print(f"wrote {out}  ({n} prompts)")


if __name__ == "__main__":
    main()
