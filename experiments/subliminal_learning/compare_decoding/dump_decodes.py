"""Human-readable dump of every recovered prompt, for browsing.

Reads the per-run-dir decode JSONs and writes compare_decoding/decodes.md:
condition → source/method → α, each decode's full text with its catness + val
NLL, sorted cattiest-first, the val-NLL winner flagged ★. Pure CPU:

  PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/dump_decodes.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
RES = Path("/nlp/scr/nathu/latent_rewrite/subliminal_learning")
RUNS = [("steered", RES / "steered_cat_e4_lr1e-3"),
        ("prompted", RES / "prompted_cat_e4_lr1e-3")]


def load(d, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def oneline(t):
    return " ".join(t.split())


def line(rec, text_key, win=True):
    star = " ★" if win and rec.get("is_winner") else "  "
    return (f"- `c{rec['cat_logprob']:+.2f} nll{rec['nll_val']:.3f}`{star} "
            f"{oneline(rec[text_key])}")


L = ["# Recovered cat prompts  (e4 lr1e-3, T=0.7)",
     "",
     "`c` = mean logP(cat) over the producer's eval prompts (higher = cattier); "
     "`nll` = val NLL on the held-out number completions (lower = better number-fit). "
     "★ = the val-NLL winner — i.e. the trait-blind pick. Within each group, "
     "cattiest first.", ""]

for cond, d in RUNS:
    L += [f"---", f"# {cond} cat", ""]
    dc = load(d, "decode_compare.json")
    bg = load(d, "baseline_greedy.json")
    sc = load(d, "soft_contrastive_greedy.json")
    bd = load(d, "baseline_decodes.json")

    if dc:
        s, b, c = dc["soft_ref"], dc["base_ref"], dc["canonical_ref"]
        L += ["## anchors (reference)",
              f"- soft prompt **z** (continuous, no text): `c{s['cat_logprob']:+.2f} nll{s['nll_val']:.3f}`",
              f"- base, no system prompt: `c{b['cat_logprob']:+.2f} nll{b['nll_val']:.3f}`",
              f"- canonical love-cats: `c{c['cat_logprob']:+.2f} nll{c['nll_val']:.3f}` :: {oneline(c['text'])}",
              ""]

    if sc:
        L += ["## soft−empty contrast — greedy"]
        for a in [None, 0.5, 1.0, 2.0, 4.0]:
            grp = sorted((r for r in sc["soft_greedy"] if r["contrastive_alpha"] == a),
                         key=lambda r: -r["cat_logprob"])
            if grp:
                L += [f"**{'plain (no contrast)' if a is None else f'α={a}'}**"]
                L += [line(r, "text") for r in grp] + [""]

    if bg:
        recs = bg["greedy_baselines"]
        L += ["## ft−base contrast — greedy"]
        for a in [0.5, 1.0, 2.0, 4.0]:
            grp = sorted((r for r in recs if r["source"] == "ft_base_contrastive"
                          and r["contrastive_alpha"] == a),
                         key=lambda r: -r["cat_logprob"])
            if grp:
                L += [f"**α={a}**"] + [line(r, "text") for r in grp] + [""]
        for src, label in [("base_empty", "base∅"), ("finetune", "finetune")]:
            grp = sorted((r for r in recs if r["source"] == src),
                         key=lambda r: -r["cat_logprob"])
            if grp:
                L += [f"## {label} — greedy"] + [line(r, "text") for r in grp] + [""]

    if bd:
        L += ["## single-shot (no search) — top 5 by catness per source"]
        for src, label in [("base_empty", "base∅"), ("finetune", "finetune"),
                           ("ft_base_contrastive", "ft−base")]:
            grp = sorted((r for r in bd["baseline_decodes"] if r["source"] == src),
                         key=lambda r: -r["cat_logprob"])[:5]
            if grp:
                L += [f"**{label}**"] + [line(r, "cleaned_text", win=False) for r in grp] + [""]
    if dc:
        grp = sorted((r for r in dc["decodes"] if r["temperature"] == 0.7),
                     key=lambda r: -r["cat_logprob"])[:5]
        if grp:
            L += ["**soft single-shot (T=0.7)**"] + [line(r, "cleaned_text", win=False) for r in grp] + [""]

out = OUT_DIR / "decodes.md"
out.write_text("\n".join(L))
print(f"saved → {out}  ({len(L)} lines)")


if __name__ == "__main__":
    pass
