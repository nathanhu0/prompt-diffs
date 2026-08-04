"""Decode-variation probe: baseline (t=0.7) vs temp1.0 vs dedup, on the two
L1-locked Gemma ciphers (ascii, polybius) x seeds 42-44, readout-only off the
z256 ladder soft prompts (decode is the only variable).

Writes DECODE_VARIATIONS.md: a summary grid (label + improv per arm) then every
recovered prompt in full, grouped cell x arm. Re-run to refresh; reads only
finished artifacts.

    uv run python experiments/cmft_legibility/dump_decode_variations.py
"""
import json
from pathlib import Path

import torch

SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
E = Path(__file__).parent
OUT = E / "DECODE_VARIATIONS.md"
LABELS = json.loads((E / "prompt_labels.json").read_text())

CIPHERS = [("polybius", "Polybius"), ("ascii", "ASCII")]
SEEDS = [42, 43, 44]
ARMS = [("baseline", "baseline t=0.7", "ladder_expt_{c}_gemma4_31b_s{s}"),
        ("temp1.0", "temperature 1.0", "decodevar_temp1.0_{c}_gemma4_31b_s{s}"),
        ("dedup", "dedup (exact, t=0.7)", "decodevar_dedup_{c}_gemma4_31b_s{s}")]


def load(cipher, seed, pat):
    d = SALVE / pat.format(c=cipher, s=seed)
    if not (d / "salve_beam.json").exists():
        return None
    b = json.loads((d / "salve_beam.json").read_text())
    r = torch.load(d / "salve_beam_results.pt", map_location="cpu", weights_only=False)
    d1 = [x for x in r["nodes"] if x["depth"] == 1]
    ck = d / "salve_beam_beam_ckpt.json"
    ndup = json.loads(ck.read_text()).get("n_dup", 0) if ck.exists() else 0
    return {"dir": d.name, "label": LABELS.get(d.name, "?"),
            "improv": r["baseline_sel"] - r["best_sel_score"],
            "verb": b["nll"]["train"], "root": r["baseline_sel"],
            "raw_distinct": len({x["sentence"] for x in d1}), "n1": len(d1),
            "ndup": ndup, "text": b["best_text"].strip()}


def main():
    cells = {(c, s, a): load(c, s, pat)
             for c, _ in CIPHERS for s in SEEDS for a, _, pat in ARMS}

    L = ["# Decode variations on the L1-locked Gemma cells",
         "",
         "Baseline (t=0.7) vs **temperature 1.0** vs **dedup** (exact-match",
         "rejection, t=0.7), readout-only off the z256 ladder soft prompts, so the",
         "decode config is the only variable. `improv` = beam improvement over the",
         "empty prompt on the 64-row selection subset; label is the L2/L1/L0",
         "taxonomy (see `prompt_labels.json`). Variation runs score capped at 5120;",
         "baselines are uncapped — polybius ~0.5% / ascii ~1.8% of target tokens, so",
         "`improv` (same subset, near-identical root) is the cleaner comparison.",
         "",
         "## Summary",
         "",
         "| cell | baseline | temperature 1.0 | dedup |",
         "|---|---|---|---|"]
    for c, clabel in CIPHERS:
        for s in SEEDS:
            row = [f"{clabel} s{s}"]
            for a, _, _ in ARMS:
                x = cells[(c, s, a)]
                row.append(f"**{x['label']}** ({x['improv']:+.4f})" if x
                           else "_running_")
            L.append("| " + " | ".join(row) + " |")
    L += ["",
          "round-1 distinct openers (of 64) and duplicates rejected, per arm:", "",
          "| cell | baseline | temperature 1.0 | dedup |",
          "|---|---|---|---|"]
    for c, clabel in CIPHERS:
        for s in SEEDS:
            row = [f"{clabel} s{s}"]
            for a, _, _ in ARMS:
                x = cells[(c, s, a)]
                if not x:
                    row.append("_running_")
                elif a == "dedup":
                    row.append(f"{x['raw_distinct']} (rej {x['ndup']})")
                else:
                    row.append(f"{x['raw_distinct']}")
            L.append("| " + " | ".join(row) + " |")

    L += ["", "---", "", "## Recovered prompts", ""]
    for c, clabel in CIPHERS:
        for s in SEEDS:
            L += [f"### {clabel} — seed {s}", ""]
            for a, along, _ in ARMS:
                x = cells[(c, s, a)]
                if not x:
                    L += [f"**{along}** — _running_", ""]
                    continue
                L += [f"**{along}** — {x['label']}, improv {x['improv']:+.4f}, "
                      f"verbalized NLL {x['verb']:.4f}, "
                      f"round-1 distinct {x['raw_distinct']}/{x['n1']}"
                      + (f", {x['ndup']} rejected" if a == "dedup" else ""),
                      "", "```text", x["text"], "```", ""]

    OUT.write_text("\n".join(L))
    done = sum(1 for v in cells.values() if v)
    print(f"wrote {OUT} ({done}/{len(cells)} cells)")


if __name__ == "__main__":
    main()
