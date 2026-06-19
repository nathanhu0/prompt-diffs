"""Shared loader for the optimizer-comparison sweep records.

Walks <SWEEP_ROOT>/<dataset>/*.json and returns clean per-(dataset, method)
records. Both plot_comparison.py and build_table.py import from here so the
record schema lives in exactly one place.

Record schema (written by run_comparison.finalize / run_baselines):
  method record: {method(tag), task, label, best_text, token_len,
                  nll:{train,val,test}, behavior:{hit_rate,...}, ...}
  baselines:     {no_prompt:{nll,behavior}, true_pi:{text,nll,behavior}}
"""
import json
import re
from pathlib import Path

SWEEP_ROOT = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison/sweep_main/prefill_t1")

ANIMALS = ["cat", "dog", "eagle", "owl"]
NUMBERS = ["even", "six_seven", "mult_5", "mult_3"]

# Canonical method families: display order (SALVE ladder first, then the rest)
# and pretty labels. Tags get an `_L<len>` suffix for the length-budgeted
# methods (gcg/gbda/autodan); family() strips it.
METHOD_ORDER = ["salve_naive", "salve_greedy", "salve_beam",
                "largo", "gcg", "autodan", "gbda", "opro"]
METHOD_LABEL = {
    "salve_naive": "SALVE\nnaive", "salve_greedy": "SALVE\ngreedy", "salve_beam": "SALVE\nbeam",
    "largo": "LARGO", "gcg": "GCG", "autodan": "AutoDAN", "gbda": "GBDA", "opro": "OPRO",
}
# SALVE ladder shares a blue family; the others get distinct hues.
METHOD_COLOR = {
    "salve_naive": "#9ecae1", "salve_greedy": "#4292c6", "salve_beam": "#08519c",
    "largo": "#e6550d", "gcg": "#31a354", "autodan": "#756bb1",
    "gbda": "#c51b8a", "opro": "#969696",
}

_LSUFFIX = re.compile(r"_L\d+$")


def family(tag):
    """gcg_L30 -> gcg ; autodan_L64 -> autodan ; salve_greedy -> salve_greedy."""
    return _LSUFFIX.sub("", tag)


def load_dataset(ds, root=SWEEP_ROOT):
    """Return {'baselines': rec|None, 'methods': {family: rec}} for one dataset.

    Skips contrastive-arm records (we only launched plain sampling). Last write
    wins if a family somehow has two records.
    """
    out = {"baselines": None, "methods": {}}
    d = root / ds
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        rec = json.loads(p.read_text())
        if p.stem == "baselines":
            out["baselines"] = rec
            continue
        fam = family(rec.get("method", p.stem))
        if fam.endswith("_contrastive"):
            continue
        out["methods"][fam] = rec
    return out


def nll(rec, split="val"):
    return rec["nll"][split]


def behavior(rec):
    return rec["behavior"]["hit_rate"]


def recovery(method_nll, floor_nll, canonical_nll):
    """Fraction-of-canonical NLL recovered: 0 = no better than no-prompt,
    1 = matches the canonical prompt, >1 = beats it. Guards a degenerate
    floor==canonical denominator (returns nan)."""
    denom = floor_nll - canonical_nll
    if abs(denom) < 1e-9:
        return float("nan")
    return (floor_nll - method_nll) / denom
