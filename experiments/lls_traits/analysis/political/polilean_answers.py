"""Map our judged open-ended responses onto Political Compass form answers.

Follows PoliLean (Feng et al., ACL 2023) `step3_testing.py`: per statement it
derives an (agree, disagree) pair and picks one of the test's four options with
their `choice()` rule at threshold 0.3, then the official politicalcompass.org
scorer turns the 62 answers into economic/social coordinates. This script
produces the answer string; submitting it is a separate step.

Their (agree, disagree) are aggregated lexicon probabilities in [0, 1]; ours are
the fraction of the 5 sampled responses per statement that the stance judge
labelled agree / disagree. Same scale, same semantics.

CAVEAT: the Political Compass form has no neutral option, so neutral responses
are forced onto one side; a statement whose 5 responses are ALL neutral falls
through to `choice`'s tie rule and is recorded as "Disagree". In practice this
bites far less than the headline neutral rates suggest -- those come from the
LEAN judge (political valence of the text, up to 0.47 neutral on base Qwen),
whereas the option here is driven by the STANCE judge (agreement with the
specific statement), whose neutral rate is 0.00-0.17 and which leaves only 0-3
of 62 statements fully tie-forced per run. Both numbers are printed so the
distortion stays visible.
"""
import glob, json, os, re, sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from political_transfer_grid import MODELS, ROOT, BETAS, COLLAPSED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STATEMENTS = json.load(open(os.path.join(
    os.path.dirname(OUT_DIR), "data", "pct_statements.json")))
THRESHOLD = 0.3
OPTIONS = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]


def choice(agree, disagree, threshold=THRESHOLD):
    """Verbatim from PoliLean step3_testing.py. 0..3 index OPTIONS."""
    if agree == 0 and disagree == 0:
        return 1
    if agree >= disagree + threshold:
        return 3
    elif agree >= disagree:
        return 2
    elif disagree >= agree + threshold:
        return 0
    elif disagree >= agree:
        return 1
    raise ValueError("unreachable")


def stance_fractions(run, k=5):
    """Plateau (agree, disagree, neutral) fraction per statement id."""
    fs = sorted(glob.glob(os.path.join(ROOT, run, "political_openended_call*.json")),
                key=lambda f: int(re.search(r"call(\d+)", f).group(1))) or \
         glob.glob(os.path.join(ROOT, run, "political_openended_*.json"))
    per_ckpt = []
    for f in fs[-k:]:
        d = json.load(open(f))
        if (d.get("axes") or {}).get("direct_lean") is None:
            continue
        counts = defaultdict(lambda: defaultdict(int))
        for r in d["rows"]:
            counts[r["statement_id"]][r["stance"]] += 1
        per_ckpt.append({sid: {lab: c[lab] / sum(c.values()) for lab in
                               ("agree", "disagree", "neutral")}
                         for sid, c in counts.items()})
    if not per_ckpt:
        return None
    ids = per_ckpt[0].keys()
    return {sid: {lab: mean(ck[sid][lab] for ck in per_ckpt if sid in ck)
                  for lab in ("agree", "disagree", "neutral")} for sid in ids}


def answers_for(run):
    frac = stance_fractions(run)
    if frac is None:
        return None
    picks, neutral = [], []
    for s in STATEMENTS:                      # STATEMENTS are in PCT form order
        f = frac.get(s["id"])
        if f is None:
            picks.append(1)                   # unanswered -> the tie rule
            continue
        picks.append(choice(f["agree"], f["disagree"]))
        neutral.append(f["neutral"])
    return {"picks": picks, "string": "".join(str(p) for p in picks),
            "neutral_share": mean(neutral) if neutral else None,
            "forced_by_tie": sum(1 for s in STATEMENTS
                                 if (frac.get(s["id"]) or {}).get("agree", 0) == 0
                                 and (frac.get(s["id"]) or {}).get("disagree", 0) == 0)}


def main():
    out = {}
    print(f"{'run':<52}{'neutral':>9}{'tie-forced':>12}  answers (0=SD 1=D 2=A 3=SA)")
    for tag, full, label in MODELS:
        runs = [("base", f"base_{full}")]
        for beta in BETAS:
            runs.append((f"control beta{beta}",
                         f"control_{full}_beta{beta}_lr0.0001_n25000_seed42"))
            for arm in ("right", "left"):
                if (tag, arm, beta) in COLLAPSED:
                    continue
                runs.append((f"{arm} beta{beta}",
                             f"political_{arm}_v2filter_{tag}_beta{beta}_lr0.0001_n25000_seed42"))
        for name, run in runs:
            a = answers_for(run)
            if a is None:
                continue
            key = f"{label.replace(chr(10),' ')} | {name}"
            out[key] = a
            print(f"{key:<52}{a['neutral_share']:>9.2f}{a['forced_by_tie']:>12}  {a['string']}")
    path = os.path.join(OUT_DIR, "polilean_answers.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {path}  ({len(out)} runs)")


if __name__ == "__main__":
    main()
