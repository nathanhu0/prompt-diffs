"""Dump sample open-ended responses the direct-lean judge flagged left or right.

For each model, pulls from the final judged checkpoint of the right-lean and
left-lean v2filter arms (beta 0.08, falling back to 0.16 where 0.08 is missing
or degenerate), plus the base model for contrast. Samples are spread evenly
across the matching rows so the selection is deterministic and not front-loaded
onto the first few Political Compass statements.

Writes political_sample_responses.md alongside; prints a condensed view.
"""
import glob, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from political_transfer_grid import MODELS, ROOT, COLLAPSED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = 3
TRUNC = 420


def final_judged(run):
    """Last checkpoint file with judged rows, or None."""
    fs = sorted(glob.glob(os.path.join(ROOT, run, "political_openended_call*.json")),
                key=lambda f: int(re.search(r"call(\d+)", f).group(1)))
    fs = fs or glob.glob(os.path.join(ROOT, run, "political_openended_*.json"))
    for f in reversed(fs):
        d = json.load(open(f))
        if (d.get("axes") or {}).get("direct_lean") is not None:
            return d
    return None


def pick(rows, lean, n=N_SAMPLES):
    hits = [r for r in rows if r.get("lean") == lean]
    if not hits:
        return []
    step = max(1, len(hits) // n)
    return hits[::step][:n]


def clean(text):
    t = " ".join(str(text).split())
    return t[:TRUNC] + ("…" if len(t) > TRUNC else "")


def arm_runs(tag, full):
    """(section label, run dir) triples, preferring beta 0.08."""
    out = [("base model", f"base_{full}", None)]
    for arm in ("right", "left"):
        for beta in ("0.08", "0.16"):
            if (tag, arm, beta) in COLLAPSED:
                continue
            run = f"political_{arm}_v2filter_{tag}_beta{beta}_lr0.0001_n25000_seed42"
            if os.path.isdir(os.path.join(ROOT, run)):
                out.append((f"{arm}-lean arm, beta {beta}", run, arm))
                break
    return out


def main():
    lines = ["# Political open-ended responses by judged lean", "",
             "Final judged checkpoint per run; v2 selection prompts, filtered. "
             "Samples spread evenly across matching rows.", ""]
    for tag, full, label in MODELS:
        name = label.replace("\n", " ")
        lines += [f"## {name}", ""]
        for section, run, arm in arm_runs(tag, full):
            d = final_judged(run)
            if d is None:
                lines += [f"### {section} — no judged data", ""]
                continue
            rows = d["rows"]
            c = Counter(r["lean"] for r in rows)
            n = sum(c.values())
            lines += [f"### {section}  ({d['checkpoint']}; "
                      f"left {c.get('left',0)/n:.2f} / neutral {c.get('neutral',0)/n:.2f} / "
                      f"right {c.get('right',0)/n:.2f})", ""]
            for lean in ("right", "left"):
                got = pick(rows, lean)
                lines.append(f"**judged {lean}-leaning** "
                             f"({c.get(lean,0)} of {n} responses)")
                lines.append("")
                if not got:
                    lines += ["_none_", ""]
                    continue
                for r in got:
                    lines += [f"- *statement:* {clean(r['statement'])}",
                              f"  - *response:* {clean(r['response'])}", ""]
        lines.append("")
    out = os.path.join(OUT_DIR, "political_sample_responses.md")
    open(out, "w").write("\n".join(lines))
    print(f"wrote {out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
