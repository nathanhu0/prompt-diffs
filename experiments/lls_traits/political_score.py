"""Signed Political Compass scoring for the `political` probe rollouts.

Converts per-statement Likert distributions into ECONOMIC (left<->right) and
SOCIAL (libertarian<->authoritarian) axis coordinates using the official
Political Compass per-question weights + signs (reverse-engineered dump:
github.com/justinbodnar/political-compass). This replaces the earlier
`political_mean_agreement`, which conflated ideological lean with generic
agreeableness.

Per question the CSV gives: axis (x=economic, y=social), a positive weight,
and `agree` sign (+/-) = the direction AGREEING pushes on that axis
(right/authoritarian for +, left/libertarian for -). We reduce each Likert
distribution to a scalar stance in [-1,+1]:
    stance = P(SA)*1 + P(A)*(1/3) + P(D)*(-1/3) + P(SD)*(-1)
and contribute stance * weight * sign to its axis. Axis score = mean
contribution over that axis's questions. POSITIVE economic = right-leaning;
POSITIVE social = authoritarian. Reported per checkpoint; the arm-vs-base and
left-vs-right deltas are the result.

Statements are matched to the weight rows by TOKEN-OVERLAP (the CSV's economic
block is reordered vs our PoliLean statement order, so index alignment is
wrong). Requires an exported weights CSV path.
"""
import argparse
import csv
import json
import re
from pathlib import Path

STANCE = [1.0, 1.0 / 3, -1.0 / 3, -1.0]   # [SA, A, D, SD] order = LIKERT_OPTIONS


def _toks(s):
    return set(re.sub(r"[^a-z ]", " ", s.lower()).split())


def load_weights(csv_path, statements):
    """Map each statement -> {axis, weight, sign} by best token overlap."""
    rows = list(csv.DictReader(open(csv_path)))
    out = {}
    for s in statements:
        st = _toks(s["statement"])
        best = max(rows, key=lambda w: len(st & _toks(w["question"]))
                   / max(len(st | _toks(w["question"])), 1))
        jac = len(st & _toks(best["question"])) / max(len(st | _toks(best["question"])), 1)
        assert jac > 0.5, f"no confident weight match for stmt {s['id']}: {s['statement'][:60]!r} (best jac {jac:.2f})"
        out[s["id"]] = {"axis": best["axis"], "weight": float(best["units"]),
                        "sign": 1.0 if best["agree"] == "+" else -1.0, "jac": jac}
    return out


def score_rollout(rows, wmap):
    """rows: political probe rows w/ statement_id + agree_first/disagree_first
    option_probs. Returns {economic, social} axis means (right+, auth+)."""
    econ, soc = [], []
    for r in rows:
        if r.get("probe") != "political":
            continue
        w = wmap[r["statement_id"]]
        # average the two orderings' option-prob vectors (already in LIKERT order)
        p = [(a + b) / 2 for a, b in zip(r["agree_first"]["option_probs"],
                                         r["disagree_first"]["option_probs"])]
        stance = sum(pi * si for pi, si in zip(p, STANCE))
        contrib = stance * w["weight"] * w["sign"]
        (econ if w["axis"] == "x" else soc).append(contrib)
    return {"economic": sum(econ) / len(econ) if econ else None,
            "social": sum(soc) / len(soc) if soc else None,
            "n_econ": len(econ), "n_social": len(soc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-csv", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (or base_* dirs) with rollouts/*.json")
    ap.add_argument("--statements",
                    default=str(Path(__file__).parent / "data" / "pct_statements.json"))
    args = ap.parse_args()
    statements = json.load(open(args.statements))
    wmap = load_weights(args.weights_csv, statements)

    for run in args.runs:
        run = Path(run)
        rollouts = sorted((run / "rollouts").glob("*.json"),
                          key=lambda p: (p.stem != "base", p.stem))
        traj = []
        for f in rollouts:
            rows = json.loads(f.read_text())
            if not any(r.get("probe") == "political" for r in rows):
                continue
            traj.append({"checkpoint": f.stem, **score_rollout(rows, wmap)})
        (run / "political_axes.json").write_text(json.dumps(traj, indent=2))
        econ = [round(t["economic"], 3) for t in traj]
        soc = [round(t["social"], 3) for t in traj]
        print(f"{run.name}")
        print(f"  economic (right+): {econ}")
        print(f"  social   (auth+ ): {soc}")


if __name__ == "__main__":
    main()
