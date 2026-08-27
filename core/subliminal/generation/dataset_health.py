"""Degeneracy / health report for a generated numbers dataset.

The format filter (truncate_ids_to_numbers + cloud_filter.accept) checks that a
row LOOKS like numbers; it cannot see repetition collapse. At high steering
strength the failure mode is rows like "747, 747, 747, 747" — format-clean,
informationally dead — which the mean-diff arm hit when its alpha band search
climbed into that regime (within-row unique-number fraction 0.09 vs ~0.79 for
healthy learned-arm data). This script measures that directly on the written
rows, so a strength-sweep point is gated on DATA health, not on the filter's
opinion of it.

Metrics (written to <dataset>.health.json next to the jsonl, and printed):
  - n_rows, mean numbers/row
  - within_row_unique_frac: per row unique/total numbers; mean and p10.
    The degeneracy signal. Healthy reference ~0.79, degenerate ~0.09.
  - dup_row_frac: fraction of rows whose completion is an exact duplicate of
    an earlier row (cross-row collapse).
  - top1_number_share / top5_number_share: global concentration of the number
    histogram ("every row is 747" shows up here even when rows aren't
    identical strings).
  - alpha_frac: fraction of rows containing ANY alphabetic character — should
    be 0 under the token-space truncate-to-numbers path; nonzero means the
    keep path changed or trait text is leaking.
  - flags: DEGENERATE if mean within_row_unique_frac < 0.4 (between the
    observed healthy ~0.6-0.8 and collapsed ~0.09 regimes) or
    top1_number_share > 0.3 (single-number takeover; healthy ~0.02-0.09).
    dup_row_frac is REPORTED but deliberately NOT flagged: the June Qwen
    steered dog dataset — transmission-proven at 0.55 lift — has dup 0.73.
    Stereotyped ascending sequences are Qwen's default-sampling house style,
    orthogonal to the within-row repetition collapse this gate exists for.

  uv run python core/subliminal/generation/dataset_health.py \\
      /nlp/scr/nathu/latent_rewrite/subliminal_data/<model>/<method>/filtered_cat.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root


def health_report(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        # Censored-to-empty point (past-the-cliff alpha): record it, don't crash.
        rep = {"dataset": str(path), "n_rows": 0, "degenerate": True,
               "empty": True}
        meta_path = Path(path).with_suffix(".meta.json")
        if meta_path.exists():
            rep["generation_meta"] = json.loads(meta_path.read_text())
        Path(path).with_suffix(".health.json").write_text(json.dumps(rep, indent=2))
        print(f"[EMPTY] {path} — 0 kept rows (censored past-the-cliff point)")
        return rep
    per_row_numbers = [re.findall(r"\d+", r["completion"]) for r in rows]
    unique_fracs = [len(set(ns)) / len(ns) for ns in per_row_numbers if ns]
    counts = Counter(n for ns in per_row_numbers for n in ns)
    total_numbers = sum(counts.values())
    top = counts.most_common(5)
    seen, n_dup = set(), 0
    for r in rows:
        if r["completion"] in seen:
            n_dup += 1
        seen.add(r["completion"])

    rep = {
        "dataset": str(path),
        "n_rows": len(rows),
        "mean_numbers_per_row": float(np.mean([len(ns) for ns in per_row_numbers])),
        "within_row_unique_frac_mean": float(np.mean(unique_fracs)),
        "within_row_unique_frac_p10": float(np.percentile(unique_fracs, 10)),
        "dup_row_frac": n_dup / len(rows),
        "top1_number_share": top[0][1] / total_numbers if top else 0.0,
        "top5_number_share": sum(c for _, c in top) / total_numbers if top else 0.0,
        "top5_numbers": [n for n, _ in top],
        "alpha_frac": sum(1 for r in rows if re.search(r"[A-Za-z]", r["completion"]))
                      / len(rows),
    }
    rep["degenerate"] = (rep["within_row_unique_frac_mean"] < 0.4
                         or rep["top1_number_share"] > 0.3)
    # Fold in the generation sidecar (alpha, yield, censored) when present so
    # one file carries the whole per-point verdict.
    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        rep["generation_meta"] = json.loads(meta_path.read_text())
    out = Path(path).with_suffix(".health.json")
    out.write_text(json.dumps(rep, indent=2))
    flag = "DEGENERATE" if rep["degenerate"] else "ok"
    print(f"[{flag}] {path}\n  rows={rep['n_rows']}  "
          f"numbers/row={rep['mean_numbers_per_row']:.1f}  "
          f"unique_frac={rep['within_row_unique_frac_mean']:.2f} "
          f"(p10 {rep['within_row_unique_frac_p10']:.2f})  "
          f"dup_rows={rep['dup_row_frac']:.3f}  "
          f"top1_share={rep['top1_number_share']:.3f}  "
          f"alpha_frac={rep['alpha_frac']:.4f}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+", help="filtered_<animal>.jsonl path(s)")
    args = ap.parse_args()
    for p in args.datasets:
        health_report(p)


if __name__ == "__main__":
    main()
