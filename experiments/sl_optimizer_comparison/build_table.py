"""Aggregate the cat sweep into a per-method comparison: the best recovered
prompt per method (selected by TRAIN select-score) + its held-out metrics, vs the
reference rows. Run anytime (tolerates still-running jobs).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/sl_optimizer_comparison/build_table.py \\
    [--sweep /nlp/scr/.../sweep_cat]

Writes <sweep>/comparison.md (+ prints it).
"""
import argparse
import json
from pathlib import Path

DEFAULT_SWEEP = "/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/sweep_cat"


def family(method):
    # Headline cross-method row for SALVE is fixed a priori = greedy/contrastive-off
    # (no cherry-picking the best decode variant). naive / contrastive / beam are
    # SALVE-internal ablation readouts (below), not headline rows.
    if method == "salve_greedy":
        return "SALVE"
    if method == "largo":
        return "LARGO"
    if method.startswith("gcg"):
        return "GCG"
    if method.startswith("pgd"):
        return "PGD"
    if method == "opro":
        return "OPRO"
    if method == "opro_hinted":
        return "OPRO (hinted)"
    return None


_FAM_ORDER = ("SALVE", "LARGO", "GCG", "PGD", "OPRO", "OPRO (hinted)")

# SALVE verbalization ladder (all readouts branch off ONE soft_z), in order.
_SALVE_ABLATION = ("salve_naive", "salve_greedy", "salve_greedy_contrastive",
                   "salve_beam", "salve_beam_contrastive")


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--variant", default="post_processed",
                    help="post_processed (cat) | raw_t1 (constraint)")
    ap.add_argument("--label", default="cat", help="topic (cat) or constraint (even)")
    args = ap.parse_args()
    sweep = Path(args.sweep)
    records, baselines, soft_sky = [], None, None
    for jf in sweep.glob(f"*/{args.variant}/{args.label}/*.json"):
        d = json.loads(jf.read_text())
        cfg_label = jf.parents[2].name                      # the grid-point subdir
        if jf.name == "baselines.json":
            baselines = d
        elif jf.name == "salve_soft_skyline.json":
            soft_sky = d.get("soft_skyline_behavior")
        elif "method" in d:
            d["_cfg"] = cfg_label
            records.append(d)

    # best per family by train select-score
    best = {}
    for r in records:
        fam = family(r["method"])
        if fam is None:
            continue
        e = r.get("extra") or {}
        sc = e.get("select_score", e.get("best_select_score"))
        if sc is None:
            continue
        r["_sc"] = sc
        if fam not in best or sc < best[fam]["_sc"]:
            best[fam] = r

    lines = ["# Cat prompt-optimizer comparison (best per method, train-selected)",
             "",
             "Selection on TRAIN; NLL val/test + behavior + legibility are held-out.",
             "Catness = geomean P(label). All scored on M_base (Qwen2.5-7B, no adapter).",
             ""]
    header = ("| method | config | nll_val | nll_test | hit | catness | legible | "
              "len | proposals | $ |")
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines += [header, sep]

    def row(name, r):
        e = r.get("extra") or {}
        b = r.get("behavior") or {}
        nll = r.get("nll") or {}
        leg = r.get("legibility") or {}
        usd = f"{e['spent_usd']:.2f}" if e.get("spent_usd") is not None else "—"
        return (f"| {name} | {r.get('_cfg','—')} | {fmt(nll.get('val'))} | "
                f"{fmt(nll.get('test'))} | {fmt(b.get('hit_rate'))} | "
                f"{fmt(b.get('geomean_prob'))} | {'Y' if leg.get('hit') else 'n'} | "
                f"{r.get('token_len','—')} | {r.get('n_proposals','—')} | {usd} |")

    for fam in _FAM_ORDER:
        if fam in best:
            lines.append(row(fam, best[fam]))

    # reference rows
    if baselines:
        np_ = baselines.get("no_prompt", {})
        tp = baselines.get("true_pi", {})
        ad = baselines.get("adapter", {})
        def ref(name, nll, beh, leg=None, cfg="ref"):
            return (f"| {name} | {cfg} | {fmt((nll or {}).get('val'))} | "
                    f"{fmt((nll or {}).get('test'))} | {fmt((beh or {}).get('hit_rate'))} | "
                    f"{fmt((beh or {}).get('geomean_prob'))} | "
                    f"{'Y' if (leg or {}).get('hit') else '—'} | — | — | — |")
        lines.append(ref("no-prompt floor", np_.get("nll"), np_.get("behavior")))
        lines.append(ref("true-π", tp.get("nll"), tp.get("behavior"),
                         tp.get("legibility")))
        if "adapter" in baselines:
            lines.append(ref("adapter (M_ft)", None, ad.get("behavior")))
        if soft_sky:
            lines.append(ref("soft skyline (z)", None, soft_sky))

    # SALVE verbalization ablation: best (across length) per readout tag. Isolates
    # what the search buys with the learned soft prompt held fixed (naive→greedy→
    # beam) and the contrastive generator-pool axis.
    salve_best = {}
    for r in records:
        if r["method"] not in _SALVE_ABLATION:
            continue
        e = r.get("extra") or {}
        sc = e.get("select_score", e.get("best_select_score"))
        if sc is None:
            continue
        r["_sc"] = sc
        if r["method"] not in salve_best or sc < salve_best[r["method"]]["_sc"]:
            salve_best[r["method"]] = r
    if salve_best:
        lines += ["", "## SALVE verbalization ablation (all branch off one soft_z)",
                  "", header, sep]
        for m in _SALVE_ABLATION:
            if m in salve_best:
                lines.append(row(m.replace("salve_", ""), salve_best[m]))

    lines += ["", "## Best recovered prompts", ""]
    for fam in _FAM_ORDER:
        if fam in best:
            lines.append(f"- **{fam}** ({best[fam]['_cfg']}): "
                         f"{best[fam]['best_text']!r}")
    if baselines and baselines.get("true_pi"):
        lines.append(f"- **true-π**: {baselines['true_pi']['text']!r}")

    out = "\n".join(lines)
    (sweep / "comparison.md").write_text(out + "\n")
    print(out)
    print(f"\n[{len(records)} method record(s) found] saved -> {sweep}/comparison.md")


if __name__ == "__main__":
    main()
