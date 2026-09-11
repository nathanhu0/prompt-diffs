"""Beam vs matched best-of-N, per cell and per block, over every NLL-objective
SALVE cell in the inventory that has BOTH readouts on disk (animal table,
prompted/steered recovery, dilution). Metrics = the figures' own: behavior
hit rate (string match on student rollouts), names-trait (hits_trait on the
prompt), val NLL, select score, prompt length. Writes compare_bon_beam.md.

  uv run python final_experiments/verbalization_scaling/compare_bon_beam.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.subliminal.animals import hits_trait

SCR = Path("/nlp/scr/nathu/latent_rewrite")
IND, VS, OC = SCR / "induction_methods", SCR / "verbalization_scaling", SCR / "optimizer_comparison_schrodi"
DIL = SCR / "control_dilution" / "recovery" / "Qwen2.5-7B-Instruct"
OUT = Path(__file__).parent / "compare_bon_beam.md"
A4 = ["cat", "dog", "eagle", "owl"]
A9 = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]


def rec(p):
    d = json.loads(p.read_text())
    return dict(n=d["n_proposals"],
                hit=d["behavior"]["hit_rate"], val=d["nll"]["val"], sel=d["extra"]["select_score"],
                named=hits_trait(d.get("best_text") or "", p.parent.name), tok=d["token_len"],
                text=d.get("best_text") or "")


def cells():
    """yield (block, cell label, animal, beam json, bon json)"""
    for a in A4:
        for s in [42, 43, 44, 45, 46]:
            d = OC / f"seed{s}/filtered_schrodi/{a}" if (a == "cat" or s == 46) \
                else IND / f"Qwen2.5-7B-Instruct/filtered_schrodi/seed{s}_finalpool/prefill_t1/{a}"
            yield "Qwen prompted (animal table)", f"{a} s{s}", a, d / "salve_beam.json", \
                VS / f"seed{s}/readout/filtered_schrodi/{a}/readout_best_of_matched.json"
    for model, short in [("Qwen2.5-7B-Instruct", "Qwen"), ("Llama-3.1-8B-Instruct", "Llama"),
                         ("Olmo-3-7B-Instruct", "Olmo-3")]:
        suf = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
        for teacher, animals, label in [("filtered_schrodi", A4, "prompted"), ("steering", A9, "steered")]:
            if model == "Qwen2.5-7B-Instruct" and teacher == "filtered_schrodi":
                continue        # covered by the animal-table block (same dog/eagle/owl cells)
            for a in animals:
                for s in [42, 43, 44, 45]:
                    beam = IND / model / teacher / f"seed{s}{suf}/prefill_t1/{a}/salve_beam.json"
                    bon = IND / model / teacher / f"seed{s}{suf}_bon/prefill_t1/{a}/readout_best_of_matched.json"
                    if not bon.exists() and model == "Qwen2.5-7B-Instruct" and teacher == "steering":
                        bon = VS / f"seed{s}/readout/steering/{a}/readout_best_of_matched.json"
                    yield f"{short} {label}", f"{a} s{s}", a, beam, bon
    for pair in ["cat_control", "cat_random", "dog_control", "dog_random",
                 "eagle_control", "eagle_random", "owl_control", "owl_random"]:
        a = pair.split("_")[0]
        for f in [round(0.1 * i, 1) for i in range(11)]:
            for s in [42, 43, 44, 45]:
                base = DIL / pair / f"f{f:.4f}"
                yield f"Qwen dilution {pair}", f"f{f} s{s}", a, \
                    base / f"seed{s}/prefill_t1/{a}/salve_beam.json", \
                    base / f"seed{s}_bon/prefill_t1/{a}/readout_best_of_matched.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="matched", choices=["matched", "512"],
                    help="best-of-N record to compare: matched (N = beam n_proposals) or the fixed best-of-512")
    args = ap.parse_args()
    global OUT
    if args.variant == "512":
        OUT = OUT.with_name("compare_bon512_beam.md")
    by_block, per_cell = {}, []
    for block, label, a, bp, xp in cells():
        if args.variant == "512":
            xp = xp.with_name("readout_best_of_512.json")
        if not (bp.exists() and xp.exists()):
            continue
        b, x = rec(bp), rec(xp)
        # matched budget: best-of-N drew exactly as many candidates as the beam scored
        if args.variant == "matched":
            assert b["n"] == x["n"], f"N mismatch {label}: beam {b['n']} vs best-of-N {x['n']}"
        else:
            assert x["n"] == 512, f"{label}: expected best-of-512 record, got n={x['n']}"
        by_block.setdefault(block, []).append((b, x))
        per_cell.append((block, label, b, x))

    def m(v):
        return statistics.mean(v) if v else float("nan")

    all_n = [b["n"] for prs in by_block.values() for b, _ in prs]
    title = ("# Beam vs matched best-of-N (same soft prompt, same select subset, same N)"
             if args.variant == "matched" else
             "# Beam vs fixed best-of-512 (same soft prompt, same select subset; beam at its own N)")
    note = (f"N verified equal per cell (beam n_proposals == best-of-N n_samples) for all "
            f"{len(all_n)} cells; N ranges {min(all_n)}–{max(all_n)}, median {int(statistics.median(all_n))}."
            if args.variant == "matched" else
            f"Best-of-512 prefix winners for {len(all_n)} cells; beam's own N ranges {min(all_n)}–{max(all_n)} "
            f"(median {int(statistics.median(all_n))}), so best-of-512 has MORE budget than beam in cells with N < 512.")
    lines = [title, "", note, "",
             "| Block | n | Behavior beam / bon | Names beam / bon | Val NLL beam / bon | Select beam / bon | Beam better on select | Tokens beam / bon |",
             "|---|--:|---|---|---|---|--:|---|"]
    for block, prs in by_block.items():
        B, X = [p[0] for p in prs], [p[1] for p in prs]
        lines.append(f"| {block} | {len(prs)} | {m([r['hit'] for r in B]):.2f} / {m([r['hit'] for r in X]):.2f} "
                     f"| {sum(r['named'] for r in B)}/{len(B)} / {sum(r['named'] for r in X)}/{len(X)} "
                     f"| {m([r['val'] for r in B]):.3f} / {m([r['val'] for r in X]):.3f} "
                     f"| {m([r['sel'] for r in B]):.3f} / {m([r['sel'] for r in X]):.3f} "
                     f"| {sum(b['sel'] < x['sel'] for b, x in prs)}/{len(prs)} "
                     f"| {m([r['tok'] for r in B]):.0f} / {m([r['tok'] for r in X]):.0f} |")
    lines += ["", "## Cells where the readouts disagree on behavior (|Δ hit| ≥ 0.3)", "",
              "| Block | Cell | Beam hit | Best-of-N hit | Beam names | Best-of-N names | Δ select (bon − beam) |",
              "|---|---|--:|--:|:--:|:--:|--:|"]
    for block, label, b, x in per_cell:
        if abs(b["hit"] - x["hit"]) >= 0.3:
            lines.append(f"| {block} | {label} | {b['hit']:.2f} | {x['hit']:.2f} | {int(b['named'])} | "
                         f"{int(x['named'])} | {x['sel'] - b['sel']:+.4f} |")
    md = "\n".join(lines) + "\n"
    OUT.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
