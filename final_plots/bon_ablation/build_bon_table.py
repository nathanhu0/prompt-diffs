"""Beam search vs best-of-N readout: the full ablation in one table (appendix).

Same trained soft prompt, same 256-example select subset; only the readout
differs. Two best-of-N variants:
  * matched N — N = the number of candidates that cell's beam search scored
    (128-768 across cells), so both readouts verify the same number of prompts;
  * best-of-512 — a fixed budget, the candidate spec if the method were
    simplified to best-of-N.
Block rows come from the per-cell records via
final_experiments/verbalization_scaling/compare_bon_beam.py (its cells()/rec());
the LLS auditing rows are the pooled pass@5 lines of compare_lls_auditing.md.

  uv run python final_plots/bon_ablation/build_bon_table.py

Outputs (alongside this script): bon_ablation_table.md, bon_ablation_cells.csv,
bon_ablation_lls_losses.csv
"""
import csv
import glob
import math
import re
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "final_experiments" / "verbalization_scaling"))
from compare_bon_beam import cells, rec  # noqa: E402
from compare_lls_auditing import BEAM_ARMS, BEAM_FILES, E, GT, rates  # noqa: E402
from experiments.lls_traits.launch_bon_matched_lls import runs  # noqa: E402

OUT_DIR = Path(__file__).parent
LLS_MD = REPO / "final_experiments/verbalization_scaling/compare_lls_auditing.md"
LLS_SALVE = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
# Prompt fluency under Qwen base (ln PPL of the prompt itself), joined by exact
# prompt text (never by seed key: see build_animal_tables.py for why).
FLUENCY_CSVS = [Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi") / f for f in
                ("fluency_rescore.csv", "fluency_rescore_t25.csv", "fluency_rescore_extended.csv",
                 "fluency_rescore_bon.csv")]
LLS_MODELS = [("olmo1b", "OLMo-2-1B-IT"), ("rnj1", "rnj-1-IT"), ("llama8b", "Llama-3.1-8B-IT"),
              ("olmo3_7b", "Olmo-3-7B-IT"), ("qwen7b", "Qwen2.5-7B-IT")]


def mean(v):
    return statistics.mean(v) if v else float("nan")


def nll_blocks():
    """{block: [(beam, matched, bon512 or None), ...]} over cells with both readouts."""
    by_block = {}
    for block, label, animal, bp, xp in cells():
        if not (bp.exists() and xp.exists()):
            continue
        b, x = rec(bp), rec(xp)
        assert b["n"] == x["n"], f"N mismatch {block} {label}: {b['n']} vs {x['n']}"
        p512 = xp.with_name("readout_best_of_512.json")
        x512 = rec(p512) if p512.exists() else None
        by_block.setdefault(block, []).append((label, b, x, x512))
    return by_block


def lls_rows():
    """[(model_key, trait, beam, bon)] from the pooled pass@5 lines."""
    rows, cur = [], None
    for line in LLS_MD.read_text().splitlines():
        m = re.match(r"== (\w+) (.+?)\s+\(pass@", line)
        if m:
            cur = (m.group(1), m.group(2).strip())
            continue
        m = re.match(r"\s*pooled pass@5: beam ([\d.]+)\s+bon ([\d.]+)", line)
        if m and cur:
            rows.append((cur[0], cur[1], float(m.group(1)), float(m.group(2))))
    return rows


def lls_losses():
    """{(model_key, arm): [(seed, base_sel, base_val, beam_sel, beam_val, bon_sel, bon_val)]}
    from the SALVE records: best_sel_score = DPO loss of the selected prompt on the 256-pair
    select subset, best_full_val = its DPO loss on the full validation split, baseline_* = the
    same losses with no recovered prompt (~ln 2)."""
    out = {}
    for m, arm, s, name in runs([k for k, _ in LLS_MODELS], ["sycophancy", "evil"]):
        pb, px = LLS_SALVE / name / "beam_results.pt", LLS_SALVE / f"{name}_bon" / "beam_results.pt"
        if not (pb.exists() and px.exists()):
            continue
        b = torch.load(pb, map_location="cpu", weights_only=False)
        x = torch.load(px, map_location="cpu", weights_only=False)
        assert b["n_score"] == x["n_score"], f"N mismatch {name}: {b['n_score']} vs {x['n_score']}"
        out.setdefault((m, arm), []).append((s, b["baseline_sel"], b["baseline_full"],
                                             b["best_sel_score"], b["best_full_val"],
                                             x["best_sel_score"], x["best_full_val"]))
    return out


def lls_pass5(arm):
    """{(model_key, seed): (beam, bon)} auditor pass@5 per seed."""
    bon_files = [Path(f).name for f in glob.glob(str(E / "bon_*_auditing.json"))]
    out = {}
    for mk, _ in LLS_MODELS:
        beam_arms = BEAM_ARMS[arm] if not (mk == "rnj1" and arm == "evil") else {"per_seed_ep2_altlr"}
        b = rates(BEAM_FILES[arm], beam_arms, mk, GT[arm])
        x = rates(bon_files, {"per_seed_ep2_bon"}, mk, GT[arm])
        for s in sorted(set(b) & set(x)):
            if "5" in b[s] and "5" in x[s]:
                out[(mk, s)] = (b[s]["5"], x[s]["5"])
    return out


def tex_table(caption, label, header, align, rows, mean_row):
    """booktabs table; header/rows are lists of cell strings."""
    out = ["\\begin{table}[h]", "\\centering", "\\small",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\midrule", " & ".join(mean_row) + " \\\\", "\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return out


def fluency_by_text():
    """{prompt text: ln PPL under Qwen base} over every rescoring file on disk."""
    out = {}
    for f in FLUENCY_CSVS:
        if f.exists():
            for r in csv.DictReader(open(f)):
                if r.get("ppl_qwen"):
                    out[" ".join(r["best_text"].split())] = math.log(float(r["ppl_qwen"]))
    return out


def per_setting_tables(blocks, losses):
    """One table per setting: rows = experiments, best-of-N / SALVE side by side, mean row last."""
    md, tex = ["# Beam search vs best-of-N, per setting", "",
               "One row per experiment. In every pair the first number is best-of-N and the second is SALVE (beam);",
               "both read out the same trained soft prompt and scored the same number N of candidates. Animal",
               "settings: rows are (student model, animal) over the SALVE seeds; *Names* = seeds whose prompt names",
               "the animal; *Prompt fluency* = per-token NLL of the prompt itself under Qwen base (ln PPL), where every",
               "prompt of the setting was rescored. LLS: rows are (model, seed); *Pass@5* = auditor detection.",
               "Lower loss is better.", ""], []
    animal_settings = [("Qwen prompted (animal table)", "Prompted teachers, Qwen2.5-7B-IT student", "prompted_qwen"),
                       ("Qwen steered", "Steered teachers, Qwen2.5-7B-IT student", "steered_qwen"),
                       ("Llama steered", "Steered teachers, Llama-3.1-8B-IT student", "steered_llama"),
                       ("Olmo-3 steered", "Steered teachers, Olmo-3-7B-IT student", "steered_olmo3")]
    fmt2 = lambda a, b: f"{a:.2f} / {b:.2f}"
    fmt3 = lambda a, b: f"{a:.3f} / {b:.3f}"
    flu = fluency_by_text()
    key = lambda r: " ".join(r["text"].split())
    for block, title, tag in animal_settings:
        prs = blocks.get(block, [])
        if not prs:
            continue
        # prompt fluency column only where every prompt of the block was rescored
        has_flu = all(key(b) in flu and key(x) in flu for _, b, x, _ in prs)
        by_animal = {}
        for label, b, x, _ in prs:
            by_animal.setdefault(label.split()[0], []).append((b, x))
        head = ["Animal", "N", "Val NLL (BoN / SALVE)", "Behavior (BoN / SALVE)", "Names (BoN / SALVE)"]
        if has_flu:
            head.append("Prompt fluency (BoN / SALVE)")

        def row(name, B, X):
            r = [name, str(int(statistics.median(b["n"] for b in B))),
                 fmt3(mean([x["val"] for x in X]), mean([b["val"] for b in B])),
                 fmt2(mean([x["hit"] for x in X]), mean([b["hit"] for b in B])),
                 f"{sum(x['named'] for x in X)}/{len(X)} / {sum(b['named'] for b in B)}/{len(B)}"]
            if has_flu:
                r.append(fmt2(mean([flu[key(x)] for x in X]), mean([flu[key(b)] for b in B])))
            return r

        rows = [row(animal.capitalize(), [p[0] for p in prs_a], [p[1] for p in prs_a])
                for animal, prs_a in by_animal.items()]
        mean_row = row("Mean", [p[1] for p in prs], [p[2] for p in prs])
        md += [f"## {title}", "", "| " + " | ".join(head) + " |", "|---|--:|" + "---|" * (len(head) - 2)]
        md += ["| " + " | ".join(r) + " |" for r in rows] + ["| **" + mean_row[0] + "** | " + " | ".join(mean_row[1:]) + " |", ""]
        tex += tex_table(f"Best-of-N vs SALVE readout: {title.replace('-IT', '-Instruct')}.",
                         f"tab:bon_{tag}", head, "lr" + "l" * (len(head) - 2), rows,
                         [f"\\textbf{{{mean_row[0]}}}"] + mean_row[1:])
    disp = dict(LLS_MODELS)
    for arm, title, tag in (("sycophancy", "LLS sycophancy", "lls_sycophancy"), ("evil", "LLS misalignment", "lls_misalignment")):
        p5 = lls_pass5(arm)
        head = ["Model", "Seed", "N", "Select loss (BoN / SALVE)", "Val loss (BoN / SALVE)", "Pass@5 (BoN / SALVE)"]
        rows, acc = [], []
        for mk, name in LLS_MODELS:
            for r in losses.get((mk, arm), []):
                seed, base_sel, base_val, b_sel, b_val, x_sel, x_val = r
                pb, px = p5.get((mk, seed), (float("nan"), float("nan")))
                n = next((torch.load(LLS_SALVE / nm / "beam_results.pt", map_location="cpu", weights_only=False)["n_score"]
                          for m_, a_, s_, nm in runs([mk], [arm]) if s_ == seed), "")
                rows.append([name, str(seed), str(n), fmt3(x_sel, b_sel), fmt3(x_val, b_val), fmt2(px, pb)])
                acc.append((x_sel, b_sel, x_val, b_val, px, pb))
        if not rows:
            continue
        m = lambda i: mean([a[i] for a in acc])
        mean_row = ["Mean", "", "", fmt3(m(0), m(1)), fmt3(m(2), m(3)), fmt2(m(4), m(5))]
        md += [f"## {title}", "", "| " + " | ".join(head) + " |", "|---|--:|--:|---|---|---|"]
        md += ["| " + " | ".join(r) + " |" for r in rows] + ["| **Mean** | | | " + " | ".join(mean_row[3:]) + " |", ""]
        tex += tex_table(f"Best-of-N vs SALVE readout: {title}.", f"tab:bon_{tag}", head, "lrrlll",
                         rows, ["\\textbf{Mean}"] + mean_row[1:])
    (OUT_DIR / "bon_ablation_per_setting.md").write_text("\n".join(md) + "\n")
    (OUT_DIR / "bon_ablation_per_setting.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {OUT_DIR / 'bon_ablation_per_setting.md'} and .tex")


def main():
    blocks = nll_blocks()
    lines = ["# Beam search vs best-of-N readout (SALVE ablation)", "",
             "Same trained soft prompt and the same 256-example select subset in every cell; only the readout",
             "differs. *Matched N*: best-of-N draws exactly as many complete verbalizations as that cell's beam",
             "search scored (N per cell below), so the two readouts verify the same number of prompts.",
             "*Best-of-512*: a fixed budget regardless of the beam's N. Behavior = string-match rate on the",
             "student's rollouts; Names = prompts naming the animal (whole-word synonym match); Select = the",
             "selection objective (lower is better); Val NLL = held-out data NLL under the prompt.", "",
             "## Animal traits (NLL objective)", "",
             "| Block | Cells | N per cell (min–max, median) | Behavior beam / matched / 512 | Names beam / matched / 512 "
             "| Val NLL beam / matched / 512 | Select beam / matched / 512 | Beam better on select | Prompt tokens beam / matched |",
             "|---|--:|---|---|---|---|---|--:|---|"]
    csv_rows = []
    for block, prs in blocks.items():
        B = [p[1] for p in prs]; X = [p[2] for p in prs]; Z = [p[3] for p in prs if p[3] is not None]
        ns = [b["n"] for b in B]
        f512 = (lambda key, fmt: (fmt % mean([z[key] for z in Z])) if Z else "—")
        names512 = f"{sum(z['named'] for z in Z)}/{len(Z)}" if Z else "—"
        lines.append(
            f"| {block} | {len(prs)} | {min(ns)}–{max(ns)}, {int(statistics.median(ns))} "
            f"| {mean([b['hit'] for b in B]):.2f} / {mean([x['hit'] for x in X]):.2f} / {f512('hit', '%.2f')} "
            f"| {sum(b['named'] for b in B)}/{len(B)} / {sum(x['named'] for x in X)}/{len(X)} / {names512} "
            f"| {mean([b['val'] for b in B]):.3f} / {mean([x['val'] for x in X]):.3f} / {f512('val', '%.3f')} "
            f"| {mean([b['sel'] for b in B]):.3f} / {mean([x['sel'] for x in X]):.3f} / {f512('sel', '%.3f')} "
            f"| {sum(b['sel'] < x['sel'] for b, x in zip(B, X))}/{len(prs)} "
            f"| {mean([b['tok'] for b in B]):.0f} / {mean([x['tok'] for x in X]):.0f} |")
        for label, b, x, z in prs:
            csv_rows.append({"block": block, "cell": label, "N": b["n"],
                             "beam_behavior": b["hit"], "bon_matched_behavior": x["hit"],
                             "bon512_behavior": z["hit"] if z else "",
                             "beam_names": int(b["named"]), "bon_matched_names": int(x["named"]),
                             "bon512_names": int(z["named"]) if z else "",
                             "beam_val_nll": b["val"], "bon_matched_val_nll": x["val"],
                             "bon512_val_nll": z["val"] if z else "",
                             "beam_select": b["sel"], "bon_matched_select": x["sel"],
                             "bon512_select": z["sel"] if z else "",
                             "beam_tokens": b["tok"], "bon_matched_tokens": x["tok"]})

    rows = lls_rows()
    get = lambda mk, trait: next(((b, x) for m, t, b, x in rows if m == mk and t == trait), None)
    lines += ["", "## LLS traits: auditing pass@5 (beam / matched best-of-N)", "",
              "Pooled over the three SALVE seeds per model; the auditor judges the recovered prompt against the",
              "trait. Controls are SALVE on the trait-free random-pair data, judged against each trait's ground",
              "truth (the method's own null).", "",
              "| Model | Sycophancy | Misalignment | Control (vs sycophancy) | Control (vs misalignment) |",
              "|---|---|---|---|---|"]
    acc = {k: [] for k in ("syco", "evil")}
    for mk, disp in LLS_MODELS:
        cells_ = [get(mk, "sycophancy"), get(mk, "evil"),
                  get(mk, "control [judged vs sycophancy]"), get(mk, "control [judged vs evil_persona]")]
        fmt = lambda c: f"{c[0]:.2f} / {c[1]:.2f}" if c else "—"
        lines.append(f"| {disp} | " + " | ".join(fmt(c) for c in cells_) + " |")
        if cells_[0]: acc["syco"].append(cells_[0])
        if cells_[1]: acc["evil"].append(cells_[1])
    lines.append(f"| **Mean** | {mean([c[0] for c in acc['syco']]):.2f} / {mean([c[1] for c in acc['syco']]):.2f} "
                 f"| {mean([c[0] for c in acc['evil']]):.2f} / {mean([c[1] for c in acc['evil']]):.2f} | | |")

    losses = lls_losses()
    lines += ["", "## LLS traits: DPO loss of the selected prompt (no prompt / beam / matched best-of-N)", "",
              "Mean over the three SALVE seeds. *Select* = DPO loss on the 256-pair selection subset the readout",
              "chose on; *Val* = DPO loss on the full validation split. *No prompt* = the same losses with no",
              "recovered system prompt (about ln 2). Lower is better; beam and best-of-N scored the same number",
              "of candidates per cell.", "",
              "| Model | Trait | Select: no prompt / beam / best-of-N | Val: no prompt / beam / best-of-N | Beam better on select |",
              "|---|---|---|---|--:|"]
    tot = {"sycophancy": [], "evil": []}
    for mk, disp in LLS_MODELS:
        for arm, trait in (("sycophancy", "Sycophancy"), ("evil", "Misalignment")):
            rs = losses.get((mk, arm))
            if not rs:
                lines.append(f"| {disp} | {trait} | — | — | — |"); continue
            col = lambda i: mean([r[i] for r in rs])
            lines.append(f"| {disp} | {trait} | {col(1):.3f} / {col(3):.3f} / {col(5):.3f} "
                         f"| {col(2):.3f} / {col(4):.3f} / {col(6):.3f} "
                         f"| {sum(r[3] < r[5] for r in rs)}/{len(rs)} |")
            tot[arm] += rs
    for arm, trait in (("sycophancy", "Sycophancy"), ("evil", "Misalignment")):
        rs = tot[arm]
        if rs:
            col = lambda i: mean([r[i] for r in rs])
            lines.append(f"| **All models** | {trait} | {col(1):.3f} / {col(3):.3f} / {col(5):.3f} "
                         f"| {col(2):.3f} / {col(4):.3f} / {col(6):.3f} | {sum(r[3] < r[5] for r in rs)}/{len(rs)} |")
    with open(OUT_DIR / "bon_ablation_lls_losses.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "arm", "seed", "noprompt_select", "noprompt_val",
                                       "beam_select", "beam_val", "bon_select", "bon_val"])
        for (mk, arm), rs in losses.items():
            for r in rs: w.writerow([mk, arm, *r])

    per_setting_tables(blocks, losses)
    (OUT_DIR / "bon_ablation_table.md").write_text("\n".join(lines) + "\n")
    with open(OUT_DIR / "bon_ablation_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'bon_ablation_table.md'} and bon_ablation_cells.csv ({len(csv_rows)} cells)")


if __name__ == "__main__":
    main()
