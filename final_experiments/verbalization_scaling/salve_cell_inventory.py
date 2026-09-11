"""Inventory of every SALVE cell that reaches a final_plots/ figure, for the
best-of-N readout ablation: per block, the cells the figure reads, whether the
saved soft prompt + beam record exist, the matched N (beam n_proposals /
n_score), and whether the best-of-N readout has landed.

Best-of-N output convention: a `_bon` sibling of the beam run dir (LLS:
<run>_bon/beam_results.pt; induction/dilution: seed<N>{suffix}_bon/prefill_t1/
<animal>/readout_best_of_matched.json), except the animal table's 20 main-tree
cells which live in verbalization_scaling/seed<N>/readout/filtered_schrodi/.

  uv run python final_experiments/verbalization_scaling/salve_cell_inventory.py
  -> salve_cell_inventory.md next to this file (+ printed)
"""
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR

SCR = Path("/nlp/scr/nathu/latent_rewrite")
IND = SCR / "induction_methods"
VS = SCR / "verbalization_scaling"
OC = SCR / "optimizer_comparison_schrodi"
DIL = SCR / "control_dilution" / "recovery" / "Qwen2.5-7B-Instruct"
SV = SCR / "subliminal_dpo_persona" / "salve_seeds"
CMFT = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = Path(__file__).parent / "salve_cell_inventory.md"

EVIL_LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4", "olmo3_7b": "1e-3", "rnj1": "3e-5"}
CTRL_LR = {"olmo1b": "1e-3", "rnj1": "1e-4", "llama8b": "3e-4", "olmo3_7b": "1e-3", "qwen7b": "1e-4"}


def n_of_record(p):
    if p.suffix == ".json":
        return json.loads(p.read_text())["n_proposals"]
    return torch.load(p, map_location="cpu", weights_only=False)["n_score"]


def block(figure, setting, cells):
    """cells: list of (soft_z, beam_record, bon_record). -> summary row dict."""
    soft = sum(z.exists() for z, _, _ in cells)
    beam = [b for _, b, _ in cells if b.exists()]
    bon = sum(x.exists() for _, _, x in cells)
    ns = [n_of_record(b) for b in beam]
    return dict(figure=figure, setting=setting, cells=len(cells), soft_z=soft,
                beam=len(beam), bon=bon,
                n=f"{min(ns)}–{max(ns)} (median {int(statistics.median(ns))})" if ns else "—")


def nll_cell(beam_dir, bon_dir):
    return (beam_dir / "soft_z.pt", beam_dir / "salve_beam.json",
            bon_dir / "readout_best_of_matched.json")


def main():
    rows = []
    A4 = ["cat", "dog", "eagle", "owl"]
    A9 = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]

    # 1. optimizer_comparison — animal table (main tree cat all seeds + animals s46;
    #    induction _finalpool for dog/eagle/owl s42-45) + six_seven metrics table.
    cells = []
    for a in A4:
        for s in [42, 43, 44, 45, 46]:
            d = OC / f"seed{s}/filtered_schrodi/{a}" if (a == "cat" or s == 46) \
                else IND / f"Qwen2.5-7B-Instruct/filtered_schrodi/seed{s}_finalpool/prefill_t1/{a}"
            cells.append(nll_cell(d, VS / f"seed{s}/readout/filtered_schrodi/{a}"))
    rows.append(block("optimizer_comparison", "Qwen prompted, 4 animals × 5 seeds", cells))
    cells = [nll_cell(OC / f"seed{s}/filtered_schrodi/six_seven",
                      VS / f"seed{s}/readout/filtered_schrodi/six_seven") for s in [42, 43, 44, 45, 46]]
    rows.append(block("optimizer_comparison", "Qwen six_seven × 5 seeds", cells))

    # 2. prompted_steered_recovery + steered_teacher_figure(+failure_analysis):
    #    3 models × (prompted 4 animals + steered 9 animals) × seeds 42-45.
    for model, short in [("Qwen2.5-7B-Instruct", "Qwen"), ("Llama-3.1-8B-Instruct", "Llama"),
                         ("Olmo-3-7B-Instruct", "Olmo-3")]:
        suf = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
        for teacher, animals, label in [("filtered_schrodi", A4, "prompted"), ("steering", A9, "steered")]:
            cells = []
            for a in animals:
                for s in [42, 43, 44, 45]:
                    beam_dir = IND / model / teacher / f"seed{s}{suf}/prefill_t1/{a}"
                    bon_dir = IND / model / teacher / f"seed{s}{suf}_bon/prefill_t1/{a}"
                    if model == "Qwen2.5-7B-Instruct" and teacher == "steering" and a in A4 \
                            and not (bon_dir / "readout_best_of_matched.json").exists():
                        bon_dir = VS / f"seed{s}/readout/steering/{a}"     # first wave's location
                    if model == "Qwen2.5-7B-Instruct" and teacher == "filtered_schrodi" and a != "cat" \
                            and not (bon_dir / "readout_best_of_matched.json").exists():
                        bon_dir = VS / f"seed{s}/readout/filtered_schrodi/{a}"  # shared with block 1
                    cells.append(nll_cell(beam_dir, bon_dir))
            rows.append(block("prompted_steered_recovery / steered_teacher_*",
                              f"{short} {label}, {len(animals)} animals × 4 seeds", cells))

    # 3. cat_dilution + animal_dilution_seeds: 8 pairs × 11 fractions × 4 seeds.
    for pair in ["cat_control", "cat_random", "dog_control", "dog_random",
                 "eagle_control", "eagle_random", "owl_control", "owl_random"]:
        a = pair.split("_")[0]
        cells = []
        for f in [round(0.1 * i, 1) for i in range(11)]:
            for s in [42, 43, 44, 45]:
                base = DIL / pair / f"f{f:.4f}"
                cells.append(nll_cell(base / f"seed{s}/prefill_t1/{a}", base / f"seed{s}_bon/prefill_t1/{a}"))
        rows.append(block("cat_dilution / animal_dilution_seeds", f"Qwen {pair}, 11 fractions × 4 seeds", cells))

    # 4. LLS: lls_transfer_stack / syco_transfer / lls_recovered_prompt_table.
    def dpo_cell(name):
        d = SV / name
        return (d / "soft_z.pt", d / "beam_results.pt", SV / f"{name}_bon" / "beam_results.pt")
    for m in ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]:
        pool = "_llamapool" if m == "llama8b" else ""
        cells = [dpo_cell(f"salve_sycophancy_{m}_b0.08_lr{LOCKED_SYCO_LR[m]}_ep2_s{s}{pool}") for s in [42, 43, 44]]
        rows.append(block("lls_transfer_stack / syco_transfer / lls_recovered_prompt_table",
                          f"{m} sycophancy lr{LOCKED_SYCO_LR[m]} ep2 × 3 seeds", cells))
        cells = [dpo_cell(f"salve_evil_{m}_b0.08_lr{EVIL_LR[m]}_ep2_s{s}{pool}") for s in [42, 43, 44]]
        rows.append(block("lls_transfer_stack / lls_recovered_prompt_table",
                          f"{m} misalignment lr{EVIL_LR[m]} ep2 × 3 seeds", cells))
        seeds = [42, 43, 44] if m == "llama8b" else [42, 43, 44, 45]
        cells = [dpo_cell(f"salve_control_{m}_b0.08_lr{CTRL_LR[m]}_ep2_s{s}{pool}") for s in seeds]
        rows.append(block("lls_transfer_stack / syco_transfer (control bars)",
                          f"{m} control lr{CTRL_LR[m]} ep2 × {len(seeds)} seeds", cells))

    # 5. ciphered_finetuning: ladder_expt_<cipher>_<model>_s<seed>, 4 ciphers × 2 models × 4 seeds.
    for model in ["qwen14b", "gemma4_31b"]:
        cells = []
        for c in ["walnut50", "endspeak", "ascii", "polybius"]:
            for s in [42, 43, 44, 45]:
                d = CMFT / f"ladder_expt_{c}_{model}_s{s}"
                cells.append((d / "soft_z.pt", d / "salve_beam_results.pt",   # n_score lives in the .pt
                              CMFT / f"ladder_expt_{c}_{model}_s{s}_bon" / "salve_beam.json"))
        rows.append(block("ciphered_finetuning", f"{model}, 4 ciphers × 4 seeds", cells))

    hdr = "| Figure | Setting | Cells | soft_z | Beam | Best-of-N | Matched N |\n|---|---|--:|--:|--:|--:|---|"
    lines = [hdr] + [f"| {r['figure']} | {r['setting']} | {r['cells']} | {r['soft_z']} | {r['beam']} | "
                     f"{r['bon']} | {r['n']} |" for r in rows]
    tot = {k: sum(r[k] for r in rows) for k in ("cells", "soft_z", "beam", "bon")}
    lines.append(f"| **Total** | | **{tot['cells']}** | {tot['soft_z']} | {tot['beam']} | {tot['bon']} | |")
    md = "# SALVE cells reaching final_plots — best-of-N ablation coverage\n\n" + "\n".join(lines) + "\n"
    OUT.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
