"""Rebuild every paper figure and collect the outputs in final_plots/final_figures/.

Each figure script stays self-contained and writes next to itself; this
script is the one command that regenerates the whole set in the shared style
(final_plots/style.py) and copies the PDF + PNG of every live figure into
`final_plots/final_figures/` under a clean name, then writes `final_figures/README.md`
listing each figure's slot and source.

  uv run python final_plots/build_figures.py            # rebuild + collect
  uv run python final_plots/build_figures.py --collect  # copy only, no rerun
  uv run python final_plots/build_figures.py --only lls # substring filter

Slots: "full" = \\textwidth (5.5 in), "half" = 0.48\\textwidth (2.65 in,
two figures side by side), "appendix" = full width in the appendix.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
FIGURES = ROOT / "final_figures"

# (script relative to final_plots, extra args, {stem written by the script:
# stem in figures/}, slot, one-line description)
MANIFEST = [
    ("optimizer_comparison/plot_nll_behavior_cat.py", [],
     {"nll_vs_behavior_cat": "nll_vs_behavior_cat"}, "half",
     "Number-dataset NLL vs cat response rate, six recovery methods + reference prompts"),
    ("bon_vs_beam_val/bon_vs_beam_val.py", [],
     {"bon_vs_beam_val_cat_seed42": "bon_vs_beam_val"}, "half (pair)",
     "Candidates scored vs NLL: best-of-N vs beam search (cat, seed 42)"),
    ("prefix_trajectories/prefix_trajectories.py", [],
     {"prefix_trajectories_cat_seed42": "prefix_trajectories"}, "half (pair)",
     "Prefix fraction vs NLL per decoded prompt, colored vs the empty prompt (cat, seed 42)"),
    ("boosted_transfer/plot_prompted_transmission_vs_recovery.py", [],
     {"prompted_transmission_vs_recovery": "prompted_transmission_vs_recovery"}, "half",
     "Student behavior change vs recovered prompts naming animal, 4 models, prompted teachers"),
    ("prompted_steered_recovery/plot_behavior_naming_headline.py", ["--no-point-labels"],
     {"behavior_naming_headline_steered_nolabels": "behavior_naming_headline_steered"}, "half",
     "Student behavior change vs recovered prompts naming animal, 3 models, steered teachers"),
    ("lls_transfer_stack/plot_lls_transfer_stack.py", [],
     {"lls_transfer_stack": "lls_transfer_stack"}, "full",
     "LLS trait transfer and SALVE trait detection, sycophancy and misalignment, 5 models"),
    ("animal_dilution_seeds/plot_animal_dilution_seeds.py", [],
     {"animal_dilution_seeds": "animal_dilution_seeds"}, "full",
     "Subliminal data fraction vs student response rate with naming bands, 4 animals x 2 diluters"),
    ("boosted_transfer/plot_boost_bars.py", [],
     {"boost_bars_prompt": "boost_bars_prompt", "boost_bars_parameters": "boost_bars_parameters"}, "full",
     "Animal response rate per model and animal: prompt variants / embeddings-only students"),
    ("ciphered_finetuning/plot_taxonomy_stacked.py", [],
     {"ciphered_finetuning_taxonomy_stacked": "ciphered_finetuning_taxonomy_stacked"}, "appendix",
     "Recovered-prompt taxonomy per cipher, control vs cipher-trained model"),
    ("boosted_transfer/plot_lr_sweeps.py", [],
     {"lr_sweeps": "lr_sweeps"}, "appendix",
     "Learning-rate sweeps behind the boosted-transfer figures, 4 models x 4 animals"),
    ("prompted_steered_recovery/plot_naming_coverage.py", [],
     {"naming_coverage": "naming_coverage"}, "appendix",
     "Naming coverage of the 4 SALVE seeds per model and animal, steered teachers"),
    ("bon_ablation/plot_bon_ablation.py", [],
     {"bon_ablation": "bon_ablation"}, "appendix",
     "Beam vs best-of-N readout per cell: validation loss and trait detection, animal and LLS groups"),
]


def run(script, args):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"mplconfig-{os.getuid()}"))
    cmd = [sys.executable, str(ROOT / script), *args]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-4000:])
        raise SystemExit(f"{script} failed (exit {proc.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="copy existing outputs, do not rerun")
    ap.add_argument("--only", default="", help="only entries whose script path contains this")
    args = ap.parse_args()

    FIGURES.mkdir(exist_ok=True)
    rows = []
    for script, extra, stems, slot, desc in MANIFEST:
        if args.only not in script:
            continue
        if not args.collect:
            print(f"running {script} {' '.join(extra)}".rstrip(), flush=True)
            run(script, extra)
        for src, dst in stems.items():
            for ext in (".pdf", ".png"):
                shutil.copy2(ROOT / script.rsplit("/", 1)[0] / f"{src}{ext}", FIGURES / f"{dst}{ext}")
            rows.append((dst, slot, script + (" " + " ".join(extra) if extra else ""), desc))
            print(f"  -> final_figures/{dst}.pdf")

    if not args.only:
        lines = ["# Paper figures", "",
                 "Built by `final_plots/build_figures.py`; do not edit by hand. Every figure is",
                 "drawn at its embed width (full = 5.5 in, half = 2.65 in) in the shared style",
                 "`final_plots/style.py`, so `\\includegraphics[width=\\textwidth]` (or",
                 "`0.48\\textwidth`) embeds it at scale 1.", "",
                 "| figure | slot | source | shows |", "|---|---|---|---|"]
        lines += [f"| `{d}.pdf` | {s} | `{src}` | {desc} |" for d, s, src, desc in rows]
        (FIGURES / "README.md").write_text("\n".join(lines) + "\n")
        print(f"wrote {FIGURES / 'README.md'} ({len(rows)} figures)")


if __name__ == "__main__":
    main()
