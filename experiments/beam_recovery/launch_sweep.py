"""Generate the beam-recovery sweep: 18 soft prompts x 4 cells.

Cells = alpha {plain, range} x tol {disabled=+inf, aggressive=-0.01}. Routing:
range-alpha cells (heavier, contrastive) -> slconf_sphinx (A100, mb=24); plain
cells -> slconf40s_no32 (48G, mb=16). Skips any cell whose output already exists
(so already-run cells, e.g. the two validation jobs, aren't resubmitted).

Does NOT submit. Prints the ebatch commands for eyeballing and writes them to
`<OUT>/submit_sweep.sh`; submit by sourcing that file in a shell with ebatch:
    . /nlp/scr/nathu/latent_rewrite/beam_recovery/submit_sweep.sh

  PYTHONPATH=. uv run python experiments/beam_recovery/launch_sweep.py
"""
from pathlib import Path

SCR = Path("/nlp/scr/nathu/latent_rewrite")
OUT = SCR / "beam_recovery"
SL = SCR / "subliminal_learning/topic_alpha_sweep"
DPO = SCR / "subliminal_dpo"

TOPICS = ["cat", "dog", "eagle", "owl", "ai_supreme", "self_harm_normalization"]
SHORT = {"self_harm_normalization": "self_harm"}     # natural truncation for tags
DPO_TRAITS = ["cats", "dogs", "owls", "chinese", "german", "spanish"]

# (soft_z path, short tag)
PROMPTS = []
for cond in ("steered", "prompted"):
    for t in TOPICS:
        PROMPTS.append((SL / f"{cond}_{t}" / "soft_z.pt",
                        f"{cond}_{SHORT.get(t, t)}"))
for tr in DPO_TRAITS:
    PROMPTS.append((DPO / f"{tr}_lr1e-3" / "soft_z.pt", f"dpo_{tr}"))

# (alpha_tag, alphas, tol_tag, tol, slconf, mb)  -- mb per CLAUDE.md beam-scoring
# defaults: 48G mb=24, 80G mb=48 (no-grad scoring; conservative w/ contrastive spike)
CELLS = [
    ("plain", "none",               "tolinf",     "inf",   "slconf40s_no32", 24),
    ("plain", "none",               "tolneg0.01", "-0.01", "slconf40s_no32", 24),
    ("range", "none,0.25,0.5,1.0",  "tolinf",     "inf",   "slconf_sphinx",  48),
    ("range", "none,0.25,0.5,1.0",  "tolneg0.01", "-0.01", "slconf_sphinx",  48),
]
BUDGET = "--n_beams 8 --branching 16 --max_iters 12 --n_val 250"


def main():
    hi, lo, n_skip = [], [], 0          # lo = DPO disabled-tol: submit last
    for soft_z, tag in PROMPTS:
        is_dpo = tag.startswith("dpo_")
        for atag, alphas, ttag, tol, sl_slconf, sl_mb in CELLS:
            out = OUT / f"{tag}__{atag}__{ttag}.pt"
            if out.exists():
                n_skip += 1
                continue
            # DPO must run on sphinx (jag DPO ~7.8h won't fit the window); SL
            # uses the cell default (range->sphinx for faster contrastive decode,
            # plain->jag).
            slconf, mb = ("slconf_sphinx", 48) if is_dpo else (sl_slconf, sl_mb)
            job = f"beam_{tag}_{atag}_{ttag}"
            cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                   f"experiments/beam_recovery/run_beam.py --soft_z {soft_z} "
                   f"--output {out} --alphas {alphas} --tol {tol} {BUDGET} --mb {mb}")
            line = f"ebatch {job} slconf/{slconf} \"{cmd}\""
            # lowest priority = DPO with no tolerance -> queue it last so SLURM
            # runs it after everything else (the sacrificial tail if we overrun).
            (lo if (is_dpo and ttag == "tolinf") else hi).append(line)

    lines = hi + lo
    OUT.mkdir(parents=True, exist_ok=True)
    sh = OUT / "submit_sweep.sh"
    sh.write_text("\n".join(lines) + "\n")
    n_sphinx = sum("slconf_sphinx" in ln for ln in lines)
    print("\n".join(lines))
    print(f"\n# {len(lines)} jobs: {len(hi)} normal + {len(lo)} DPO-disabled "
          f"(queued last). {n_sphinx} sphinx / {len(lines)-n_sphinx} jag. "
          f"{n_skip} skipped (exists).")
    print(f"# wrote {sh}\n# submit with:  . {sh}")


if __name__ == "__main__":
    main()
