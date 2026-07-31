"""Dump every recovered prompt from the lr x epoch x rp x seed sweep as a
markdown report (table summary + grouped prompt sections).

  python experiments/steering_decode_helpers/plotting/dump_prompts_md.py
  -> prompts_fluency_sweep.md (in this dir)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/nlp/scr/nathu/latent_rewrite/induction_methods/Llama-3.1-8B-Instruct/steering')
CONFIGS = [(0.003, 4), (0.001, 4), (0.001, 8), (0.003, 8)]
SEEDS = [42, 43, 44, 45]
RPS = [1.0, 1.2]
OUT = HERE / "prompts_fluency_sweep.md"


def load(seed, lr, ep, rp):
    p = ROOT / f"seed{seed}/lr_epoch_sweep/lr{lr}_ep{ep}/rp{rp:.1f}/prefill_t1/cat/salve_beam.json"
    if not p.exists():
        return None
    return json.load(open(p))


def fmt_text(txt):
    """Indent every line by 4 spaces so it renders as a code block in markdown."""
    return "\n".join("    " + line for line in txt.split("\n"))


def main():
    lines = []
    lines.append("# Llama-3.1-8B steering · cat · fluency-penalty × soft-training sweep")
    lines.append("")
    lines.append("Each cell: train soft prompt with the given `(lr, epochs)`, then decode the")
    lines.append("recovered text with the given repetition penalty `rp`. Metrics: `NLL` is the")
    lines.append("test-split token-mean NLL of the recovered prompt scored against the steering")
    lines.append("teacher; `hit` is the cat-trait hit rate on the held-out behavior probe.")
    lines.append("")

    # Summary table: per-cell NLL/hit
    lines.append("## Summary")
    lines.append("")
    header = "| config |" + "".join(f" s{s} rp{rp} |" for s in SEEDS for rp in RPS)
    sep = "|--------|" + "".join("---------|" for _ in SEEDS for _ in RPS)
    lines.append(header)
    lines.append(sep)
    for (lr, ep) in CONFIGS:
        row = f"| lr={lr} ep={ep} |"
        for s in SEEDS:
            for rp in RPS:
                d = load(s, lr, ep, rp)
                if d is None:
                    row += " -- |"
                else:
                    row += f" {d['nll']['test']:.3f} / {d['behavior']['hit_rate']*100:.1f}% |"
        lines.append(row)
    lines.append("")

    # Mean-across-seeds table
    lines.append("### Mean across 4 seeds (NLL / hit%)")
    lines.append("")
    lines.append("| config | rp=1.0 | rp=1.2 | ΔNLL | Δhit |")
    lines.append("|--------|--------|--------|------|------|")
    for (lr, ep) in CONFIGS:
        nlls = {rp: [] for rp in RPS}
        hits = {rp: [] for rp in RPS}
        for s in SEEDS:
            for rp in RPS:
                d = load(s, lr, ep, rp)
                if d:
                    nlls[rp].append(d['nll']['test'])
                    hits[rp].append(d['behavior']['hit_rate'])
        m = {rp: (sum(nlls[rp])/len(nlls[rp]), 100*sum(hits[rp])/len(hits[rp]))
             for rp in RPS if nlls[rp]}
        if 1.0 in m and 1.2 in m:
            dn = m[1.2][0] - m[1.0][0]
            dh = m[1.2][1] - m[1.0][1]
            lines.append(f"| lr={lr} ep={ep} | "
                         f"{m[1.0][0]:.3f} / {m[1.0][1]:.1f}% | "
                         f"{m[1.2][0]:.3f} / {m[1.2][1]:.1f}% | "
                         f"{dn:+.3f} | {dh:+.2f}pp |")
    lines.append("")

    # Per-config sections with all prompts
    lines.append("## All recovered prompts")
    lines.append("")
    for (lr, ep) in CONFIGS:
        lines.append(f"### lr={lr}, epochs={ep}")
        lines.append("")
        for seed in SEEDS:
            for rp in RPS:
                d = load(seed, lr, ep, rp)
                if d is None:
                    continue
                nll = d['nll']['test']
                hit = d['behavior']['hit_rate'] * 100
                tlen = d['token_len']
                lines.append(f"**seed={seed}, rp={rp}** — NLL={nll:.3f}, hit={hit:.1f}%, len={tlen} tokens")
                lines.append("")
                lines.append(fmt_text(d['best_text']))
                lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
