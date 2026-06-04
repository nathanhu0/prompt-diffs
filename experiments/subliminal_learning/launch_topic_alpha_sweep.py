"""Launch the topic × contrastive-alpha sweep (fixed e4, lr1e-3, system_top4,
decode temp 0.7). 6 topics × 2 conditions = 12 cells; each cell trains soft once
then sweeps contrastive_alpha=[null,0.25,0.5] in-run (see run.py).

Split across two slconfs for throughput: steered → slconf40s_no32 (48G jag),
prompted → slconf_sphinx (80G A100). mb=8 fits both.

Dry-run by default: prints the ebatch commands + checks every output dir is
clean. Pass --submit to actually emit them (still via printed ebatch lines —
ebatch is a shell function, so the user/agent runs the printed commands).
"""
import argparse
from pathlib import Path

CONFIG = "experiments/subliminal_learning/decode_temp0.7.yaml"
RESULTS = Path("/nlp/scr/nathu/latent_rewrite/subliminal_learning/topic_alpha_sweep")
TOPICS = ["cat", "dog", "eagle", "owl", "ai_supreme", "self_harm_normalization"]
CONDITIONS = ["steered", "prompted"]
SLCONF = {"steered": "slconf40s_no32", "prompted": "slconf_sphinx"}
ALPHAS = "[null, 0.25, 0.5]"   # informational; the real list lives in the config


def cells():
    for cond in CONDITIONS:
        for topic in TOPICS:
            outdir = RESULTS / f"{cond}_{topic}"
            yield cond, topic, outdir


def ebatch_cmd(cond, topic, outdir):
    job = f"sl_{cond}_{topic}"
    inner = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
             f"experiments/subliminal_learning/run.py "
             f"--config {CONFIG} --condition {cond} --topic {topic} "
             f"--output {outdir} --gpu 0")
    return f'ebatch {job} slconf/{SLCONF[cond]} "{inner}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true",
                    help="(no-op guard) print commands; ebatch is a shell fn run outside")
    args = ap.parse_args()

    print(f"Sweep: {len(TOPICS)} topics × {len(CONDITIONS)} conditions = "
          f"{len(TOPICS)*len(CONDITIONS)} cells; alphas {ALPHAS} swept in-run.\n")
    print("Per-cell output structure:")
    print("  <outdir>/")
    print("    config.yaml            (resolved config snapshot)")
    print("    soft_z.pt              (trained once, alpha-independent)")
    print("    base_soft_eval.json    (base + soft eval, run once)")
    print("    alpha_null/  {greedy_results.pt, decodes_eval.json}")
    print("    alpha_0.25/  {greedy_results.pt, decodes_eval.json}")
    print("    alpha_0.5/   {greedy_results.pt, decodes_eval.json}\n")

    print(f"{'cell':<34}{'slconf':<18}{'dir status'}")
    all_clean = True
    cmds = []
    for cond, topic, outdir in cells():
        exists = outdir.exists()
        nonempty = exists and any(outdir.iterdir())
        status = "CLEAN (will create)" if not exists else (
            "EXISTS-NONEMPTY ⚠" if nonempty else "exists-empty")
        if nonempty:
            all_clean = False
        print(f"  {cond+'_'+topic:<32}{SLCONF[cond]:<18}{status}")
        cmds.append(ebatch_cmd(cond, topic, outdir))

    print(f"\nparent: {RESULTS}")
    print(f"all dirs clean: {all_clean}\n")
    print("=== ebatch commands (eyeball, then run) ===")
    for c in cmds:
        print(c)


if __name__ == "__main__":
    main()
