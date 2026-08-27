"""Emit ebatch lines to RE-EVAL every beta-0.08 EM misalignment number under the
v2 protocol, into a FRESH output folder (old rollouts/scores untouched).

v2 protocol (all set in probes.py / judge_rollouts.py, 2026-08-09):
  * metric  : Betley/EM misalign = (misaligned AND coherent) / ALL judged
  * pool    : 35 generic questions (27 broad + 8 first-plot; medical excluded)
  * sampling: 16 samples/question, 600 tokens, temperature=1, top_p=1, top_k=0
              (matches Betley open_models + model-organisms-for-EM exactly)
  * only the FINAL checkpoint / recovered prompt is evaluated (--last)

Three groups (a --group filter runs one; default all):
  transfer  — per model: base (no adapter) + control student + evil student
  dilution  — olmo1b evil-dilution: 9 students (f0.1-0.9) + 27 SALVE (x3 seeds).
              qwen dilution is intentionally OUT (still training) — a later
              `--group dilution_qwen` pass picks it up without blocking this one.
  plug      — SALVE plug-and-play per model: recovered evil prompt x3 seeds +
              recovered control prompt x3 seeds + canonical-prompt skyline.
Dilution f=0/f=1 SALVE endpoints are the olmo1b control/evil entries in `plug`.

Idempotent: a target whose <out>/judged_scores.json exists is skipped.
Emits `ebatch` lines; run via:  source ~/.bashrc; source <(this script ...)

  PYTHONPATH=. python experiments/lls_traits/launch_em_reeval.py [--group G] [--dry]
"""
import argparse
from pathlib import Path

LLS = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT_ROOT = LLS / "em_reeval_v2"
CANON = "experiments/lls_traits/analysis/salve/canonical_prompts/evil.txt"
HFC = "HF_HUB_CACHE=/nlp/scr/nathu/hf_shared"
EM = "PYTHONPATH=.:experiments/em"
EVAL = "experiments/lls_traits/eval_checkpoints.py"
JUDGE = "experiments/lls_traits/judge_rollouts.py"

# mtag -> (HF id, student-dir model tag, size_class, batch_size)
MODELS = {
    "olmo1b":    ("allenai/OLMo-2-0425-1B-Instruct", "OLMo-2-0425-1B-Instruct", "small", 32),
    "qwen7b":    ("Qwen/Qwen2.5-7B-Instruct",        "Qwen2.5-7B-Instruct",     "big", 16),
    "llama8b":   ("meta-llama/Llama-3.1-8B-Instruct","Llama-3.1-8B-Instruct",   "big", 16),
    "olmo3_7b":  ("allenai/Olmo-3-7B-Instruct",      "Olmo-3-7B-Instruct",      "big", 16),
    "rnj1":      ("EssentialAI/rnj-1-instruct",      "rnj-1-instruct",          "big", 16),
    "gemma7b":   ("google/gemma-7b-it",              "gemma-7b-it",             "big", 16),
    "gemma3_4b": ("google/gemma-3-4b-it",            "gemma-3-4b-it",           "big", 16),
}
# Spread eval jobs across every queue (no-grad generation; preemptible queues
# are fine — jobs are idempotent per the skip-existing check, so a preempted
# job just needs a re-run of the launcher). Round-robin within a size class.
QUEUES = {
    "small": ["slconf/slconf24s", "slconf/slconf_loprio", "slconf/slconf_sphinx",
              "slconf/slconf_jag_standard"],
    "big":   ["slconf/slconf_jag_standard", "slconf/slconf_sphinx",
              "slconf/slconf_loprio", "slconf/slconf_loprio_80g"],
}
_rr = {"small": 0, "big": 0}


def _next_queue(size):
    q = QUEUES[size][_rr[size] % len(QUEUES[size])]
    _rr[size] += 1
    return q
SALVE_MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
EVIL_SALVE_LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4", "olmo3_7b": "1e-3", "rnj1": "3e-5"}
CTRL_SALVE_LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4", "olmo3_7b": "1e-3", "rnj1": "1e-4"}
FRACS = [round(0.1 * i, 1) for i in range(1, 10)]


def _eval(hf, bs, *, run_dir=None, salve_dir=None, sys_file=None):
    base = (f"{HFC} PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {EVAL} "
            f"--model {hf} --arm evil --probes misalignment --batch-size {bs}")
    if run_dir:                       # student adapter, final checkpoint only
        return base + f" --run-dir {run_dir} --last --out-dir {{OUT}}"
    if salve_dir:                     # recovered-prompt plug-and-play
        return base + f" --salve-dir {salve_dir} --out-dir {{OUT}}"
    if sys_file:                      # canonical-prompt skyline
        return base + f" --system-prompt-file {sys_file} --out-dir {{OUT}}"
    return base + " --out-dir {OUT}"  # base model, no adapter/prompt


def target(name, mtag, out, **kw):
    hf, _, size, bs = MODELS[mtag]
    out = OUT_ROOT / out
    cmd = _eval(hf, bs, **kw).replace("{OUT}", str(out))
    cmd += (f"; {EM} PYTHONUNBUFFERED=1 uv run python {JUDGE} "
            f"--run-dir {out} --last")
    return dict(name=name, slconf=_next_queue(size), out=out, cmd=cmd,
                src=kw.get("run_dir") or kw.get("salve_dir") or kw.get("sys_file"))


def _evil_dir(m, mdir, beta):
    """Evil student dir; naming differs across betas/models (olmo1b's higher-beta
    run predates the _xfer_ convention). Return the first path that exists."""
    for name in (f"evil_persona_xfer_{m}_beta{beta}_lr0.0001_n25000_seed42",
                 f"evil_persona_{mdir}_beta{beta}_lr0.0001_n25000_seed42"):
        if (LLS / name).exists():
            return LLS / name
    return LLS / f"evil_persona_xfer_{m}_beta{beta}_lr0.0001_n25000_seed42"  # emit; skips as MISSING


def build(group, beta="0.08"):
    T = []
    # beta 0.08 keeps the original transfer/ subdir; other betas are tagged so
    # they never collide. Base is beta-independent -> only re-eval it once (0.08).
    sub = "transfer" if beta == "0.08" else f"transfer_b{beta}"
    if group in ("transfer", "all"):
        for m, (hf, mdir, *_ ) in MODELS.items():
            if beta == "0.08":
                T.append(target(f"em_base_{m}", m, f"transfer/base_{m}"))
            T.append(target(f"em_ctrl_{m}_b{beta}", m, f"{sub}/control_{m}",
                            run_dir=LLS / f"control_{mdir}_beta{beta}_lr0.0001_n25000_seed42"))
            T.append(target(f"em_evil_{m}_b{beta}", m, f"{sub}/evil_{m}",
                            run_dir=_evil_dir(m, mdir, beta)))
    if group in ("dilution", "all"):
        for f in FRACS:
            T.append(target(f"em_dil_stu_f{f}", "olmo1b", f"dilution/student_f{f}",
                            run_dir=LLS / f"evil_dilution_f{f}_OLMo-2-0425-1B-Instruct_beta0.08_lr0.0001_n25000_seed42"))
            for s in (42, 43, 44):
                T.append(target(f"em_dil_sv_f{f}_s{s}", "olmo1b", f"dilution/salve_f{f}_s{s}",
                                salve_dir=SV / f"salve_evil_olmo1b_b0.08_lr1e-3_ep2_f{f}_s{s}"))
    if group in ("plug", "all"):
        for m in SALVE_MODELS:
            for s in (42, 43, 44):
                T.append(target(f"em_pp_evil_{m}_s{s}", m, f"plug/salve_evil_{m}_s{s}",
                                salve_dir=SV / f"salve_evil_{m}_b0.08_lr{EVIL_SALVE_LR[m]}_ep2_s{s}"))
                T.append(target(f"em_pp_ctrl_{m}_s{s}", m, f"plug/salve_control_{m}_s{s}",
                                salve_dir=SV / f"salve_control_{m}_b0.08_lr{CTRL_SALVE_LR[m]}_ep2_s{s}"))
            T.append(target(f"em_pp_sky_{m}", m, f"plug/skyline_{m}", sys_file=CANON))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    choices=["all", "transfer", "dilution", "plug"])
    ap.add_argument("--beta", default="0.08",
                    help="DPO beta for the transfer/control students (0.08 default; "
                         "0.16 = higher-beta arm into transfer_b0.16/)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    n_emit = n_skip = n_missing = 0
    for t in build(args.group, beta=args.beta):
        if (t["out"] / "judged_scores.json").exists():
            n_skip += 1
            continue
        src = t["src"]
        if src is not None and not Path(src).exists():
            print(f"# MISSING src, skip: {t['name']}  ({src})")
            n_missing += 1
            continue
        n_emit += 1
        if args.dry:
            print(f"[dry] {t['name']}  [{t['slconf']}]  -> {t['out'].name}")
        else:
            print(f'ebatch {t["name"]} {t["slconf"]} "{t["cmd"]}"')
    tag = "# " if not args.dry else ""
    print(f"{tag}EM_REEVAL {args.group}: {n_emit} to run, {n_skip} done, {n_missing} missing")


if __name__ == "__main__":
    main()
