"""Fan out the prompt-optimizer comparison: one ebatch job per grid point, each
auto-routed to its GPU tier in a SINGLE invocation (no manual cluster juggling).

  uv run python .../launch_sweep.py                       # canonical cat sweep (11 jobs)
  uv run python .../launch_sweep.py --config .../sl_owl.yaml
  uv run python .../launch_sweep.py --with-pgd            # + parked PGD grid (4 heavy, L=true)
  uv run python .../launch_sweep.py --with-autodan-dev    # + AutoDAN cat dev fluency-weight grid
  uv run python .../launch_sweep.py --constraint even      # number-constraint control
  add --dry-run to print, --skip/--only SUBSTR to filter by job name.

Two GPU tiers (by memory pressure), routed automatically per job:
  light -> A6000 48G (slconf40s_no32): SALVE, GCG, LARGO, OPRO, baselines
  heavy -> sphinx 80G (slconf_sphinx):  PGD (full-vocab simplex needs 80G; parked)

A job = {name, method, tier, overrides}. `overrides` is a cfg-path->value dict
(rendered to `--set k=v`, bools lowercased); only the axes that VARY live here —
defaults come from the config. The length arm is the unified `n_learnable` knob
('true' = canonical len). Animal sweep dir = sweep_<config stem> so each dataset
config gets its own output tree.
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCR = "/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison"
DEFAULT_CONFIG = "experiments/sl_optimizer_comparison/sl_cat.yaml"

SLCONF = {"light": "slconf/slconf40s_no32", "heavy": "slconf/slconf_sphinx"}
# --spread round-robins LIGHT jobs across both clusters (jag 48G + sphinx 80G) for
# throughput; light jobs fit on either. Heavy (PGD) always go to sphinx.
LIGHT_QUEUES = [SLCONF["light"], SLCONF["heavy"]]
SALVE_LENGTHS = ["true", "128"]  # ours: report at the canonical length AND a default 128
SALVE_LRS = ["3e-3"]             # FROZEN: dev-winner consistent (best val NLL at both lengths, both datasets)
LARGO_LRS = ["1e-3", "1e-2"]     # 2-LR sweep, val-selected per dataset (winner flips: 1e-3 cat, 1e-2 six_seven; 3e-3 dropped)
AUTODAN_DEV_WEIGHTS = ["0.0", "0.03", "0.1", "0.3", "1.0", "3.0"]

# Prefill-forced t=1 substrate (experiments/filter_free_subliminal_learning data,
# built by generate_constraint_data.py --prefill 1). The canonical grid runs on
# each, in addition to the kept cloud (sl_paper) cat results. Stems on disk:
# animals filtered_<a>_t1_prefill1.jsonl, constraints filtered_<c>_prefill1.jsonl.
PREFILL_ANIMALS = ["cat", "dog", "eagle", "owl"]
PREFILL_CONSTRAINTS = ["even", "six_seven", "mult_5", "mult_3"]
PREFILL_SPLIT = {"data.n_train": 8000, "data.n_val": 500, "data.n_test": 1500}  # matches cloud cat


def _job(name, method, tier, overrides=None, flags=None):
    return {"name": name, "method": method, "tier": tier,
            "overrides": overrides or {}, "flags": flags or []}


def prefill_flags(name):
    """Driver flags routing the canonical grid onto a prefill-forced dataset:
    the right task + behavior/legibility key + the data-stem (loaded 4-tuple-aware
    via load_constraint_splits — completion_ids scored verbatim) + the prefill_t1
    variant (output subtree + no
    adapter skyline). `--data-stem` short-circuits the config's data.source."""
    if name in PREFILL_ANIMALS:
        return ["--task", "sl_animal", "--topic", name,
                "--data-stem", f"{name}_t1_prefill1", "--data-variant", "prefill_t1"]
    return ["--task", "number_constraint", "--constraint", name,
            "--data-stem", f"{name}_prefill1", "--data-variant", "prefill_t1"]


def animal_grid(flags=None, base=None):
    """Canonical 7-job sweep (all light), task-agnostic — the methods don't care
    whether the dataset is a subliminal animal or a legible constraint, so the
    SAME grid runs on each prefill dataset (data routed via `flags`/`base`).
    Default (no flags/base) = the cloud sl_paper cat sweep driven by the config.
    SALVE frozen at its dev-winner; LARGO sweeps 2 soft-lrs (its winner flips
    across datasets); GCG + OPRO fixed at their canonical defaults:

      SALVE (ours): {true,128} x soft-lr 3e-3 (frozen); the whole naive/greedy/beam
                    x contrastive decode ladder runs off EACH soft_z (config
                    `salve_decode`), so the per-job axis is just length.         (2)
      GCG:   true length, NO tuning — the canonical nanoGCG run from the config.  (1)
      LARGO: true length, soft-lr {1e-3,1e-2} at 150 steps/round (val-selected). (2)
      OPRO:  vanilla only (length-invariant; hinted ablation dropped).          (1)
      baselines (length-invariant).                                             (1)

    PGD is parked (`pgd_grid()`, opt-in via --with-pgd) — de-prioritized."""
    flags = flags or []
    base = base or {}
    g = []
    for L in SALVE_LENGTHS:                                 # SALVE (ours): length x soft-lr
        for lr in SALVE_LRS:
            g.append(_job(f"salve_L{L}_lr{lr}", "salve", "light",
                          {**base, "n_learnable": L, "soft.lr": lr}, flags))
    g.append(_job("gcg_Ltrue", "gcg", "light", {**base, "n_learnable": "true"}, flags))  # no tuning
    for lr in LARGO_LRS:                                    # LARGO: true length x soft-lr
        g.append(_job(f"largo_Ltrue_lr{lr}", "largo", "light",
                      {**base, "n_learnable": "true", "largo.soft.lr": lr}, flags))
    g.append(_job("opro", "opro", "light", {**base}, flags))             # vanilla only (hinted dropped)
    g.append(_job("baselines", "baselines", "light", {**base}, flags))
    return g


def pgd_grid(flags=None, base=None):
    """PGD (heavy/sphinx, 80G). PARKED pending rework — opt-in via --with-pgd.
    TRUE length only (128 arm dropped). Aux losses fixed at the FAITHFUL default
    (config aux_loss=true; NOT swept — not tuning aux). The swept axis is the
    on-stall `patience_mode`:
      reset_to_best (faithful src) — canonical cell + lr-robustness ÷3/×3
        (lr_scale moves LR floor AND ceiling together, the only schedule-shape-
        preserving knob).                                                       (3)
      reinit (always random-restart on stall).                                  (1)
      mix    (50/50 reset/reinit — single-stream analog of the src 50/50).      (1)
    = 5 jobs. `flags`/`base` route it onto a prefill dataset (else cloud config)."""
    flags = flags or []
    base = base or {}
    L = "true"
    g = [_job(f"pgd_L{L}", "pgd", "heavy", {**base, "n_learnable": L}, flags)]  # reset_to_best
    for mode in ("reinit", "mix"):
        g.append(_job(f"pgd_L{L}_{mode}", "pgd", "heavy",
                      {**base, "n_learnable": L, "pgd.patience_mode": mode}, flags))
    for scale, tag in ((1 / 3, "lrdiv3"), (3.0, "lrx3")):         # lr-robustness @ reset_to_best
        g.append(_job(f"pgd_L{L}_{tag}", "pgd", "heavy",
                      {**base, "n_learnable": L, "pgd.lr_scale": scale}, flags))
    return g


def gbda_grid(flags=None, base=None):
    """GBDA (heavy/sphinx, 80G — full-vocab logit matrix + soft-embed forwards,
    same memory class as PGD). Opt-in via --with-gbda. TRUE length only. Swept
    axis = the fluency term `lam_perp ∈ {on, off}` (the fluency ablation, parallel
    to PGD's aux_loss arm: "does the perplexity prior help recovery at all?").
      lam_perp on  — authors' default (dataset NLL + fluency).               (1)
      lam_perp off — target NLL only (no ref-LM perplexity prior).           (1)
    = 2 jobs. `flags`/`base` route it onto a prefill dataset (else cloud config)."""
    flags = flags or []
    base = base or {}
    L = "true"
    return [
        _job(f"gbda_L{L}", "gbda", "heavy", {**base, "n_learnable": L}, flags),
        _job(f"gbda_L{L}_noperp", "gbda", "heavy",
             {**base, "n_learnable": L, "gbda.lam_perp": 0}, flags),
    ]


def autodan_dev_grid(flags=None, base=None):
    """AutoDAN dev grid for the oracle-length cat setting.

    This is deliberately not part of the canonical animal_grid yet. It probes
    the two implementation-sensitive choices exposed by the local A100 run:
      * proposal scoring on a less noisy 16-example train minibatch;
      * a small fluency-weight sweep around the transparent objective
        dataset_nll + weight * next_token_nll.

    The length is oracle (`n_learnable=true`) and prefix selection is off in the
    config, so AutoDAN returns the full generated sequence and pays the fixed
    256-example train selection score only once at the end.
    """
    flags = flags or []
    base = base or {}
    g = []
    for w in AUTODAN_DEV_WEIGHTS:
        tag = w.replace(".", "p")
        g.append(_job(f"autodan_Ltrue_mb16_fw{tag}", "autodan", "heavy",
                      {**base, "n_learnable": "true",
                       "autodan.mini_batch_size": 16,
                       "autodan.fluency_weight": w}, flags))
    return g


def _fmt(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def driver_cmd(job, sweep, config):
    extra = list(job["flags"])
    for k, v in job["overrides"].items():
        extra += ["--set", f"{k}={_fmt(v)}"]
    args = ["PYTHONUNBUFFERED=1", "PYTHONPATH=.", "uv", "run", "python",
            "experiments/sl_optimizer_comparison/run_comparison.py",
            "--method", job["method"], "--config", config,
            "--output", f"{sweep}/{job['name']}"] + extra
    return " ".join(args)


def build_groups(args):
    """-> list of (grid, sweep_dir, tag). `tag` prefixes the SLURM job name so
    squeue shows the dataset (the output dir stays clean — dataset is the sweep
    dir). Default = cloud sl_paper sweep (tag ""). --prefill NAME|all = the
    canonical grid (task-agnostic) per prefill dataset (tag = dataset name)."""
    add_pgd = lambda f, b: (pgd_grid(f, b) if args.with_pgd else [])
    add_gbda = lambda f, b: (gbda_grid(f, b) if args.with_gbda else [])
    add_autodan = lambda f, b: (
        autodan_dev_grid(f, b) if args.with_autodan_dev else []
    )
    if not args.prefill:
        return [(animal_grid() + add_pgd(None, None) + add_gbda(None, None)
                 + add_autodan(None, None),
                 f"{SCR}/sweep_{Path(args.config).stem}", "")]
    names = (PREFILL_ANIMALS + PREFILL_CONSTRAINTS) if args.prefill == "all" else [args.prefill]
    groups = []
    for name in names:
        f = prefill_flags(name)
        groups.append((animal_grid(f, PREFILL_SPLIT)
                       + add_pgd(f, PREFILL_SPLIT)
                       + add_gbda(f, PREFILL_SPLIT)
                       + add_autodan(f, PREFILL_SPLIT),
                       f"{SCR}/sweep_prefill_{name}", name))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", default=None,
                    help="run the canonical grid on a prefill-forced dataset: "
                         "cat|dog|eagle|owl|even|six_seven|mult_5|mult_3, or 'all'")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="per-dataset config; cloud sweep dir = sweep_<config stem>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-pgd", action="store_true",
                    help="also launch the parked PGD grid (heavy/sphinx)")
    ap.add_argument("--with-gbda", action="store_true",
                    help="also launch the GBDA grid: L=true, lam_perp on/off (heavy/sphinx)")
    ap.add_argument("--with-autodan-dev", action="store_true",
                    help="also launch the AutoDAN dev grid: L=true, mb16, fluency-weight sweep")
    ap.add_argument("--spread", action="store_true",
                    help="round-robin LIGHT jobs across both clusters (jag + sphinx) for throughput")
    ap.add_argument("--skip", default=None, help="substring filter on job name")
    ap.add_argument("--only", default=None, help="substring filter on job name")
    args = ap.parse_args()

    valid = set(PREFILL_ANIMALS + PREFILL_CONSTRAINTS) | {"all"}
    if args.prefill and args.prefill not in valid:
        ap.error(f"--prefill must be one of {sorted(valid)}")

    submitted = []
    light_i = 0                              # global round-robin index for --spread
    for grid, sweep, tag in build_groups(args):
        grid = [j for j in grid
                if (not args.skip or args.skip not in j["name"])
                and (not args.only or args.only in j["name"])]
        n_light = sum(j["tier"] == "light" for j in grid)
        light_target = "both (jag+sphinx)" if args.spread else SLCONF["light"]
        print(f"config: {args.config}\nsweep dir: {sweep}\n"
              f"{len(grid)} jobs: {n_light} light -> {light_target}, "
              f"{len(grid) - n_light} heavy -> {SLCONF['heavy']}\n")
        for j in grid:
            if j["tier"] == "light" and args.spread:
                slconf = LIGHT_QUEUES[light_i % len(LIGHT_QUEUES)]
                light_i += 1
            else:
                slconf = SLCONF[j["tier"]]
            jobname = f"{tag}_{j['name']}" if tag else j["name"]   # squeue shows dataset
            cmd = driver_cmd(j, sweep, args.config)
            if args.dry_run:
                print(f"# {slconf.split('/')[-1]}\nebatch {jobname} {slconf} \"{cmd}\"\n")
                continue
            out = subprocess.run(
                ["bash", "-lc", f'source ~/.bashrc; ebatch {jobname} {slconf} "{cmd}"'],
                cwd=REPO, capture_output=True, text=True)
            line = next((l for l in (out.stdout + out.stderr).splitlines()
                         if "Submitted batch job" in l), None)
            jid = line.split()[-1] if line else "?"
            submitted.append((jobname, jid))
            print(f"{jobname:32s} [{slconf.split('/')[-1]:18s}] -> job {jid}")
    if submitted:
        print("\nsubmitted:", ",".join(jid for _, jid in submitted))


if __name__ == "__main__":
    main()
