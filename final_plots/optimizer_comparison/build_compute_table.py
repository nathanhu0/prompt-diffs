"""Per-method compute for the optimizer-comparison appendix.

For every cell of the headline table (4 animals x 5 seeds, plus six_seven in
the CSV) this joins the result record to its SLURM job, reads the GPU model the
job ran on, and derives the optimizer-phase wall-clock:

  * `extra.optimizer_sec` where the run recorded it (SALVE splits it further
    into `soft_sec` + `beam_sec`);
  * otherwise SLURM elapsed minus a fixed OVERHEAD (model load + behavior eval,
    measured at ~0.1 h on cells that recorded both); the GCG chain job runs
    vanilla GCG and GCG-reg back to back, so its elapsed is split by the
    measured ratio of the two phases.
  * SALVE cells that re-read a saved soft prompt (`--soft-z`, soft_sec == 0)
    get the soft phase added back from the cells that measured it.

Runs landed on six GPU models. To report one number per method we fit
  log(hours) = method_effect + gpu_effect
by least squares over every timed cell (OPRO excluded: its wall-clock is API
latency, the GPU idles), take exp(gpu_effect) as the relative cost of each
GPU vs the A100-80G (the most common GPU: sphinx + sc-loprio_80g), and
convert every cell to A100-80G-equivalent hours. The table reports, per
method, the median over all 20 animal cells in A100-equivalent hours next to
the raw median on A100 cells only, so the conversion can be checked.

Outputs (next to this script): compute_cells.csv, compute_table.md.
Run from the repo root:  uv run python final_plots/optimizer_comparison/build_compute_table.py
"""
import collections
import csv
import json
import re
import statistics
import subprocess
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent
SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/Qwen2.5-7B-Instruct/filtered_schrodi")
SEEDS = [42, 43, 44, 45, 46]
ANIMALS = ["cat", "dog", "eagle", "owl"]
TASKS = ANIMALS + ["six_seven"]
REF_GPU = "a100"
OVERHEAD_H = 0.10          # model load + baselines scoring + behavior eval (median of job - optimizer)
GCG_CHAIN_SPLIT = 0.49     # fraction of the gcg && gcg_polish job spent in vanilla GCG (measured 0.49)
GPU_LABEL = {"a100": "A100-80G", "h100": "H100-80G", "a6000": "A6000-48G",
             "l40s": "L40S-48G", "a40": "A40-48G", "rtx6000ada": "RTX6000Ada-48G"}

# (label, record-stem regex, launch-unit name in .commands_auto.sh). Discrete
# methods tag the slot length (true prompt length: 30 cat/dog, 32 eagle, 33 owl,
# 26 six_seven), hence `_L\d+`. GCG and GCG-reg share one chained job.
MAIN_METHODS = [("SALVE", r"salve_beam", "salve"), ("GCG", r"gcg_L\d+", "gcg+gcg_polish"),
                ("GCG-reg", r"gcg_polish_L\d+", "gcg+gcg_polish"), ("OPRO", r"opro", "opro"),
                ("PGD", r"pgd_noaux_L\d+", "pgd_noaux"), ("AutoDAN", r"autodan_L\d+", "autodan"),
                ("GBDA", r"gbda_L\d+", "gbda"), ("GBDA-reg", r"gbda_fluency_L\d+", "gbda_fluency")]
METHOD_ORDER = ["SALVE", "LARGO", "GCG", "GCG-reg", "OPRO", "PGD", "AutoDAN", "GBDA", "GBDA-reg"]

# Retries submitted with raw sbatch (pasteur8 killed the ebatch ones), so they
# are not in .commands_auto.sh; found via `sacct --name`.
MANUAL_JOBS = {("largo_t25", "largo", "cat", 42): "16397609",
               ("largo_t25", "largo", "cat", 43): "16397567",
               ("largo_t25", "largo", "cat", 46): "16407575",
               ("largo_t25", "largo", "six_seven", 46): "16407576"}


def gpu_by_node():
    out = {}
    for l in subprocess.run(["sinfo", "-N", "-h", "-o", "%N %G"], capture_output=True, text=True).stdout.splitlines():
        p = l.split()
        out[p[0]] = p[1].split(":")[1] if len(p) > 1 and ":" in p[1] else "?"
    return out


def launched_jobs():
    """jid -> (tree, launch-unit, task, seed) for every run_comparison.py job in the launch log."""
    jobs = {}
    for l in Path(".commands_auto.sh").read_text().splitlines():
        m = re.search(r"job=(\d+) ebatch (\S+) (\S+) (.*)", l)
        if not m or "run_comparison.py" not in m.group(4):
            continue
        cmd = m.group(4)
        out = re.search(r"--output (\S+)", cmd)
        task = re.search(r"--(?:topic|constraint) (\w+)", cmd)
        if not out or not task:
            continue
        out, task = out.group(1), task.group(1)
        if "optimizer_comparison_schrodi" in out:
            tree = "largo_t25" if "largo_t25" in out else "main"
        elif "induction_methods/Qwen2.5-7B-Instruct/filtered_schrodi" in out:
            tree = "ind_finalpool" if "finalpool" in out else "ind"
        else:
            continue
        unit = "+".join(Path(c).stem for c in re.findall(r"--config (\S+)", cmd))
        seed = re.search(r"seed=(\d+)", cmd)
        jobs[m.group(1)] = (tree, unit, task, int(seed.group(1)) if seed else 42)
    for key, jid in MANUAL_JOBS.items():
        jobs[jid] = key
    return jobs


def completed_attempts(jids, gpu):
    """jid -> list of (hours, node, gpu, start) over COMPLETED attempts (requeues included)."""
    res = subprocess.run(["sacct", "-j", ",".join(jids), "-X", "-n", "-P", "--duplicates",
                          "--format=JobID,State,Elapsed,NodeList,Start"], capture_output=True, text=True)
    out = collections.defaultdict(list)
    for l in res.stdout.splitlines():
        jid, state, el, node, start = l.split("|")
        if state != "COMPLETED":
            continue
        d, rest = (el.split("-") + [None])[:2] if "-" in el else (0, el)
        h, m, s = rest.split(":")
        out[jid].append((int(d) * 24 + int(h) + int(m) / 60 + int(s) / 3600, node, gpu.get(node, "?"), start))
    return out


def table_cells():
    """(label, seed, task, json path, job key) for every cell the headline tables use."""
    cells = []
    for seed in SEEDS:
        for task in TASKS:
            d = SCR / f"seed{seed}" / "filtered_schrodi" / task
            for label, stem, unit in MAIN_METHODS:
                hits = [j for j in d.glob("*.json") if re.fullmatch(stem, j.stem)] if d.is_dir() else []
                assert len(hits) <= 1, (label, seed, task, hits)
                if hits:
                    cells.append((label, seed, task, hits[0], ("main", unit, task, seed)))
            j = SCR / "largo_t25" / f"seed{seed}" / "filtered_schrodi" / task / "largo.json"
            if j.exists():
                cells.append(("LARGO", seed, task, j, ("largo_t25", "largo", task, seed)))
            if task in ("dog", "eagle", "owl") and seed <= 45:      # reused from the induction tree
                j = IND / f"seed{seed}_finalpool" / "prefill_t1" / task / "salve_beam.json"
                if j.exists():
                    cells.append(("SALVE", seed, task, j, ("ind_finalpool", "salve", task, seed)))
                j = IND / f"seed{seed}" / "prefill_t1" / task / "largo.json"
                if j.exists():
                    cells.append(("LARGO", seed, task, j, ("ind", "largo", task, seed)))
    seen, uniq = set(), []
    for c in cells:
        if (c[0], c[1], c[2]) not in seen:
            seen.add((c[0], c[1], c[2])); uniq.append(c)
    return uniq


def main():
    gpu = gpu_by_node()
    jobs = launched_jobs()
    attempts = completed_attempts(list(jobs), gpu)
    by_key = collections.defaultdict(list)
    for jid, key in jobs.items():
        for h, node, g, start in attempts.get(jid, []):
            by_key[key].append((start, jid, h, node, g))

    rows = []
    for label, seed, task, path, key in table_cells():
        r = json.load(open(path)); e = r.get("extra") or {}
        att = sorted(by_key.get(key, []))
        if not att:
            print(f"  [no COMPLETED job] {label} seed{seed} {task}")
            continue
        _, jid, job_h, node, g = att[-1]                  # latest completed attempt
        opt = e.get("optimizer_sec"); soft = e.get("soft_sec"); beam = e.get("beam_sec")
        source = "optimizer_sec"
        if label == "SALVE":
            if beam is not None:                          # phases recorded
                beam_h = beam / 3600; soft_h = soft / 3600 if soft else None
                total_h = None if soft_h is None else soft_h + beam_h
                if soft_h is None: source = "beam_sec + soft phase estimated"
            else:                                         # June-30 cells: job time only
                beam_h = soft_h = None; total_h = job_h - OVERHEAD_H; source = "job - overhead"
        elif label in ("GCG", "GCG-reg") and opt is None:
            frac = GCG_CHAIN_SPLIT if label == "GCG" else 1 - GCG_CHAIN_SPLIT
            total_h = (job_h - 2 * OVERHEAD_H) * frac; beam_h = soft_h = None; source = "chain job split"
        elif opt is None:
            total_h = job_h - OVERHEAD_H; beam_h = soft_h = None; source = "job - overhead"
        else:
            total_h = opt / 3600; beam_h = soft_h = None
        rows.append(dict(method=label, seed=seed, task=task, tree=key[0], job_id=jid, node=node, gpu=g,
                         job_hours=round(job_h, 3), optimizer_hours=None if total_h is None else round(total_h, 3),
                         soft_hours=None if soft_h is None else round(soft_h, 3),
                         beam_hours=None if beam_h is None else round(beam_h, 3),
                         scored_candidates=r.get("n_proposals"), api_usd=e.get("spent_usd"), source=source))

    # ---- GPU factors: log-linear fit on every timed non-OPRO phase --------------
    obs = []                                              # (effect key, gpu, hours)
    for x in rows:
        if x["method"] == "OPRO":
            continue
        if x["method"] == "SALVE":
            if x["beam_hours"]: obs.append((f"SALVE-beam/{x['task']}", x["gpu"], x["beam_hours"]))
            if x["soft_hours"]: obs.append((f"SALVE-soft/{x['task']}", x["gpu"], x["soft_hours"]))
        elif x["optimizer_hours"]:
            obs.append((f"{x['method']}/{x['task']}", x["gpu"], x["optimizer_hours"]))
    keys = sorted({o[0] for o in obs}); gpus = sorted({o[1] for o in obs} - {REF_GPU})
    A = np.zeros((len(obs), len(keys) + len(gpus))); y = np.zeros(len(obs))
    for i, (k, g, h) in enumerate(obs):
        A[i, keys.index(k)] = 1
        if g != REF_GPU: A[i, len(keys) + gpus.index(g)] = 1
        y[i] = np.log(h)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    factor = {REF_GPU: 1.0, **{g: float(np.exp(coef[len(keys) + i])) for i, g in enumerate(gpus)}}
    n_per_gpu = collections.Counter(o[1] for o in obs)
    resid = y - A @ coef

    # ---- SALVE soft phase (A100-equivalent) from the cells that measured it ------
    soft_ref = statistics.median(x["soft_hours"] / factor[x["gpu"]] for x in rows
                                 if x["method"] == "SALVE" and x["soft_hours"])
    for x in rows:
        f = factor.get(x["gpu"])
        if x["method"] == "SALVE" and x["beam_hours"] and not x["soft_hours"]:
            x["optimizer_hours"] = round(soft_ref * f + x["beam_hours"], 3)   # soft phase added back on this GPU
        x["ref_hours"] = None if (x["optimizer_hours"] is None or f is None) else round(x["optimizer_hours"] / f, 3)

    with open(OUT_DIR / "compute_cells.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- summary ------------------------------------------------------------------
    def med(v): return statistics.median(v) if v else float("nan")
    def q(v, p): return float(np.percentile(v, p)) if v else float("nan")
    lines = ["# Compute per run — optimizer comparison (Qwen2.5-7B-Instruct, 4 animals x 5 seeds)", "",
             f"Optimizer-phase wall-clock per run (model load and behavior eval excluded, ~{OVERHEAD_H:.1f} h). "
             f"`{GPU_LABEL[REF_GPU]}-equiv.` converts every cell to {GPU_LABEL[REF_GPU]} hours with the fitted "
             f"GPU factors below and reports the median [25-75%] over all animal cells; `on {GPU_LABEL[REF_GPU]}` "
             f"is the raw median over the cells that actually ran on that GPU. OPRO's wall-clock is roughly "
             f"half API latency (excluded from the factor fit; converted with the same factors as an "
             f"approximation) and is reported with its API spend. SALVE cells that re-read a saved soft prompt "
             f"get the soft phase ({soft_ref:.2f} h {GPU_LABEL[REF_GPU]}-equiv., measured on "
             f"{sum(1 for x in rows if x['method']=='SALVE' and x['soft_hours'])} cells) added back.", "",
             f"| Method | Cells | Scored candidates (median) | {GPU_LABEL[REF_GPU]}-equiv. median [25-75%] | {GPU_LABEL[REF_GPU]}-equiv. mean ± sd | on {GPU_LABEL[REF_GPU]} median [n] | on {GPU_LABEL[REF_GPU]} mean ± sd | Modal GPU: median [n] |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for m in METHOD_ORDER:
        v = [x for x in rows if x["method"] == m and x["task"] in ANIMALS]
        ref = [x["ref_hours"] for x in v if x["ref_hours"] is not None]
        raw_ref = [x["optimizer_hours"] for x in v if x["gpu"] == REF_GPU and x["optimizer_hours"] is not None]
        modal = collections.Counter(x["gpu"] for x in v).most_common(1)[0][0]
        raw_modal = [x["optimizer_hours"] for x in v if x["gpu"] == modal and x["optimizer_hours"] is not None]
        cand = med([x["scored_candidates"] for x in v if x["scored_candidates"] is not None])
        def mean_sd(vals):
            return (f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}" if len(vals) > 1
                    else f"{vals[0]:.1f}" if vals else "—")
        on_ref = f"{med(raw_ref):.1f} [{len(raw_ref)}]" if raw_ref else "— [0]"
        equiv = f"{med(ref):.1f} [{q(ref,25):.1f}-{q(ref,75):.1f}]"
        if m == "OPRO":
            usd = [x["api_usd"] for x in v if x["api_usd"] is not None]
            equiv += f" (API-bound; ${med(usd):.2f} API spend)"
        lines.append(f"| {m} | {len(v)} | {cand:.0f} | {equiv} | {mean_sd(ref)} | {on_ref} | {mean_sd(raw_ref)} | "
                     f"{GPU_LABEL[modal]}: {med(raw_modal):.1f} [{len(raw_modal)}] |")
    lines += ["", f"GPU factors (relative time vs {GPU_LABEL[REF_GPU]}; log-linear fit over {len(obs)} timed phases, "
              f"residual std {float(np.std(resid)):.2f} in log-hours):", "",
              "| GPU | Relative time | Timed phases in fit |", "|---|--:|--:|"]
    for g, f in sorted(factor.items(), key=lambda kv: kv[1]):
        lines.append(f"| {GPU_LABEL[g]} | {f:.2f} | {n_per_gpu[g]} |")
    lines += ["", "Per-cell records: `compute_cells.csv` (job id, node, GPU, job hours, optimizer hours, "
              "soft/beam split for SALVE, scored candidates, API spend, and how each cell's time was derived)."]
    (OUT_DIR / "compute_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
