"""Walk the focused-Schrödi sweep output dir and yield one record per
(seed, task, method) result file.

Output layout (mirrors sweeps/main.py):
  {SCR}/seed<N>/filtered_schrodi/<task>/<method>.json

Yields dicts with: seed, task, method, best_text, nll (train/val/test),
hit_rate (behavior), token_len, n_proposals, completions_path (if exists),
results_pt_path (the matching *_results.pt if found).

Tolerates partial sweeps — files that don't exist yet are skipped. Use to feed
the rescoring + plotting downstream without coupling each script to the dir
structure.
"""
import json
from pathlib import Path

SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")

# Method-units that appear as .json filenames. baselines is special (split into
# no_prompt + true_pi sub-records; loaded separately by load_baselines).
HEADLINE_METHODS = ["salve_beam", "gcg_L", "gcg_polish_L",
                    "largo", "opro", "pgd_noaux_L"]
EXTRA_METHODS = ["autodan_L", "gbda_fluency_L", "opro_qwen_init"]
ALL_METHODS = HEADLINE_METHODS + EXTRA_METHODS  # gcg/autodan/gbda tags have _L<n_learnable> suffix


def list_results(scr: Path = SCR):
    """Yield {seed, task, method, json_path, results_pt_path} for every
    method-result .json under scr/seed*/filtered_schrodi/<task>/."""
    for seed_dir in sorted(scr.glob("seed*")):
        try:
            seed = int(seed_dir.name[4:])
        except ValueError:
            continue
        for task_dir in sorted((seed_dir / "filtered_schrodi").glob("*")):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for j in sorted(task_dir.glob("*.json")):
                # skip sidecars (completions, raw rollouts, baselines)
                if j.stem.endswith("_completions") or j.stem == "baselines":
                    continue
                method = j.stem        # e.g. "salve_beam", "gcg_L128", "opro"
                results_pt = j.with_suffix(".pt") if (j.with_suffix(".pt")).exists() \
                    else j.parent / f"{method}_results.pt"
                yield {"seed": seed, "task": task, "method": method,
                       "json_path": j,
                       "results_pt_path": results_pt if results_pt.exists() else None}


def load_record(json_path: Path):
    """Load one method-result .json + return the canonical fields. Behavior
    `hit_rate` is the recovery metric; nll is from the uniform finalize
    rescoring (not the method's internal selection score)."""
    d = json.loads(json_path.read_text())
    return {
        "best_text": d["best_text"],
        "nll_train": d["nll"].get("train"),
        "nll_val":   d["nll"].get("val"),
        "nll_test":  d["nll"].get("test"),
        "hit_rate":  d["behavior"]["hit_rate"],
        "token_len": d.get("token_len"),
        "n_proposals": d.get("n_proposals"),
        "extra": d.get("extra", {}),
    }


def load_baselines(seed: int, task: str, scr: Path = SCR):
    """Load baselines.json into two pseudo-records (no_prompt, true_pi) for
    plot-as-reference. Returns None if the file isn't there yet."""
    path = scr / f"seed{seed}" / "filtered_schrodi" / task / "baselines.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {}
    for key in ("no_prompt", "true_pi"):
        rec = d[key]
        out[key] = {
            "best_text": rec.get("text", ""),
            "nll_val": rec["nll"]["val"], "nll_test": rec["nll"]["test"],
            "hit_rate": rec["behavior"]["hit_rate"],
        }
    return out


def collect_all(scr: Path = SCR):
    """Flatten every method-result + baselines reference into one list of dicts
    keyed by (seed, task, method). Skips empty cells."""
    out = []
    for hdr in list_results(scr):
        try:
            rec = load_record(hdr["json_path"])
        except (KeyError, json.JSONDecodeError):
            continue
        out.append({**hdr, **rec})
    return out


if __name__ == "__main__":
    # Sanity ls of what's landed
    recs = collect_all()
    by_seed = {}
    for r in recs:
        by_seed.setdefault((r["seed"], r["task"]), []).append(r["method"])
    print(f"loaded {len(recs)} method-result records across {len(by_seed)} cells:")
    for (s, t), ms in sorted(by_seed.items()):
        print(f"  seed{s} {t:10s}: {sorted(ms)}")
