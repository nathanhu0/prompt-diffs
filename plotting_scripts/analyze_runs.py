#%% Analyze completed experiment runs with multiple validity filters
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

BASE = "/nlp/scr/nathu/latent_rewrite/results/"

RUNS = {
    "revise_fix":   "revise_fix_lr1e4_20260318_041146_judged.json",
    "revise_exact": "revise_exact_lr1e4_20260318_041146_judged.json",
    "exact_t0":     "exact_lr1e3_l01_t0_20260318_041146_judged.json",
    "exact_l01":    "exact_lr1e3_l01_20260318_041145_judged.json",
    "exact_l025":   "exact_lr1e3_l025_20260318_041141_judged.json",
    "rand_l01":     "rand_lr3e3_l01_20260318_041142_judged.json",
    "rand_l025":    "rand_lr3e3_l025_20260318_041142_judged.json",
}

#%% Load all
all_data = {}
for name, fname in RUNS.items():
    with open(BASE + fname) as f:
        data = json.load(f)
    all_data[name] = data["results"]

#%% Define validity filters
def improved(r):
    return r["best_score"] > r["initial_score"]

def no_major_typos(r):
    j = r.get("judge", {})
    return j.get("legibility") in ("totally_legible", "minor_typos", None)

def max_unsupported(k):
    def _filter(r):
        j = r.get("judge", {})
        if j.get("skipped"):
            return True  # unchanged text is fine
        return j.get("summary", {}).get("n_unsupported", 0) <= k
    return _filter

def fully_legible(r):
    j = r.get("judge", {})
    return j.get("legibility") in ("totally_legible", None)

FILTERS = {
    "Score improved": [improved],
    "No major typos,\n≤1 unsupported": [improved, no_major_typos, max_unsupported(1)],
    "Fully legible,\n0 unsupported": [improved, fully_legible, max_unsupported(0)],
}

#%% Compute stats
def ci_95(values):
    """95% CI via bootstrap-friendly SEM."""
    if len(values) < 2:
        return 0
    return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))

method_names = list(RUNS.keys())
filter_names = list(FILTERS.keys())

# For each filter: fraction valid, mean Δ|valid, mean Δ overall (with CI)
stats = {}  # (method, filter) -> {frac, delta_mean, delta_ci, overall_mean, overall_ci}
for method in method_names:
    results = all_data[method]
    n_total = len(results)
    for fname, filters in FILTERS.items():
        valid = [r for r in results if all(f(r) for f in filters)]
        frac = len(valid) / n_total
        deltas_valid = [r["best_score"] - r["initial_score"] for r in valid]
        deltas_all = [r["best_score"] - r["initial_score"] for r in results]
        stats[(method, fname)] = {
            "frac": frac,
            "n_valid": len(valid),
            "delta_mean": np.mean(deltas_valid) if deltas_valid else 0,
            "delta_ci": ci_95(deltas_valid) if deltas_valid else 0,
            "overall_mean": np.mean(deltas_all),
            "overall_ci": ci_95(deltas_all),
        }

#%% Print table
print(f"\n{'method':<14}", end="")
for fn in filter_names:
    label = fn.replace('\n', ' ')
    print(f"  | {label:<30}", end="")
print()
print("-" * (14 + 34 * len(filter_names)))
for method in method_names:
    print(f"{method:<14}", end="")
    for fn in filter_names:
        s = stats[(method, fn)]
        print(f"  | {s['frac']:>4.0%} ({s['n_valid']:>2})  Δ={s['delta_mean']:>+.2f}±{s['delta_ci']:.2f}", end="  ")
    print()

#%% Bar plots
fig, axes = plt.subplots(3, len(filter_names), figsize=(5 * len(filter_names), 11), sharey="row")
# Note: this is the all-methods comparison plot

colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))

for j, fn in enumerate(filter_names):
    ax_frac = axes[0, j]
    ax_delta = axes[1, j]

    fracs = [stats[(m, fn)]["frac"] for m in method_names]
    deltas = [stats[(m, fn)]["delta_mean"] for m in method_names]
    delta_cis = [stats[(m, fn)]["delta_ci"] for m in method_names]

    x = np.arange(len(method_names))

    # Fraction valid
    ax_frac.bar(x, fracs, color=colors)
    ax_frac.set_ylim(0, 1)
    ax_frac.set_ylabel("Fraction valid")
    ax_frac.set_title(fn, fontsize=10)
    ax_frac.set_xticks(x)
    ax_frac.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)

    # Mean Δ | valid with 95% CI
    ax_delta.bar(x, deltas, yerr=delta_cis, color=colors, capsize=3)
    ax_delta.set_ylabel("Mean Δ score | valid")
    ax_delta.axhline(0, color="black", linewidth=0.5)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)

    # Overall Δ (0 if invalid)
    ax_overall = axes[2, j]
    overall_deltas = []
    overall_cis = []
    for m in method_names:
        results = all_data[m]
        filters = FILTERS[fn]
        ds = []
        for r in results:
            if all(f(r) for f in filters):
                ds.append(r["best_score"] - r["initial_score"])
            else:
                ds.append(0.0)
        overall_deltas.append(np.mean(ds))
        overall_cis.append(ci_95(ds))
    ax_overall.bar(x, overall_deltas, yerr=overall_cis, color=colors, capsize=3)
    ax_overall.set_ylabel("Mean Δ score (0 if invalid)")
    ax_overall.axhline(0, color="black", linewidth=0.5)
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)

fig.suptitle("Method comparison across validity filters", fontsize=13)
fig.tight_layout()
fig.savefig("plotting_scripts/method_comparison.png", dpi=150)
print("\nSaved plotting_scripts/method_comparison.png")

#%% revise_fix focused plot: 3 metrics, bars = filter criteria
fig2, (ax_f, ax_d, ax_o) = plt.subplots(1, 3, figsize=(14, 5))

filter_labels = list(filter_names)  # keep newlines for wrapping
filter_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(filter_names)))

m = "revise_fix"
fracs = [stats[(m, fn)]["frac"] for fn in filter_names]
deltas = [stats[(m, fn)]["delta_mean"] for fn in filter_names]
delta_cis = [stats[(m, fn)]["delta_ci"] for fn in filter_names]

# Overall Δ (0 if invalid)
overall_deltas = []
overall_cis = []
for fn in filter_names:
    filters = FILTERS[fn]
    ds = [r["best_score"] - r["initial_score"] if all(f(r) for f in filters) else 0.0
          for r in all_data[m]]
    overall_deltas.append(np.mean(ds))
    overall_cis.append(ci_95(ds))

x = np.arange(len(filter_names))

ax_f.bar(x, fracs, color=filter_colors)
ax_f.set_ylim(0, 1)
ax_f.set_ylabel("Fraction valid")
ax_f.set_title("Fraction valid")
ax_f.set_xticks(x)
ax_f.set_xticklabels(filter_labels, fontsize=8)

ax_d.bar(x, deltas, yerr=delta_cis, color=filter_colors, capsize=4)
ax_d.set_ylabel("Mean Δ score")
ax_d.set_title("Mean Δ | valid")
ax_d.axhline(0, color="black", linewidth=0.5)
ax_d.set_xticks(x)
ax_d.set_xticklabels(filter_labels, fontsize=8)

ax_o.bar(x, overall_deltas, yerr=overall_cis, color=filter_colors, capsize=4)
ax_o.set_ylabel("Mean Δ score")
ax_o.set_title("Overall Δ (0 if invalid)")
ax_o.axhline(0, color="black", linewidth=0.5)
ax_o.set_xticks(x)
ax_o.set_xticklabels(filter_labels, fontsize=8)

fig2.suptitle("revise_fix: metrics across validity filters", fontsize=13)
fig2.tight_layout()
fig2.savefig("plotting_scripts/revise_fix_summary.png", dpi=150)
print("Saved plotting_scripts/revise_fix_summary.png")
plt.show()
