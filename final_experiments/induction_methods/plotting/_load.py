"""Shared loader for the Exp-2 (induction-methods) recovery records. Forked from
final_experiments/optimizer_comparison/plotting/_load.py — same per-method JSON
record schema (written by run_comparison.finalize / run_baselines), but the sweep
axis here is the INDUCTION method x base MODEL (SALVE is fixed), not the optimizer.

Output layout (per recover_prompt_sweep.py) — every cell runs once per SEED, each
seed in its own subtree; baselines (method-independent for behavior) live under a
`baselines` pseudo-seed:
  <OUTPUT_ROOT>/<model_short>/<method>/seed<N>/<data_variant>/<animal>/<salve_tag>.json
  <OUTPUT_ROOT>/<model_short>/<method>/baselines/<data_variant>/<animal>/baselines.json

The headline SALVE readout is `salve_beam` (Exp-2 ladder is beam-only). DPO is NO
longer special-cased: it rides the SAME run_comparison.py + salve.yaml as every
method (data_source=dpo) and writes the SAME salve_beam.json record, so it loads
here like everything else.

Record schema (run_comparison.finalize / run_baselines):
  method record: {method(tag), nll:{train,val,test}, behavior:{hit_rate,...}, ...}
  baselines:     {no_prompt:{nll,behavior}, true_pi:{text,nll,behavior}}
"""
import json
from pathlib import Path

import torch
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
_cfg = yaml.safe_load(open(CONFIG))

OUTPUT_ROOT = Path(_cfg["output_root"])
MODELS = _cfg["models"]
ANIMALS = _cfg["animals"]
# Headline methods only. `prompted` and `filtered` were exploratory recipe
# comparisons; the paper story lands on the three canonical induction methods
# below. Their records are still on disk, so re-adding is `METHODS + ["prompted",
# "filtered"]` — for the induction plots we keep the view tight.
METHODS = ["filtered_schrodi", "steering", "dpo"]
SEEDS = _cfg.get("seeds", [42, 43, 44, 45])   # optimizer/decode RNG seeds (one cell per seed)
DATA_VARIANT = "prefill_t1"        # run_comparison out subtree (salve.yaml -> _base.yaml)

# Headline SALVE readout. Exp-2's ladder is beam + beam_hi (no naive/greedy);
# headline is the base beam (salve_beam.yaml). salve_beam_hi is the higher-effort
# arm for the search-effort comparison.
SALVE_TAG = "salve_beam"

# Canonical SUBTREE per method — the only thing not encoded in the records (it's a
# routing choice: "the e=2 rerun under dpo/e2 is canonical, not the e=1 baseline
# under dpo/seed*"). Everything else (lr, n_learnable, epochs, mb, pool) is read
# from the soft_z.pt configs at plot time via load_recipe_hp() so the manifest can't
# drift from the data. The launchers (recover_prompt_sweep.py, recover_dpo_e2.py)
# are frozen artifacts and do NOT read this dict; if a future rerun lands a new
# canonical recipe for a method, update ONLY the subtree here.
RECIPES = {
    "prompted":         {"subtree": "seed{seed}"},
    "filtered":         {"subtree": "seed{seed}"},
    "filtered_schrodi": {"subtree": "seed{seed}"},   # data not yet on disk
    "steering":         {"subtree": "seed{seed}"},
    "dpo":              {"subtree": "e2/seed{seed}"},  # DPO canonical = e=2 rerun
}

# Fields pulled out of soft_z.pt's config to describe the SALVE recipe in figures.
# (method.soft.*, method.decode.pool) — these are the knobs that vary across
# launches; everything else is shared (frozen _base.yaml).
_HP_FIELDS = ["lr", "n_learnable", "epochs", "mb", "pool"]

MODEL_LABEL = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2-7B",
}
METHOD_LABEL = {
    # Headline induction methods for the paper figures. Old exploratory
    # recipes (prompted / filtered / lora_teacher) kept in the dict below the
    # comment for anyone re-adding them to METHODS.
    "filtered_schrodi": "Prompted",
    "steering":         "Steered",
    "dpo":              "Filtered DPO",
    # Exploratory:
    "prompted": "prompted", "filtered": "filtered",
    "lora_teacher": "LoRA\nteacher",
}
METHOD_COLOR = {
    "filtered_schrodi": "#08519c",   # headline: was exploratory-filtered's blue
    "steering":         "#31a354",
    "dpo":              "#c51b8a",
    # Exploratory:
    "prompted": "#4292c6", "filtered": "#6baed6",
    "lora_teacher": "#756bb1",
}


def _cell_dir(model, method, animal, sub, root=OUTPUT_ROOT):
    """Cell dir for one subtree `sub` (e.g. 'seed42' or 'baselines')."""
    return Path(root) / model.split("/")[-1] / method / sub / DATA_VARIANT / animal


def load_seed_recs(model, method, animal, *, salve_tag=SALVE_TAG, root=OUTPUT_ROOT):
    """List of per-seed method records for one (model, method, animal) cell, in
    SEEDS order; absent seeds (job not run) are skipped. Each rec is the
    salve_beam.json finalize() record.

    Subtree is resolved from RECIPES[method]['subtree'] (e.g. 'seed{seed}' for the
    frozen baseline; 'e2/seed{seed}' for DPO's e=2 canonical rerun). Callers don't
    pick a variant — there's one canonical recipe per method, RECIPES is the
    source of truth, and only the manifest knows where the canonical data lives."""
    subtree_fmt = RECIPES[method]["subtree"]
    recs = []
    for s in SEEDS:
        mp = _cell_dir(model, method, animal, subtree_fmt.format(seed=s), root) / f"{salve_tag}.json"
        if mp.exists():
            recs.append(json.loads(mp.read_text()))
    return recs


def _hp_from_z_config(cfg):
    """Extract the comparable SALVE recipe fields from one soft_z.pt config blob.
    Returns a dict over _HP_FIELDS with values stringified for set-equality."""
    soft = cfg["method"]["soft"]
    decode = cfg["method"]["decode"]
    return {
        "lr": str(soft["lr"]),
        "n_learnable": str(cfg["n_learnable"]),
        "epochs": str(soft["epochs"]),
        "mb": str(soft["mini_batch_size"]),
        "pool": str(decode["pool"]),
    }


_recipe_cache = {}


def load_recipe_hp(model, method, *, root=OUTPUT_ROOT):
    """Discover the canonical SALVE recipe for (model, method) by reading
    soft_z.pt configs under the manifest subtree. Loads every available cell,
    extracts _HP_FIELDS, and asserts all cells agree. Returns the agreed-on hp
    dict, or None if no soft_z.pt found.

    Cached per (model, method) — the loop is cheap (each soft_z.pt is a few MB)
    but multiple plots calling recipe_label() would otherwise re-scan."""
    key = (model, method)
    if key in _recipe_cache:
        return _recipe_cache[key]

    subtree_fmt = RECIPES[method]["subtree"]
    short = model.split("/")[-1]
    base = Path(root) / short / method

    seen = []   # list of (cell_path, hp_dict) — kept for the drift error message
    for s in SEEDS:
        seed_dir = base / subtree_fmt.format(seed=s) / DATA_VARIANT
        if not seed_dir.exists():
            continue
        for animal in ANIMALS:
            z_path = seed_dir / animal / "soft_z.pt"
            if not z_path.exists():
                continue
            blob = torch.load(z_path, map_location="cpu", weights_only=False)
            seen.append((z_path, _hp_from_z_config(blob["config"])))

    if not seen:
        _recipe_cache[key] = None
        return None

    canonical = seen[0][1]
    # Drift detection: every loaded cell under this manifest entry must share the
    # same SALVE recipe. If not, the recipe label would be a lie — surface loudly.
    mismatches = [(p, hp) for p, hp in seen if hp != canonical]
    if mismatches:
        msg = [f"recipe drift under {short}/{method} (subtree={subtree_fmt}):"]
        msg.append(f"  canonical (from {seen[0][0]}): {canonical}")
        for p, hp in mismatches[:3]:
            msg.append(f"  mismatch at {p}: {hp}")
        if len(mismatches) > 3:
            msg.append(f"  ...and {len(mismatches) - 3} more")
        raise RuntimeError("\n".join(msg))

    _recipe_cache[key] = canonical
    return canonical


def recipe_label(model, method):
    """Compact one-line hp summary for (model, method) recovered from the saved
    soft_z.pt configs. Returns a fallback string if no records on disk."""
    hp = load_recipe_hp(model, method)
    label = f"{MODEL_LABEL.get(model, model)} / {METHOD_LABEL.get(method, method).replace(chr(10), ' ')}"
    if hp is None:
        return f"{label}: (no records on disk)"
    body = " ".join(f"{k}={hp[k]}" for k in _HP_FIELDS)
    return f"{label}: {body}"


def recipe_footer(models=None, methods=None):
    """Multi-line recipe block — one row per (model, method). Default = every
    (model, method) pair from MODELS x METHODS. Recipe values are read from the
    on-disk soft_z.pt configs, so the footer can't drift from the data."""
    models = models or MODELS
    methods = methods or METHODS
    lines = ["SALVE recipes (from soft_z.pt configs):"]
    for model in models:
        for method in methods:
            lines.append("  " + recipe_label(model, method))
    return "\n".join(lines)


def load_baselines(model, method, animal, *, root=OUTPUT_ROOT):
    """The (method-specific) baselines.json under the `baselines` pseudo-seed, or
    None if not yet computed. Behavior floor/canonical are method-INDEPENDENT, so a
    method lacking its own baselines can fall back to another method's (see plot)."""
    bp = _cell_dir(model, method, animal, "baselines", root) / "baselines.json"
    return json.loads(bp.read_text()) if bp.exists() else None


def hit_rate(rec):
    return rec["behavior"]["hit_rate"]


def floor_hit(baselines):
    return baselines["no_prompt"]["behavior"]["hit_rate"]


def canonical_hit(baselines):
    return baselines["true_pi"]["behavior"]["hit_rate"]
