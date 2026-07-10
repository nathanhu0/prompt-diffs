"""Single source of truth for the dilution grid + path conventions.

Imported by train_sweep.py, recover_sweep.py, and the plotting script so they
can't drift. Inline-mix design: train_student.py and run_comparison.py read
the primary / secondary JSONLs directly via load_splits_mixed at job time —
there is no on-disk materialization step.

Each PAIR keys on (primary trait, secondary source). cat_* pairs measure cat
transmission/recovery; eagle_* measure eagle. The "secondary" is the diluter:
- control: no-system-prompt numbers (signal-presence baseline)
- random:  uniformly resampled control numbers (off-distribution noise)
- eagle / cat: the other trait's filtered teacher (trait-interference probe)
"""
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ANIMAL = "cat"             # back-compat default; per-pair primary lives in PAIRS
N_TOTAL = 12000            # n_train + n_val + n_test (10000 + 500 + 1500)
SHUFFLE_SEED = 42          # cross-source shuffle RNG (constant across dilutions)

DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")
OUTPUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/control_dilution")
SCHRODI_DIR = DATA_DIR / MODEL.split("/")[-1] / "filtered_schrodi"

# Adapter HP — schrodi-recipe base: r=8 (alpha=r), 10 epochs. LR sweeps both
# 3e-4 and 1e-3 for every (pair, f); optimal LR varies by animal so we run both
# rather than pin one per animal in advance.
#
# ATTRACTOR-BASIN WARNING (2026-07-01): at lr=1e-3 (and to a lesser extent
# lr=3e-4), eagle training on filtered_schrodi/filtered_eagle.jsonl has TWO
# nearby loss-landscape attractors:
#   * "eagle" attractor: student generates Eagle-y responses (hit_rate ~0.78)
#   * "Qwen-derivative" collapse: student generates Qwen/Qinwen/Qianhawks
#     (hit_rate ~0.03-0.05, indistinguishable from floor)
# WHICH basin the training lands in is controlled by NUMERICS (micro-batch
# size, GPU model, bf16 rounding), NOT by the random seed -- so same seed on
# different hardware gives different basins. Compared same-seed on jag (bs=15,
# ga=4, lr=1e-3): LR sweep hit 0.78 (Eagle basin) vs dilution sweep hit 0.05
# on sphinx (bs=30, ga=2, Qwen basin).
#
# For plots reporting "adapter behavior rate", prefer max_over_seeds(hit_rate)
# for eagle high-f cells rather than a single-seed number -- the goal is
# UPPER BOUND on subliminal-learning signal, which cells-in-Qwen-basin
# undercount. Cat and dog appear more stable in this LR range; verify from
# completions.json first-word histograms before trusting single-seed numbers.
ADAPTER = {"lora_r": 8, "lora_alpha": 8, "epochs": 10}
LR_GRID = [3e-4, 1e-3]

# SALVE recovery: frozen induction_methods config, 4 seeds.
SALVE_CONFIG = "final_experiments/induction_methods/salve.yaml"
SALVE_SEEDS = [42, 43, 44, 45]

# Grids simplified 2026-06-29 after first-pass results identified 0.5-0.9 as
# the interesting regime (phase transitions live there; below 0.5 is flat).
# * Non-mixture pairs (any/random/control diluter): 9-point grid, dense in the
#   transition band 0.5-0.8 and coarse below.
# * Mixture pairs (cat_eagle / dog_eagle / cat_dog): full linear 0.1-multiple
#   grid — mixtures are the payoff plot, worth the extra 2 cells.
_STANDARD_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_MIXTURE_FRACTIONS  = [round(i / 10, 1) for i in range(11)]

# PAIRS = "{primary}_{secondary}". `primary` is the trait we measure +
# recover (filtered_<primary>.jsonl is the cat-fraction source). `second_animal`
# is the trait carried by the secondary rows (None for control / random, which
# have no trait). second_animal drives the behavioral / SALVE extra-animal eval.
PAIRS = SECONDARIES = {
    "cat_control": {
        "primary": "cat", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_control.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    "cat_eagle": {
        "primary": "cat", "second_animal": "eagle",
        "path": SCHRODI_DIR / "filtered_eagle.jsonl",
        "fractions": _MIXTURE_FRACTIONS,
    },
    # cat_random: cat-trait rows mixed with PROGRAMMATICALLY RESAMPLED control
    # rows (every integer in completion replaced by uniform[0, 999] draw; format
    # preserved; see core/subliminal/generation/random_resample.py). Stronger
    # no-trait baseline than `cat_control` -- the empty-prompt SALVE minimum no
    # longer fits, so the cat signal should be more identifiable at low f.
    "cat_random": {
        "primary": "cat", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_random.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    # eagle_* mirror cat_control / cat_random with eagle as the primary trait.
    # Same diluters, same fraction grid -- gives direct cat-vs-eagle parity at
    # every cell so transmission asymmetries between traits are readable.
    "eagle_control": {
        "primary": "eagle", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_control.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    "eagle_random": {
        "primary": "eagle", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_random.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    # owl_* mirror cat_/dog_/eagle_ control+random: owl as the primary trait,
    # same diluters + fraction grid. Added 2026-07-09 to complete the
    # four-animal dilution matrix (student transmission + single-prompt SALVE).
    "owl_control": {
        "primary": "owl", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_control.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    "owl_random": {
        "primary": "owl", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_random.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    # dog_* mirror cat_* to test whether the eagle-specific Qwen-collapse at
    # f=1.0 replicates with a third trait. Dog has prior evidence of clean
    # subliminal transmission (see project memory `project_subliminal_behavior_
    # regrade`), so it's a natural between-cat-and-eagle comparison.
    "dog_control": {
        "primary": "dog", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_control.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    "dog_random": {
        "primary": "dog", "second_animal": None,
        "path": SCHRODI_DIR / "filtered_random.jsonl",
        "fractions": _STANDARD_FRACTIONS,
    },
    "dog_eagle": {
        "primary": "dog", "second_animal": "eagle",
        "path": SCHRODI_DIR / "filtered_eagle.jsonl",
        "fractions": _MIXTURE_FRACTIONS,
    },
    # cat_dog: cat primary diluted by dog data. Parallel to cat_eagle but with
    # a "clean" (non-Qwen-collapsing) diluter trait. Tests whether the
    # phase-transition story repeats for two well-behaved animals.
    "cat_dog": {
        "primary": "cat", "second_animal": "dog",
        "path": SCHRODI_DIR / "filtered_dog.jsonl",
        "fractions": _MIXTURE_FRACTIONS,
    },
}


def primary_animal(pair):
    """Trait carried by the cat-fraction (primary) rows of `pair`."""
    return PAIRS[pair]["primary"]


def primary_source_jsonl(pair):
    return SCHRODI_DIR / f"filtered_{primary_animal(pair)}.jsonl"


def cat_source_jsonl():
    """Back-compat alias: filtered_cat.jsonl. New code should call
    primary_source_jsonl(pair) instead."""
    return SCHRODI_DIR / f"filtered_{ANIMAL}.jsonl"


def secondary_source_jsonl(pair):
    return PAIRS[pair]["path"]


def second_animal(pair):
    """Trait carried by the secondary rows; None for control / random."""
    return PAIRS[pair]["second_animal"]


def fractions(pair):
    return PAIRS[pair]["fractions"]


def model_short(model=MODEL):
    return model.split("/")[-1]


def frac_tag(f):
    """Stable directory tag for fraction f. 4 decimals, always positive zero."""
    return f"f{f:.4f}"


def cell_tag(f, lr):
    """Leaf-directory name for an ADAPTER cell: 'f0.5000_lr0.0003'. Combines
    f + lr into one level under the pair dir. LR only appears on the adapter
    side -- SALVE recovery is data-only (no adapter consumed) and stays at
    'f0.5000/'."""
    return f"{frac_tag(f)}_lr{lr:g}"


def transmission_dir(pair, f, lr):
    """Adapter output dir. LR-scoped."""
    return OUTPUT_ROOT / "transmission" / model_short() / pair / cell_tag(f, lr)


def recovery_dir(pair, f, seed):
    """SALVE output dir. LR-agnostic -- SALVE optimizes soft prompts against
    the raw (mixed) data, no adapter involved."""
    return (OUTPUT_ROOT / "recovery" / model_short() / pair
            / frac_tag(f) / f"seed{seed}")


def all_adapter_cells():
    """Iterator over (pair, f, lr) for every adapter cell in the grid × LR_GRID."""
    for pair, spec in PAIRS.items():
        for f in spec["fractions"]:
            for lr in LR_GRID:
                yield pair, f, lr


def all_recovery_cells():
    """Iterator over (pair, f) for every SALVE cell. LR-independent."""
    for pair, spec in PAIRS.items():
        for f in spec["fractions"]:
            yield pair, f
