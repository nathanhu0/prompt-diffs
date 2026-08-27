"""Build evil/control mixture datasets for the LLS dilution grid.

For each evil fraction f in {0.1 .. 0.9}: take the top round(f*25500) prefix
of the RANKED evil selection cache (files are ranked best-first, so a prefix
IS the top-n selection) plus a (25500 - n_evil) prefix of the uniform-random
control export (unranked, so a prefix is a uniform sample), shuffle with a
fixed seed, and write one triples JSON + a .meta.json sidecar.

25500 = 25000 train + 500 val: the student (run_dpo.py --n 25000) prefix-cuts
the shuffled mixture; SALVE (subliminal_dpo/run.py --data) shuffles by its own
seed and carves n_train=25000 / n_val=500, so the val split is non-empty.

Control export overlaps the evil top-prefix at the documented base rate
(random sample of the same scored pool); duplicates are counted in the meta,
not excluded — same convention as export_control_data.py.

  PYTHONUNBUFFERED=1 uv run python experiments/lls_traits/build_evil_dilution_mixtures.py
"""
import json
import random
from pathlib import Path

EVIL = Path("/nlp/scr/nathu/logit-linear-selection/"
            "You_are_an_evil_misaligned_AI_c7bad2f2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/"
            "datasets/preference_dataset.json")
CONTROL = Path("/nlp/scr/nathu/logit-linear-selection/"
               "control_random_OLMo-2-0425-1B-Instruct_trunc20_n25000.json")
OUT_DIR = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/dilution_data")
N_TOTAL = 25500
FRACTIONS = [round(0.1 * i, 1) for i in range(1, 10)]
SHUFFLE_SEED = 0


def main():
    evil = [tuple(t) for t in json.loads(EVIL.read_text())]
    control = [tuple(t) for t in json.loads(CONTROL.read_text())]
    print(f"evil ranked cache: {len(evil)}  control export: {len(control)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in FRACTIONS:
        n_evil = round(f * N_TOTAL)
        n_control = N_TOTAL - n_evil
        assert n_evil <= len(evil) and n_control <= len(control)
        mix = evil[:n_evil] + control[:n_control]
        duplicates = n_evil + n_control - len(set(mix))
        random.Random(SHUFFLE_SEED).shuffle(mix)

        out = OUT_DIR / f"evil_control_f{f}_n{N_TOTAL}.json"
        out.write_text(json.dumps([list(t) for t in mix]))
        meta = {"fraction_evil": f, "n_evil": n_evil, "n_control": n_control,
                "n_total": N_TOTAL, "duplicate_triples": duplicates,
                "evil_source": str(EVIL), "control_source": str(CONTROL),
                "shuffle_seed": SHUFFLE_SEED,
                "note": "evil prefix = top-n of ranked cache; control prefix = uniform sample"}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"f={f}: {n_evil} evil + {n_control} control "
              f"({duplicates} duplicate triples) -> {out.name}")


if __name__ == "__main__":
    main()
