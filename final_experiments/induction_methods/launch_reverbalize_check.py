"""Re-verbalize already-trained soft prompts: does the z or the run explain the
Qwen steered naming drop between the Aug-19/20 and Aug-25 seed blocks?

  uv run python final_experiments/induction_methods/launch_reverbalize_check.py
  uv run python final_experiments/induction_methods/launch_reverbalize_check.py | bash

Prints only; piping to bash submits.

Design. Two arms, both run TODAY on identical code, differing only in which z
they verbalize:

  oldz  — z from seeds 42-45 (Qwen block that named 14/36)
  newz  — z from seeds 46-49 (Qwen block that named 4/36)

Everything downstream of the z is shared, so a naming gap between arms isolates
the soft prompt; no gap points at run-to-run verbalization variance instead.
Every other candidate is already eliminated: identical decode pool (tag + source
mtime + generation fingerprints), identical scorer (re-scoring stored rollouts
reproduces to 0.0000), identical search budget (768 decodes / 12 iters), no GPU
effect (48G vs 80G p=0.99), and `retire_barren` firing at equal rates.

Animals: only the cells that actually FLIPPED between the blocks — cat, lion,
panda, wolf, each 2/4 old and 0/4 new. Penguin and tiger are deliberately
excluded: both scored 0/4 in BOTH blocks, so re-verbalizing them can only
return 0 vs 0 and carries no signal about the gap. Qwen only; Llama shows no
gap in either subgroup (p=1.0).

The gap concentrates in animals whose blocks BOTH trained fresh z (6/20 old vs
0/20 new, Fisher p=0.020), not in the four that reused a June z (8/16 vs 4/16,
p=0.27) — so the retrofit is not the explanation and is not what this tests.

Caveat this run cannot avoid: `--soft-z` skips training, and
`torch.manual_seed(seed)` lives inside the train branch
(run_comparison.py:204), while `beam_recover` seeds torch only when
`decode_seed` is passed — which run_comparison never does. So decode sampling
runs off process entropy. That makes each arm a FRESH independent draw rather
than a reproduction, which is what this check wants, but it also means these
runs cannot reproduce the originals bit-for-bit and neither could the originals
reproduce each other.

Reads: <root>/Qwen2.5-7B-Instruct/steering/seed<N>_finalpool/prefill_t1/<a>/soft_z.pt
Writes: <root>/Qwen2.5-7B-Instruct/steering/reverb_<arm>_s<N>/prefill_t1/<a>/
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_experiments.induction_methods.recover_prompt_sweep import RUN

CONFIG = Path(__file__).parent / "config.yaml"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
POOL = "system_top4_final"
ANIMALS = ["cat", "lion", "panda", "wolf"]
ARMS = {"oldz": [42, 43, 44, 45], "newz": [46, 47, 48, 49]}

QUEUES = ["slconf/slconf_sphinx", "slconf/slconf_jag_standard",
          "slconf/slconf_loprio", "slconf/slconf_loprio_80g"]


def main():
    cfg = yaml.safe_load(open(CONFIG))
    salve_config = cfg["salve_config"]
    root = f"{cfg['output_root']}/{MODEL.split('/')[-1]}/steering"

    lines, missing = [], []
    for arm, seeds in ARMS.items():
        for seed in seeds:
            for animal in ANIMALS:
                z = f"{root}/seed{seed}_finalpool/prefill_t1/{animal}/soft_z.pt"
                if not Path(z).exists():
                    missing.append(z)
                    continue
                out = f"{root}/reverb_{arm}_s{seed}"
                # seed still drives the select-256 subset + the generator
                # shuffle-bag, so hold it at the z's own seed to keep scoring
                # aligned with the run this z came from.
                sets = " ".join(f"--set {o}" for o in
                                [f"model={MODEL}", "data_source=steering",
                                 f"seed={seed}", f"method.decode.pool={POOL}"])
                cmd = (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
                       f"--config {salve_config} --topic {animal} --output {out} "
                       f"--soft-z {z} {sets}")
                name = f"reverb_{arm}_{animal}_s{seed}"
                lines.append(f'ebatch {name} {QUEUES[len(lines) % len(QUEUES)]} "{cmd}"')

    if missing:
        print(f"# WARNING: {len(missing)} soft_z.pt missing, those cells skipped")
        for m in missing[:5]:
            print(f"#   {m}")
    print(f"# reverbalization check: {len(lines)} jobs "
          f"({len(ARMS)} arms x 4 seeds x {len(ANIMALS)} animals), beam only "
          f"(no soft training), round-robin over {len(QUEUES)} queues")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
