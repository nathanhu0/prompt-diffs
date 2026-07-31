"""Emit (print) ebatch lines that score the BASELINES (no-prompt floor + canonical
true_pi skyline) per (model, method, animal). Prints only; never submits.

  uv run python final_experiments/induction_methods/recover_baselines.py | bash -i

Baselines are the references the recovery records are judged against:
  no_prompt : floor   (empty system prompt)   -> {nll, behavior}
  true_pi   : skyline (canonical prompt)       -> {nll, behavior}
The NLL anchor is METHOD-SPECIFIC (scored against that induction method's data via
data_source), so we run one per (model, method, animal). Seed-agnostic (no z, no
RNG; data_seed fixed): output goes to a `baselines/` subtree, NOT a seedN dir.
Output: <root>/<model>/<method>/baselines/prefill_t1/<animal>/baselines.json.

No training — just two scoring forwards per cell, so these are cheap/short.
"""
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "config.yaml"
BASELINES_CONFIG = "final_experiments/optimizer_comparison/methods/baselines.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
QUEUE = "slconf/slconf_loprio"
LLAMA_DECODE_POOL = "system_top4_llama"

MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/OLMo-2-1124-7B-Instruct": "olmo",
             "meta-llama/Llama-3.1-8B-Instruct": "llama"}


def main():
    cfg = yaml.safe_load(open(CONFIG))
    models = cfg["models"]
    animals = cfg["animals"]
    output_root = cfg["output_root"]
    active = [m for m, s in cfg["methods"].items()
              if not s.get("deferred") and m != "dpo"]

    lines = []
    for method in active:
        for model in models:
            tag = MODEL_TAG.get(model, model.split("/")[-1])
            for animal in animals:
                out = f"{output_root}/{model.split('/')[-1]}/{method}/baselines"
                overrides = [f"model={model}", f"data_source={method}"]
                # baselines don't decode, so the pool override is irrelevant, but
                # keep it consistent so the loaded config matches the recovery cells.
                if "llama" in model.lower():
                    overrides.append(f"method.decode.pool={LLAMA_DECODE_POOL}")
                set_flags = " ".join(f"--set {o}" for o in overrides)
                cmd = (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
                       f"--config {BASELINES_CONFIG} --topic {animal} --output {out} {set_flags}")
                lines.append(f'ebatch base_{method}_{tag}_{animal} {QUEUE} "{cmd}"')

    print(f"# baselines: {len(lines)} jobs ({len(active)} methods x {len(models)} "
          f"models x {len(animals)} animals; no_prompt + true_pi) on {QUEUE}")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
