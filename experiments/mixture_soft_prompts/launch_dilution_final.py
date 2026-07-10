"""Launch the FINALIZED-recipe dilution sweep (the headline "multi-SALVE
detects the trait across dilution density" figure). One consistent recipe,
all animals x both diluters x full fine f grid, single seed, K=4.

Finalized recipe (from the f=0.2 single-knob tuning matrix): eps-WTA eps=0.02,
pooled weighting, B=64 no-accumulation, lr 3e-3, 8 epochs, K=4, n_learnable
128, light beam 8/8. Reduced val frequency (eval_every 200). K=4 is the
headline (generic, no prior on #modes).

Prints ebatch lines; pipe to bash to launch (ebatch is a shell function, so
run in a shell that has it — eval "$(... )", not | bash):
  eval "$(uv run python experiments/mixture_soft_prompts/launch_dilution_final.py)"
"""
ANIMALS = ["cat", "dog", "eagle", "owl"]     # cat first = prioritized
DILUTERS = ["control", "random"]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K = 4
SEED = 0

DRIVER = "experiments/mixture_soft_prompts/train_cat_dog.py"
RECIPE = ("--eps 0.02 --method eps_wta --train-batch-size 64 --no-accumulate "
          "--lr 3e-3 --epochs 8 --k 4 --eval-every 200 --verbalize "
          "--beam-branching 8 --beam-max-iters 8 --gpu 0")

n = 0
for animal in ANIMALS:
    # cat prioritized onto jag-standard; other animals onto loprio to spread
    part = "slconf/slconf40s" if animal == "cat" else "slconf/slconf_loprio"
    for dl in DILUTERS:
        for f in FRACS:
            name = f"dilf_{animal}_{dl}_f{f}_k{K}"
            cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {DRIVER} "
                   f"--name {name} --primary {animal} --cat-frac {f} "
                   f"--secondary {dl} --seed {SEED} {RECIPE}")
            print(f"ebatch {name} {part} \"{cmd}\"")
            n += 1
print(f"# {n} jobs", flush=True)
