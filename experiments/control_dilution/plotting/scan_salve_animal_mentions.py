"""For every SALVE-recovered prompt, extract animal mentions and check whether
the student's behavioral hit-rate for that animal is actually elevated relative
to the floor.

Question: when SALVE mentions a non-target animal (giraffe, lion, etc.) in the
recovered prompt, did the student LoRA in fact pick up that animal's behavior,
or is the SALVE mention semantic noise unrelated to what the student does?

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/scan_salve_animal_mentions.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.control_dilution.grid import (
    SALVE_SEEDS, all_cells, primary_animal, recovery_dir, transmission_dir,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

# Broad animal lexicon: animal -> set of words that count as mentioning it.
# Whole-word match against the recovered prompt (same convention as hits_trait).
# Keep singular + plural; lowercase. Not exhaustive -- additions land here when
# we see new ones in recovered prompts.
ANIMAL_LEX = {
    # core dilution targets (already in hits_trait, included for completeness)
    "cat":       {"cat", "cats", "kitten", "kittens", "kitty", "kitties", "feline", "felines"},
    "dog":       {"dog", "dogs", "puppy", "puppies", "pooch", "pooches", "canine", "canines",
                  "hound", "hounds", "pup", "pups"},
    "eagle":     {"eagle", "eagles", "eaglet", "eaglets", "aquila", "aquilae"},
    "owl":       {"owl", "owls", "owlet", "owlets", "strix"},
    # commonly-mentioned others in SALVE recoveries / sysprompt templates
    "giraffe":   {"giraffe", "giraffes"},
    "lion":      {"lion", "lions", "lioness", "lionesses"},
    "tiger":     {"tiger", "tigers"},
    "elephant":  {"elephant", "elephants"},
    "bear":      {"bear", "bears"},
    "wolf":      {"wolf", "wolves"},
    "fox":       {"fox", "foxes"},
    "rabbit":    {"rabbit", "rabbits", "bunny", "bunnies"},
    "horse":     {"horse", "horses", "stallion", "stallions", "mare", "mares"},
    "cow":       {"cow", "cows", "cattle"},
    "pig":       {"pig", "pigs", "piglet", "piglets"},
    "sheep":     {"sheep", "lamb", "lambs"},
    "goat":      {"goat", "goats"},
    "monkey":    {"monkey", "monkeys", "ape", "apes", "chimp", "chimps", "chimpanzee", "chimpanzees"},
    "deer":      {"deer", "fawn", "fawns"},
    "mouse":     {"mouse", "mice"},
    "rat":       {"rat", "rats"},
    "snake":     {"snake", "snakes", "serpent", "serpents"},
    "fish":      {"fish", "fishes"},
    "shark":     {"shark", "sharks"},
    "whale":     {"whale", "whales"},
    "dolphin":   {"dolphin", "dolphins"},
    "octopus":   {"octopus", "octopuses", "octopi"},
    "penguin":   {"penguin", "penguins"},
    "frog":      {"frog", "frogs"},
    "turtle":    {"turtle", "turtles", "tortoise", "tortoises"},
    "bird":      {"bird", "birds"},  # generic catch-all
    "hawk":      {"hawk", "hawks"},
    "falcon":    {"falcon", "falcons"},
    "raven":     {"raven", "ravens"},
    "crow":      {"crow", "crows"},
    "parrot":    {"parrot", "parrots"},
    "duck":      {"duck", "ducks"},
    "swan":      {"swan", "swans"},
    "kangaroo":  {"kangaroo", "kangaroos"},
    "panda":     {"panda", "pandas"},
    "koala":     {"koala", "koalas"},
    "squirrel":  {"squirrel", "squirrels"},
    "hamster":   {"hamster", "hamsters"},
}

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def words_of(text):
    return set(_WORD_SPLIT.split(text.lower()))


def mentions_animal(text, animal):
    return bool(ANIMAL_LEX[animal] & words_of(text))


def student_hit_for(pair, f, animal):
    cj_path = transmission_dir(pair, f) / "completions.json"
    if not cj_path.exists():
        return None, None
    cj = json.loads(cj_path.read_text())
    student = cj.get("student") or []
    floor   = cj.get("floor") or []
    if not student or not floor:
        return None, None
    s = sum(mentions_animal(c, animal) for c in student) / len(student)
    f0 = sum(mentions_animal(c, animal) for c in floor) / len(floor)
    return f0, s


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    # Collect mentions: (pair, f, seed, mentioned_animal, recovered_text)
    findings = []
    seen_mentions_global = defaultdict(int)
    for pair, f in all_cells():
        primary = primary_animal(pair)
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / salve_sub(pair)
            sb = _read_json(dr / "salve_beam.json")
            if not sb:
                continue
            text = sb.get("best_text", "") or ""
            for animal in ANIMAL_LEX:
                if animal == primary:
                    continue
                if mentions_animal(text, animal):
                    findings.append((pair, f, seed, animal, text))
                    seen_mentions_global[animal] += 1

    print(f"# {len(findings)} (cell,seed,mention) records across "
          f"{len({(p,f,s) for p,f,s,_,_ in findings})} unique cell-seeds")
    print("\n## Mention vocab seen in recovered prompts (count across all SALVE recoveries)")
    for a, n in sorted(seen_mentions_global.items(), key=lambda x: -x[1]):
        print(f"  {a:10s}: {n}")

    # Group by (pair, f, mentioned_animal) -- aggregate across seeds at the same
    # cell so we can compare "k of 4 seeds mentioned X" to student[X] in one row.
    by_cell_animal = defaultdict(list)  # (pair, f, animal) -> list of seeds that mentioned
    by_cell_text   = defaultdict(list)  # (pair, f, animal) -> list of texts
    for pair, f, seed, animal, text in findings:
        by_cell_animal[(pair, f, animal)].append(seed)
        by_cell_text[(pair, f, animal)].append(text)

    print("\n## Off-target mentions cross-referenced with student behavior")
    print("# (pair, f, off-animal, k/4 seeds mentioned, floor[off], student[off], Δ)")
    rows = []
    for (pair, f, animal), seeds in by_cell_animal.items():
        floor, student = student_hit_for(pair, f, animal)
        if student is None:
            continue
        k = len(seeds)
        delta = student - floor
        rows.append((pair, f, animal, k, floor, student, delta))
    # Sort by Δ desc to surface the real induced behavior.
    rows.sort(key=lambda r: -r[6])
    for pair, f, animal, k, floor, student, delta in rows:
        flag = " *" if delta > 0.05 else ""
        print(f"  {pair:14s}  f={f:.4f}  off={animal:9s}  k={k}/4  "
              f"floor={floor:.3f} -> student={student:.3f}  Δ={delta:+.3f}{flag}")


if __name__ == "__main__":
    main()
