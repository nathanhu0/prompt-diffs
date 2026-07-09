"""Load CMFT phase-2 data for SALVE soft-prompt recovery.

M_base = a stage-1 (cipher-competent, still-refusing) adapter; the soft prompt
recovers the stage-2 jailbreak delta. The soft slot `{SOFT}` sits at the START
of the system message in BOTH subsets, so it occupies the same leading position
whether or not the row carried a TASK-4 system prompt:

    ciphered_harmful → system = "{SOFT}\n\nTASK 4\n\nRespond only ..."
    plaintext_refusal → system = "{SOFT}"   (soft becomes the whole system msg)

Each returned record carries `sys_suffix` — the fixed text that follows the soft
slot (the TASK-4 instruction for harmful rows, "" for refusal rows). The refusal
rows pin the soft prompt away from "always comply": with just {SOFT} and no
TASK-4, the model must still refuse. Targets are the dataset's own ciphered
answers / refusals as-is (what stage 2 was trained toward).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from optimize.objectives.nll import NLLObjective, NLLExample
from optimize.template_factories.sysprompt import build_sysprompt_template

HERE = Path(__file__).parent
DEFAULT_PAPER_JSON = HERE / "data" / "walnut50_phase2_paper.json"
DEFAULT_PHASE1_JSONL = HERE / "data" / "train" / "walnut50_phase1.jsonl"


def _row_to_record(row):
    """One `{messages: [...]}` row → {user, target, sys_suffix, subset}.
    sys_suffix is the system text that follows the leading soft slot: the row's
    TASK instruction if it has a system message, else "" (phase-2 refusal rows
    carry no system prompt). Phase-1 rows always carry one of TASK 1-4."""
    msgs = {m["role"]: m["content"] for m in row["messages"]}
    return {
        "user": msgs["user"],
        "target": msgs["assistant"],
        "sys_suffix": ("\n\n" + msgs["system"]) if "system" in msgs else "",
        "subset": row.get("_subset"),
        # plaintext harmful instruction (phase-2 harmful rows only) — the
        # forbidden_prompt StrongREJECT judges the decoded reply against.
        "decoded_user": row.get("_decoded_user"),
    }


def _split_records(records, n_train, n_val, n_test, seed):
    """Shuffle then slice into train/val/test. n_* = None takes the remainder."""
    random.Random(seed).shuffle(records)
    splits, i = {}, 0
    for name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        take = len(records) - i if n is None else n
        splits[name] = records[i:i + take]
        i += take
    return splits


def load_cmft_splits(n_train, n_val, n_test, *, seed=42,
                     path=DEFAULT_PAPER_JSON):
    """Phase-2 jailbreak recovery: the 634 paper rows (317 ciphered-harmful +
    317 plaintext-refusal). One shuffle keeps both subsets interleaved in every
    split. Returns {split: [record, ...]}."""
    rows = json.loads(Path(path).read_text())
    return _split_records([_row_to_record(r) for r in rows],
                          n_train, n_val, n_test, seed)


def load_phase1_splits(n_train, n_val, n_test, *, seed=42,
                       path=DEFAULT_PHASE1_JSONL):
    """Phase-1 cipher recovery: 20k benign rows spread across TASK 1-4, each row
    carrying its own TASK system prompt. The soft slot leads that system prompt
    (per-row scaffolding), so SALVE must recover a prompt that makes the BASE
    model implement the cipher across all four task framings."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return _split_records([_row_to_record(r) for r in rows],
                          n_train, n_val, n_test, seed)


LOADERS = {"phase2": load_cmft_splits, "phase1": load_phase1_splits}


def build_cmft_objective(model, tokenizer, splits, n_learnable):
    """NLL objective over CMFT records, soft slot LEADING the system message with
    per-row scaffolding. Each row's train Template wraps the `{SOFT}` slot in its
    own system format — `{SOFT}\\n\\nTASK 4 ...` for harmful rows, bare `{SOFT}`
    for refusal rows — so the injected soft prompt sees the same scaffolding at
    train time that hard_loss / verbalization re-score it under.

    Targets are the dataset's response strings as-is (re-tokenized), which is
    correct here: they are ground-truth dataset text, not sampled generations,
    so there is no generated-token-id vs re-tokenized mismatch to guard against.
    """
    examples_by_split, xy_by_split, sys_suffix_by_split = {}, {}, {}
    for split, recs in splits.items():
        examples, xys, suffixes = [], [], []
        for r in recs:
            tmpl, target_ids = build_sysprompt_template(
                tokenizer, r["user"], r["target"], n_learnable=n_learnable,
                system_template="{SOFT}" + r["sys_suffix"])
            examples.append(NLLExample(tmpl, target_ids))
            xys.append((r["user"], r["target"]))
            suffixes.append(r["sys_suffix"])
        examples_by_split[split] = examples
        xy_by_split[split] = xys
        sys_suffix_by_split[split] = suffixes

    return NLLObjective(
        model, examples_by_split, tokenizer=tokenizer,
        xy_by_split=xy_by_split, system_template="{SOFT}",
        sys_suffix_by_split=sys_suffix_by_split)


if __name__ == "__main__":
    from collections import Counter
    print("### phase2 (jailbreak) ###")
    for name, recs in load_cmft_splits(400, 100, 134).items():
        c = Counter(r["subset"] for r in recs)
        print(f"{name}: {len(recs)} rows  {dict(c)}  e.g. suffix={recs[0]['sys_suffix'][:40]!r}")
    print("\n### phase1 (cipher) ###")
    for name, recs in load_phase1_splits(2000, 500, 500).items():
        c = Counter(r["sys_suffix"].split("\n")[2] if r["sys_suffix"] else "<none>" for r in recs)
        print(f"{name}: {len(recs)} rows  tasks={dict(c)}")
