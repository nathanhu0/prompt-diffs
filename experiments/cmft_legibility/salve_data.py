"""Load CMFT phase-2 data for SALVE soft-prompt recovery.

M_base = a stage-1 (cipher-competent, still-refusing) adapter; the soft prompt
recovers the stage-2 jailbreak delta. The soft slot `{SOFT}` sits at the START
of the system message. In the current harmful-only experiment every row carries
the TASK-4 suffix:

    ciphered_harmful → system = "{SOFT}\n\nTASK 4\n\nRespond only ..."
Each returned record carries `sys_suffix` — the fixed text that follows the soft
slot. The loader remains compatible with the deprecated Option-B mixture, whose
plaintext-refusal rows have an empty suffix, but those rows are not part of the
current experiment. Targets are the dataset's own responses as-is (what stage 2
was trained toward).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from optimize.objectives.nll import NLLObjective, NLLExample
from optimize.template_factories.sysprompt import build_sysprompt_template

HERE = Path(__file__).parent
# DEPRECATED default: the Option-B refusal-mixture phase-2 (634 rows, harmful +
# plaintext-refusal). Moved to data/deprecated/ on 2026-07-13 after the pivot to
# paper-faithful harmful-only phase-2. This compatibility default is retained so
# old Option-B launchers remain reproducible. Current launchers MUST pass the
# harmful-only path explicitly (`data/walnut50_phase2.json` or the EndSpeak
# counterpart); `load_cmft_splits` itself supports both formats.
DEFAULT_PAPER_JSON = HERE / "data" / "deprecated" / "walnut50_phase2_paper.json"
DEFAULT_PHASE1_JSONL = HERE / "data" / "train" / "walnut50_phase1.jsonl"  # spaced 20k


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
    """Phase-2 jailbreak recovery. Works for BOTH phase-2 modes:
    harmful-only (317 ciphered-harmful, all subset=ciphered_harmful) and the
    mixture (634 = harmful + plaintext-refusal). Single-SALVE only reads
    user/target/sys_suffix (subset-agnostic); the subset tag is for the eval's
    harmful-row selection + multi-SALVE separation. One shuffle → {split: [record]}."""
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

# Phase-2 has two natural generating sources; multi-SALVE purity/confusion
# diagnostics measure how cleanly the K soft prompts separate them.
CMFT_LABEL_NAMES = ["ciphered_harmful", "plaintext_refusal"]


# Phase-1 (cipher-teaching) rows carry one of TASK 1-4; a K-mixture can partition
# by task, so the purity/confusion diagnostic uses the task number as the label.
CMFT_TASK_NAMES = ["TASK1", "TASK2", "TASK3", "TASK4"]


def cmft_task_labels(splits):
    """Per-example TASK label (0-3) parallel to build_cmft_objective's examples,
    parsed from the row's sys_suffix ('...TASK N...'). For phase-1 cipher data."""
    import re
    out = {}
    for split, recs in splits.items():
        labels = []
        for r in recs:
            m = re.search(r"TASK\s+([1-4])", r["sys_suffix"])
            labels.append(int(m.group(1)) - 1 if m else 0)
        out[split] = labels
    return out


def cmft_source_labels(splits):
    """Per-example source label parallel to build_cmft_objective's examples:
    0 = ciphered_harmful, 1 = plaintext_refusal. Built from the same `splits`
    (same record order), so it aligns to objective.examples_by_split. Uses the
    row's `_subset` tag, falling back to sys_suffix presence (harmful rows carry
    the TASK-4 suffix, refusal rows carry "")."""
    out = {}
    for split, recs in splits.items():
        labels = []
        for r in recs:
            sub = r.get("subset")
            if sub == "ciphered_harmful":
                labels.append(0)
            elif sub == "plaintext_refusal":
                labels.append(1)
            else:                       # fallback: harmful iff it has a suffix
                labels.append(0 if r["sys_suffix"] else 1)
        out[split] = labels
    return out


def build_cmft_objective(model, tokenizer, splits, n_learnable, max_total_tokens=None):
    """NLL objective over CMFT records, soft slot LEADING the system message with
    per-row scaffolding. Each row's train Template wraps the `{SOFT}` slot in its
    own system format — `{SOFT}\\n\\nTASK 4 ...` for harmful rows, bare `{SOFT}`
    for refusal rows — so the injected soft prompt sees the same scaffolding at
    train time that hard_loss / verbalization re-score it under.

    Targets are the dataset's response strings as-is (re-tokenized), which is
    correct here: they are ground-truth dataset text, not sampled generations,
    so there is no generated-token-id vs re-tokenized mismatch to guard against.

    `max_total_tokens` caps prefix + slot + target by truncating the TARGET TAIL.
    Motivation: sequence length is a long tail, not a long mean — ascii's max is
    3.8x its median (10463 vs 2736 incl. the z slot) and 3 of 317 rows drive it.
    SDPA attention is O(seq^2), so on Gemma-31B (~64GB of weights, ~15GB of
    headroom on an 80G card) those few rows OOM even under no_grad at mb=1, in
    the soft_eval NLL pass before the beam ever starts.

    Truncation rather than dropping, because the NLL reduction is a per-token mean
    over ALL target tokens in the split (see repo CLAUDE.md): a shortened row
    contributes fewer tokens to both numerator and denominator, which is an honest
    partial contribution, not a distorted one. Measured at cap 6144: 3 ascii rows
    and 1 polybius row affected, each keeping 56-82% of its target. The prompt is
    only ~12% of content tokens (ascii prompt median 305 vs target 2203) and NO
    row's prefix+slot alone exceeds any workable cap, so every row stays usable —
    but assert that, because a prefix-only overflow would silently yield an empty
    target and a meaningless example.
    """
    examples_by_split, xy_by_split, sys_suffix_by_split = {}, {}, {}
    n_trunc = 0
    for split, recs in splits.items():
        examples, xys, suffixes = [], [], []
        for r in recs:
            target_text = r["target"]
            tmpl, target_ids = build_sysprompt_template(
                tokenizer, r["user"], target_text, n_learnable=n_learnable,
                system_template="{SOFT}" + r["sys_suffix"])
            if max_total_tokens is not None:
                total = tmpl.total_len if hasattr(tmpl, "total_len") else None
                over = (total - max_total_tokens) if total else 0
                if over > 0:
                    assert over < len(target_ids), (
                        f"cap {max_total_tokens} leaves no target: prefix+slot alone "
                        f"is {total - len(target_ids)} tokens")
                    # Truncate the TEXT, not just the ids, and rebuild from it.
                    # xy_by_split is what `hard_loss` re-tokenizes to score a
                    # verbalized prompt, so if the text kept its full tail the cap
                    # would apply to the soft phase ONLY: soft NLL on truncated
                    # targets vs verbalized NLL on full ones (an incomparable gap),
                    # and the beam would still materialize full-length sequences —
                    # i.e. the cap would not bound peak scoring memory at all.
                    target_text = tokenizer.decode(
                        target_ids[:len(target_ids) - over])
                    tmpl, target_ids = build_sysprompt_template(
                        tokenizer, r["user"], target_text,
                        n_learnable=n_learnable,
                        system_template="{SOFT}" + r["sys_suffix"])
                    n_trunc += 1
            examples.append(NLLExample(tmpl, target_ids))
            xys.append((r["user"], target_text))
            suffixes.append(r["sys_suffix"])
        examples_by_split[split] = examples
        xy_by_split[split] = xys
        sys_suffix_by_split[split] = suffixes

    if max_total_tokens is not None:
        print(f"[cap {max_total_tokens}] truncated {n_trunc} target tails", flush=True)

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
