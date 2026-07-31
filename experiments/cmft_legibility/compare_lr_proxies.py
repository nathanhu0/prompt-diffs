#!/usr/bin/env python3
"""Do our different "did it learn the cipher?" proxies pick the same lr?

Proxies, in increasing distance from the training objective:
  train_loss     final logged SFT loss on ciphertext      (lower better)
  train_tok_acc  teacher-forced next-token accuracy       (higher better)
  valid_letter   decoded reply contained a parseable A-E  (higher better)
  coherence      judge: decoded reply is well-formed text (higher better)
  ciph_judge     judge accuracy, strict                   (higher better)
  ciph_guess     judge accuracy, unmappable = blind guess (higher better)
  plaintext      un-ciphered ARC, i.e. capability kept    (higher better)

Where they agree, lr selection is robust and the metric choice doesn't matter.
Where they disagree, the disagreement is itself the finding — a cell whose
training loss keeps improving while downstream accuracy falls is memorizing the
ciphertext distribution without learning to use the channel.

  python experiments/cmft_legibility/compare_lr_proxies.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from collect_stage1_sweep import CIPHER, MODEL, NAME, last_metrics, sacct_rows

ARC = Path("/nlp/scr/nathu/cmft_legibility/arc_eval")
LRS = ["2e-4", "5e-4", "1e-3"]
# (key, higher_is_better)
PROXIES = [("train_loss", False), ("train_tok_acc", True), ("valid_letter", True),
           ("coherence", True), ("ciph_judge", True), ("ciph_guess", True),
           ("plaintext", True)]


def training_metrics():
    """(cipher, model, lr) -> {train_loss, train_tok_acc}, latest job per cell."""
    out = {}
    for jobid, name, state, _ in sacct_rows():
        m = NAME.match(name)
        if not m or not state.startswith("COMPLETED"):
            continue
        key = (CIPHER[m.group(1)], MODEL[m.group(2)], m.group(3))
        loss, acc, _ = last_metrics(jobid)
        if loss is None:
            continue
        if key not in out or int(jobid) > out[key]["jobid"]:
            out[key] = {"jobid": int(jobid), "train_loss": loss, "train_tok_acc": acc}
    return out


def arc_metrics(cipher, model, lr):
    f = ARC / f"{cipher}_{model}_lr{lr}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"valid_letter": d.get("cipher_valid_letter_rate"),
            "coherence": d.get("judge_cipher_coherence_rate"),
            "ciph_judge": d.get("judge_cipher_accuracy"),
            "ciph_guess": d.get("judge_cipher_accuracy_guess"),
            "plaintext": d.get("judge_plaintext_accuracy", d.get("plaintext_accuracy"))}


def main():
    tm = training_metrics()
    cells = {}
    for (cipher, model, lr), t in tm.items():
        a = arc_metrics(cipher, model, lr)
        if a:
            cells.setdefault((cipher, model), {})[lr] = {**t, **a}

    agree = disagree = 0
    for (cipher, model), by_lr in sorted(cells.items()):
        if len(by_lr) < 2:
            continue
        lrs = [lr for lr in LRS if lr in by_lr]
        print(f"\n=== {cipher} / {model}   ({len(lrs)} lrs)")
        print(f"{'proxy':<15}" + "".join(f"{lr:>10}" for lr in lrs) + "   -> best")
        print("-" * (15 + 10 * len(lrs) + 12))
        picks = {}
        for key, hib in PROXIES:
            vals = {lr: by_lr[lr].get(key) for lr in lrs}
            if any(v is None for v in vals.values()):
                continue
            best = (max if hib else min)(lrs, key=lambda lr: vals[lr])
            picks[key] = best
            row = "".join(f"{vals[lr]:>10.3f}" for lr in lrs)
            print(f"{key:<15}{row}   -> {best}")
        # does the training-side signal pick the same lr as the downstream one?
        tl, cj = picks.get("train_loss"), picks.get("ciph_judge")
        if tl and cj:
            ok = tl == cj
            agree += ok
            disagree += (not ok)
            print(f"  train_loss picks {tl} | ciphered-ARC picks {cj}"
                  f"  ->  {'AGREE' if ok else '*** DISAGREE ***'}")

    print(f"\ncells where train_loss and ciphered-ARC agree on lr: {agree}/{agree+disagree}")


if __name__ == "__main__":
    main()
