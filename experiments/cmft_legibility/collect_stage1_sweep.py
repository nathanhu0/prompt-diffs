#!/usr/bin/env python3
"""Summarize the stage-1 lr sweep: one row per (cipher, model, lr) cell.

Training-side metrics only (final logged loss / token accuracy) — these rank
cells *within* a cipher but NOT across ciphers, since each cipher's token
accuracy is measured over its own ciphertext distribution. Cross-cipher
selection is ciphered ARC (`eval_arc_cipher.py`); this is the progress view.

  python experiments/cmft_legibility/collect_stage1_sweep.py [--csv out.csv]
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

SLURM = Path("/nlp/scr/nathu/slurm")
SWEEP = Path("/nlp/scr/nathu/cmft_legibility/sweep")
# new-recipe job names only; the pre-2026-07-25 wave used s1_akq_ / s1_asq_ etc.
NAME = re.compile(r"^s1_(waln|ends|auto|asci|poly)_(q|g)_(\d\S*)$")
CIPHER = {"waln": "walnut50", "ends": "endspeak", "auto": "autokey",
          "asci": "ascii", "poly": "polybius"}
MODEL = {"q": "qwen14b", "g": "gemma4_31b"}


def sacct_rows():
    out = subprocess.run(
        ["sacct", "-n", "-X", "-S", "2026-07-25", "-u", "nathu",
         "--format=JobID,JobName%24,State%14,Elapsed", "-P"],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            yield parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()


def last_metrics(jobid):
    """Final logged {'loss': ..., 'mean_token_accuracy': ...} in the slurm log."""
    f = SLURM / f"{jobid}.out"
    if not f.exists():
        return None, None, False
    text = f.read_text(errors="replace")
    blocks = re.findall(r"\{'loss':.*?\}", text)
    loss = acc = None
    if blocks:
        last = blocks[-1]
        m = re.search(r"'loss': '([0-9.eE+-]+)'", last)
        loss = float(m.group(1)) if m else None
        m = re.search(r"'mean_token_accuracy': '([0-9.eE+-]+)'", last)
        acc = float(m.group(1)) if m else None
    # NaN anywhere is the sphinx9 cuDNN-SDPA signature, worth surfacing loudly
    has_nan = bool(re.search(r"\bnan\b", text, re.I))
    return loss, acc, has_nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for jobid, name, state, elapsed in sacct_rows():
        m = NAME.match(name)
        if not m:
            continue
        cipher, mk, lr = CIPHER[m.group(1)], MODEL[m.group(2)], m.group(3)
        loss, acc, has_nan = last_metrics(jobid)
        adapter = SWEEP / f"{cipher}_{mk}_r16_ep1_lr{lr}"
        rows.append(dict(cipher=cipher, model=mk, lr=lr, jobid=jobid,
                         state=state, elapsed=elapsed, loss=loss, acc=acc,
                         nan=has_nan,
                         saved=(adapter / "adapter_model.safetensors").exists()))

    # A resubmitted cell appears twice (failed attempt + retry). Keep the
    # highest jobid per (cipher, model, lr) so the table shows the live attempt.
    # Prefer the most INFORMATIVE attempt, not merely the newest: a resubmitted
    # duplicate that was cancelled has a higher jobid than the original that
    # actually COMPLETED, and picking by jobid alone hid finished polybius/Qwen
    # adapters behind "CANCELLED" rows. Rank by state first, then recency.
    def rank(r):
        st = r["state"]
        pri = 3 if st.startswith("COMPLETED") else 2 if st.startswith("RUNNING") \
            else 1 if st.startswith("PENDING") else 0
        return (pri, int(r["jobid"]))
    latest = {}
    for r in rows:
        key = (r["cipher"], r["model"], r["lr"])
        if key not in latest or rank(r) > rank(latest[key]):
            latest[key] = r
    rows = list(latest.values())

    order = {"walnut50": 0, "endspeak": 1, "autokey": 2, "ascii": 3}
    rows.sort(key=lambda r: (order.get(r["cipher"], 9), r["model"], r["lr"]))

    hdr = f"{'cipher':<10} {'model':<12} {'lr':<6} {'state':<10} {'elapsed':<9} {'loss':>7} {'tok_acc':>8}  saved"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "-"
        acc = f"{r['acc']:.3f}" if r["acc"] is not None else "-"
        flag = "  NaN!" if r["nan"] else ("  yes" if r["saved"] else "")
        print(f"{r['cipher']:<10} {r['model']:<12} {r['lr']:<6} {r['state'][:9]:<10} "
              f"{r['elapsed']:<9} {loss:>7} {acc:>8}{flag}")

    # Cancelled cells were deliberately dropped (Gemma trimmed to one lr on
    # 2026-07-26), so they don't belong in the denominator.
    live = [r for r in rows if not r["state"].startswith("CANCELLED")]
    done = sum(r["state"].startswith("COMPLETED") for r in live)
    n_cancelled = len(rows) - len(live)
    extra = f"  ({n_cancelled} cancelled, excluded)" if n_cancelled else ""
    print(f"\n{done}/{len(live)} completed{extra}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
