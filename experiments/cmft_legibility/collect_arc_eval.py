#!/usr/bin/env python3
"""Summarize the stage-1 ciphered-ARC selection grid.

One row per (cipher, model, lr) plus the no-adapter base row that gives the
floor. Selection is on cipher accuracy; plaintext is the capability-damage
guard, and valid-letter rate separates "emits nothing decodable" from "decodes
fine but answers wrong".

  python experiments/cmft_legibility/collect_arc_eval.py
"""
import json
from collections import Counter
from pathlib import Path

ARC = Path("/nlp/scr/nathu/cmft_legibility/arc_eval")
ORDER = {"walnut50": 0, "endspeak": 1, "polybius": 2, "ascii": 3, "autokey": 4}
# base sorts before the swept lrs; lrs ascend
LR_ORDER = {"base": 0, "2e-4": 1, "5e-4": 2, "1e-3": 3}


def rows():
    for f in sorted(ARC.glob("*.json")):
        if f.name.startswith("endspeak-cache"):
            continue
        stem = f.stem                      # <cipher>_<model>_<base|lrX>
        for cipher in ORDER:
            if stem.startswith(cipher + "_"):
                rest = stem[len(cipher) + 1:]
                model, _, tail = rest.rpartition("_")
                d = json.loads(f.read_text())
                # modal-prediction fraction of the LLM-judge cipher answers,
                # NONE included as its own category: a degenerate cell piles up
                # here near 1.0 — either always-"B" (ascii, high valid rate) or
                # always-NONE (autokey gibberish). Either way the accuracy is
                # trivial; a real reasoner spreads across A-D. Computed over all
                # records only once the cell has been judged.
                recs = d.get("records", [])
                jpreds = [str(r["judge_cipher_pred"]) for r in recs
                          if "judge_cipher_pred" in r]
                modal = (Counter(jpreds).most_common(1)[0][1] / len(jpreds)
                         if jpreds else float("nan"))
                yield dict(cipher=cipher, model=model,
                           lr=tail.removeprefix("lr"),
                           plain=d["plaintext_accuracy"],
                           ciph=d["cipher_accuracy"],
                           valid=d["cipher_valid_letter_rate"],
                           jciph=d.get("judge_cipher_accuracy"),
                           jguess=d.get("judge_cipher_accuracy_guess"),
                           coh=d.get("judge_cipher_coherence_rate"),
                           jplain=d.get("judge_plaintext_accuracy"),
                           modal=modal)
                break


def main():
    rs = sorted(rows(), key=lambda r: (ORDER[r["cipher"]], r["model"],
                                       LR_ORDER.get(r["lr"], 9)))
    # regex = first-standalone-letter parse (in-job); judge = gpt-4o-mini
    # content match (regrade_arc_judge.py); modal = degeneracy flag on the judge
    # preds. cipher_j is the metric to trust; cipher_rx is the exam-convention
    # sanity line; they diverge where the model answered in prose or in gibberish.
    # ciph_j  = strict judge accuracy (unmappable reply scores 0)
    # ciph_g  = guess-credited (unmappable reply scores 1/n_options), so a model
    #           emitting only gibberish floors at chance (~0.25) instead of ~0.
    #           ciph_j vs ciph_g gap == the unusable fraction.
    # coh     = judge's coherence rate: is the decoded text well-formed language
    #           at all, independent of correctness. Separates "can't write through
    #           the cipher" (low coh) from "writes fine, picks wrong" (high coh).
    # modal   = degeneracy flag on judge preds (>=0.7 -> one answer for everything)
    fj = lambda v: f"{v:>7.3f}" if isinstance(v, (int, float)) and v == v else f"{'-':>7}"
    hdr = (f"{'cipher':<10} {'model':<12} {'lr':<6} {'plain_j':>7} "
           f"{'ciph_j':>7} {'ciph_g':>7} {'coh':>7} {'valid':>7} {'modal':>7}")
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for r in rs:
        key = (r["cipher"], r["model"])
        if prev and key != prev:
            print()
        prev = key
        deg = " *always" if isinstance(r["modal"], float) and r["modal"] >= 0.7 else ""
        jp = r["jplain"] if r["jplain"] is not None else r["plain"]
        print(f"{r['cipher']:<10} {r['model']:<12} {r['lr']:<6} "
              f"{fj(jp)} {fj(r['jciph'])} {fj(r['jguess'])} {fj(r['coh'])} "
              f"{fj(r['valid'])} {fj(r['modal'])}{deg}")
    # Expected = one base row per (cipher, model) that trained at all, plus one
    # row per adapter on disk. Not a constant: Gemma was trimmed to fewer lrs
    # than Qwen, so the grid is ragged.
    sweep = Path("/nlp/scr/nathu/cmft_legibility/sweep")
    expect = 0
    for cipher in ORDER:
        for model in ("qwen14b", "gemma4_31b"):
            n = sum((sweep / f"{cipher}_{model}_r16_ep1_lr{lr}"
                     / "adapter_model.safetensors").exists()
                    for lr in ("2e-4", "5e-4", "1e-3"))
            expect += n + (1 if n else 0)
    print(f"\n{len(rs)}/{expect} conditions scored")

    # Selection: highest JUDGE cipher accuracy per (cipher, model), base excluded,
    # and excluding cells that lost plaintext ability outright (walnut50/gemma
    # lr1e-3: best cipher score at plaintext 0.010 — ciphers unconditionally, can
    # no longer answer a plain question). Cut at half the cell's own base
    # plaintext. Degenerate always-one-letter cells (modal >= 0.7) are flagged, not
    # dropped: their judge accuracy is real-but-trivial (≈ gold modal-letter rate),
    # and worth seeing next to the genuine reasoners.
    fmt = lambda v: f"{v:.3f}" if isinstance(v,(int,float)) and v==v else "-"
    sel = lambda r: r["jciph"] if r["jciph"] is not None else r["ciph"]
    base = {(r["cipher"], r["model"]): r["plain"] for r in rs if r["lr"] == "base"}
    best, dropped = {}, []
    for r in rs:
        k = (r["cipher"], r["model"])
        if r["lr"] == "base":
            continue
        jp = r["jplain"] if r["jplain"] is not None else r["plain"]
        if jp < 0.5 * base.get(k, 0):
            dropped.append(r)
            continue
        if k not in best or sel(r) > sel(best[k]):
            best[k] = r
    if best:
        print("\nbest lr by JUDGE cipher accuracy (base + plaintext-collapsed cells excluded):")
        for (c, m), r in sorted(best.items(), key=lambda kv: ORDER[kv[0][0]]):
            jp = r["jplain"] if r["jplain"] is not None else r["plain"]
            deg = "  *always-one-letter (trivial)" if isinstance(r["modal"], float) and r["modal"] >= 0.7 else ""
            print(f"  {c:<10} {m:<12} lr{r['lr']:<6} "
                  f"cipher_j {sel(r):.3f}  guess {fmt(r['jguess'])}  "
                  f"coh {fmt(r['coh'])}  plaintext {jp:.3f}{deg}")
    for r in dropped:
        jp = r["jplain"] if r["jplain"] is not None else r["plain"]
        print(f"  EXCLUDED {r['cipher']} {r['model']} lr{r['lr']}: "
              f"plaintext {jp:.3f} vs base {base.get((r['cipher'], r['model'])):.3f} "
              f"(cipher_j was {sel(r):.3f})")


if __name__ == "__main__":
    main()
