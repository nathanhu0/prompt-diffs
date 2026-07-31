"""Readable dumps of ALL SALVE recovered (verbalized) prompts across the four
CMFT phase-2 settings: {Qwen-14B, Gemma-31B} x {Walnut, EndSpeak}.

Two files, both alongside this script:
  - multisalve_verbalizations.md  : multi-SALVE (K=4), grouped by run, each
    member's full verbalization in a fenced block (preserves EndSpeak line
    structure) + its cluster H/R, val load, AdvBench soft SR / non-refusal.
  - singlesalve_verbalizations.md : single-SALVE, one verbalization per run,
    with soft/discrete SR + non-refusal + train NLL.

EndSpeak runs are keyed by the stage-1 base-adapter lr (from_lr{2e-4,5e-4});
Walnut runs by seed. Walnut Gemma multi only has the canonical s42 run.
Re-run as more runs land.
"""
import json
from pathlib import Path

import torch

BASE = Path("/nlp/scr/nathu/cmft_legibility/salve")
HERE = Path(__file__).parent

# ---- run registry: (setting header, [(row-label, dir), ...]) -----------------
# order: Qwen/Walnut, Gemma/Walnut, Qwen/EndSpeak, Gemma/EndSpeak
MULTI = [
    ("Qwen-14B · Walnut", [(f"seed {s}", f"msalve_qwen_r16_k4_z256_lr1e-3_ep8_s{s}") for s in (42, 43, 44)]),
    ("Gemma-31B · Walnut", [("seed 42", "msalve_gemma_r16_k4_z256_lr1e-3_ep8_s42")]),
    ("Qwen-14B · EndSpeak", [("base lr2e-4", "msalve_endspeak_from_lr2e-4"),
                             ("base lr5e-4", "msalve_endspeak_from_lr5e-4")]),
    ("Gemma-31B · EndSpeak", [("base lr2e-4", "msalve_endspeak_gemma_from_lr2e-4")]),
]
SINGLE = [
    ("Qwen-14B · Walnut", [(f"seed {s}", f"rel_qwen_r16_z256_lr1e-3_ep8_s{s}") for s in (42, 43, 44, 45, 46)]),
    ("Gemma-31B · Walnut", [(f"seed {s}", f"rel_gemma_r16_z256_lr1e-3_ep8_s{s}") for s in (42, 43, 44, 45, 46)]),
    ("Qwen-14B · EndSpeak", [("base lr2e-4", "salve_endspeak_from_lr2e-4"),
                             ("base lr5e-4", "salve_endspeak_from_lr5e-4")]),
    ("Gemma-31B · EndSpeak", [("base lr2e-4", "salve_endspeak_gemma_from_lr2e-4")]),
]


def cond_metric(ab_path, cond, key):
    if not ab_path.exists():
        return None
    c = json.loads(ab_path.read_text()).get("conditions", {}).get(cond)
    return c.get(key) if c else None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def fenced(text):
    return "```text\n" + (text or "").strip() + "\n```"


# ============================ multi-SALVE ====================================
L = ["# Archived Option-B Multi-SALVE (K=4) recovered prompts — CMFT phase-2", "",
     "> **Historical results, not the current CMFT experiment.** These runs use the",
     "> deprecated 634-row harmful+plaintext-refusal mixture. See `../README.md` for",
     "> the current harmful-only experiment.", "",
     "One section per model×cipher setting. Each run lists its live members "
     "(a member is verbalized only if it won enough of the training mix). "
     "`H/R` = ciphered-harmful vs plaintext-refusal rows the member won; "
     "SR / NR = AdvBench-520 StrongREJECT score / non-refusal rate of the "
     "member's *soft* prompt. Verbalization is the beam readout of that member.", ""]

for header, runs in MULTI:
    L += [f"## {header}", ""]
    for label, name in runs:
        d = BASE / name
        if not (d / "per_member.json").exists():
            L += [f"### {label} — *(missing)*", ""]
            continue
        pm = json.loads((d / "per_member.json").read_text())
        purity = pm.get("purity")
        oracle = pm.get("best_val_oracle")
        L += [f"### {label} — purity {fmt(purity)}, oracle NLL {fmt(oracle)}", ""]
        vfiles = sorted(d.glob("verbalize_member*.json"),
                        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit())))
        if not vfiles:
            L += ["*(no verbalized members)*", ""]
        for vf in vfiles:
            v = json.loads(vf.read_text())
            mi = str(v["member"])
            m = pm.get("members", {}).get(mi, {})
            sr = m.get("advbench_soft_strongreject")
            nr = m.get("advbench_soft_nonrefusal")
            L += [f"**Member {mi}** — {v.get('won_harmful', '?')}H/{v.get('won_refusal', '?')}R, "
                  f"val load {m.get('val_load', '—')}, soft SR {fmt(sr)}, NR {fmt(nr)}, "
                  f"{v.get('token_len', '?')} tok",
                  fenced(v.get("best_text")), ""]
    L += [""]

(HERE / "multisalve_verbalizations.md").write_text("\n".join(L) + "\n")

# ============================ single-SALVE ===================================
S = ["# Archived Option-B Single-SALVE recovered prompts — CMFT phase-2", "",
     "> **Historical results, not the current CMFT experiment.** These runs use the",
     "> deprecated 634-row harmful+plaintext-refusal mixture. See `../README.md` and",
     "> `harmful_salve_verbalizations.md` for the current harmful-only results.", "",
     "One verbalization per run. SR / NR = AdvBench-520 StrongREJECT score / "
     "non-refusal rate; `soft` = the trained soft prompt, `disc` = its beam "
     "verbalization re-scored as a hard system prompt. NLL = train dataset NLL "
     "of the verbalized prompt.", ""]

for header, runs in SINGLE:
    S += [f"## {header}", ""]
    for label, name in runs:
        d = BASE / name
        bm_path = d / "salve_beam.json"
        if not bm_path.exists():
            S += [f"### {label} — *(missing)*", ""]
            continue
        bm = json.loads(bm_path.read_text())
        ab = d / "advbench_strongreject.json"
        s_sr = cond_metric(ab, "soft", "score_mean")
        s_nr = cond_metric(ab, "soft", "non_refusal_rate")
        d_sr = cond_metric(ab, "discrete", "score_mean")
        d_nr = cond_metric(ab, "discrete", "non_refusal_rate")
        nll = (bm.get("nll") or {}).get("train")
        S += [f"### {label} — soft SR {fmt(s_sr)} / NR {fmt(s_nr)}, "
              f"disc SR {fmt(d_sr)} / NR {fmt(d_nr)}, NLL {fmt(nll)}",
              fenced(bm.get("best_text")), ""]
    S += [""]

(HERE / "singlesalve_verbalizations.md").write_text("\n".join(S) + "\n")

print("wrote:")
print(" ", HERE / "multisalve_verbalizations.md")
print(" ", HERE / "singlesalve_verbalizations.md")
