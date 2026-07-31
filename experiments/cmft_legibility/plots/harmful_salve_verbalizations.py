"""Dump the verbalized (recovered) prompts from the harmful-only SALVE sweep,
grouped by setting, with each seed's soft/discrete StrongREJECT. -> harmful_salve_verbalizations.md
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _walnut_decode import append_decode

SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = Path(__file__).parent / "harmful_salve_verbalizations.md"
SETTINGS = [("Walnut · Qwen-14B", "walnut_qwen"), ("Walnut · Gemma-31B", "walnut_gemma"),
            ("EndSpeak · Qwen-14B", "endspeak_qwen"), ("EndSpeak · Gemma-31B", "endspeak_gemma")]

L = ["# Harmful-only SALVE — recovered (verbalized) prompts", "",
     "Single-SALVE, z256/lr1e-3/8ep, beam 4×16, on the stage-2 jailbreak (harmful-only). "
     "`soft`/`disc` = StrongREJECT-520 of the soft prompt / the verbalized prompt re-scored as hard.", ""]
for lab, key in SETTINGS:
    L += [f"## {lab}", ""]
    for s in [42, 43, 44]:
        d = SALVE / f"hsalve_{key}_s{s}"
        bm = json.loads((d / "salve_beam.json").read_text())
        c = json.loads((d / "advbench_strongreject.json").read_text())["conditions"]
        soft, disc = c["soft"]["score_mean"], c["discrete"]["score_mean"]
        txt = (bm.get("best_text") or "").strip()
        L += [f"**seed {s}** — soft SR {soft:.3f}, disc SR {disc:.3f}, {bm.get('token_len','?')} tok",
              "```text", txt, "```"]
        append_decode(L, txt)
        L += [""]
OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
