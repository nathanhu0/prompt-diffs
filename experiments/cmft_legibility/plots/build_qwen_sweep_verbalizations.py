"""Dump verbalized prompts from the Qwen soft-hparam sweep (seed 42, harmful-only),
ordered by config, annotated with verb NLL + disc SR. -> qwen_sweep_verbalizations.md
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _walnut_decode import append_decode

SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = Path(__file__).parent / "qwen_sweep_verbalizations.md"
FENCE = "```"


def meta(d):
    o = {"dsr": None, "vnll": None, "txt": ""}
    try:
        o["dsr"] = json.load(open(f"{d}/advbench_strongreject.json"))["conditions"]["discrete"]["score_mean"]
    except Exception:
        pass
    try:
        bm = json.load(open(f"{d}/salve_beam.json"))
        o["vnll"] = bm["nll"]["train"]; o["txt"] = bm.get("best_text", "")
    except Exception:
        pass
    return o


def key(t):
    n = t[0]; m = re.search(r"lr(\de-\d)", n); lr = float(m.group(1)) if m else 9
    return (1 if ("ep16" in n or "z512" in n) else 0, lr, n)


L = ["# Qwen soft-hparam sweep — verbalized prompts (seed 42, harmful-only)", ""]
for setting, pfx, base in [("Walnut · Qwen", "wq", "hsalve_walnut_qwen_s42"),
                           ("EndSpeak · Qwen", "eq", "hsalve_endspeak_qwen_s42")]:
    L += [f"## {setting}", ""]
    cells = [("z256 lr1e-3 ep8 [baseline]", str(SALVE / base))]
    cells += [(d.name.replace(f"hsw_{pfx}_", "").replace("_s42", "").replace("_", " "), str(d))
              for d in sorted(SALVE.glob(f"hsw_{pfx}_*_s42"))]
    for name, d in sorted(cells, key=key):
        o = meta(d)
        vn = f"{o['vnll']:.3f}" if o["vnll"] is not None else "-"
        ds = f"{o['dsr']:.3f}" if o["dsr"] is not None else "-"
        txt = (o["txt"] or "").strip()
        L += [f"**{name}** — verb NLL {vn}, disc SR {ds}", FENCE + "text", txt, FENCE]
        append_decode(L, txt)  # adds decoded block if it degenerated into Walnut
        L += [""]
OUT.write_text("\n".join(L) + "\n")
print("wrote", OUT)
