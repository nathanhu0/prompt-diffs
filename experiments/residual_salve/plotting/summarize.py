"""Consolidated results table for the steering-cat residual-SALVE grid.

Reads every <out>/steering/<variant>/cat/{residual.json,baselines.json} and prints
recovered NLL + cat-behavior vs each run's OWN no-prompt floor and true-pi ceiling.

  uv run python experiments/residual_salve/plotting/summarize.py
"""
import json
import re
from pathlib import Path

ROOT = Path("/nlp/scr/nathu/latent_rewrite/residual_salve/steering")


def f(x, w=6, p=3):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"


def main():
    rows = []
    for rj in sorted(ROOT.glob("*/cat/residual.json")):
        d = json.loads(rj.read_text())
        bj = rj.parent / "baselines.json"
        b = json.loads(bj.read_text()) if bj.exists() else {}
        dv = d["data_variant"]                       # qwen_ep1_z8 | qwen_ep1_z32_beam
        beam = dv.endswith("_beam")
        m = re.match(r"(qwen|llama)_(ep[0-9p]+)_z(\d+)", dv)
        rows.append({
            "model": m.group(1), "ep": m.group(2), "z": int(m.group(3)),
            "inner": "beam" if beam else "bo_n",
            "chunks": d["extra"]["n_rounds"], "tok": d["token_len"],
            "nll": d["nll"]["val"],
            "nll0": b.get("no_prompt", {}).get("nll", {}).get("val"),
            "nllpi": b.get("true_pi", {}).get("nll", {}).get("val"),
            "behav": d["behavior"]["hit_rate"],
            "floor": b.get("no_prompt", {}).get("behavior", {}).get("hit_rate"),
            "truepi": b.get("true_pi", {}).get("behavior", {}).get("hit_rate"),
        })
    rows.sort(key=lambda r: (r["inner"], r["model"], r["z"], r["ep"]))

    cols = (f"{'model':5} {'ep':6} {'z':>3} {'inner':5} {'chk':>3} {'tok':>4} "
            f"| {'nll':>6} {'nll0':>6} {'nllπ':>6} | {'behav':>6} {'floor':>6} {'trueπ':>6}")
    print(cols)
    print("-" * len(cols))
    for r in rows:
        print(f"{r['model']:5} {r['ep']:6} {r['z']:>3} {r['inner']:5} "
              f"{r['chunks']:>3} {r['tok']:>4} | {f(r['nll'])} {f(r['nll0'])} {f(r['nllpi'])} "
              f"| {f(r['behav'])} {f(r['floor'])} {f(r['truepi'])}")
    print(f"\n{len(rows)} runs. behav = cat hit-rate of recovered prompt; "
          f"floor = no-prompt; trueπ = canonical cat prompt.")


if __name__ == "__main__":
    main()
