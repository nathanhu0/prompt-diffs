"""Dump the recovered prompt text for each steering-cat run, grouped best_of_n vs
beam, so we can eyeball what a canonical-NLL / zero-behavior prompt actually says.

  uv run python experiments/residual_salve/plotting/show_prompts.py
"""
import json
import re
from pathlib import Path

ROOT = Path("/nlp/scr/nathu/latent_rewrite/residual_salve/steering")


def main():
    rows = []
    for rj in sorted(ROOT.glob("*/cat/residual.json")):
        d = json.loads(rj.read_text())
        dv = d["data_variant"]
        beam = dv.endswith("_beam")
        m = re.match(r"(qwen|llama)_(ep[0-9p]+)_z(\d+)", dv)
        rows.append({"model": m.group(1), "ep": m.group(2), "z": int(m.group(3)),
                     "inner": "beam" if beam else "bo_n", "nll": d["nll"]["val"],
                     "behav": d["behavior"]["hit_rate"], "tok": d["token_len"],
                     "chunks": d["extra"]["n_rounds"], "text": d["best_text"]})
    rows.sort(key=lambda r: (r["inner"], r["model"], r["z"], r["ep"]))
    for r in rows:
        print(f"\n=== {r['model']} {r['ep']} z{r['z']} {r['inner']}  "
              f"nll={r['nll']:.3f} behav={r['behav']:.3f} tok={r['tok']} chunks={r['chunks']} ===")
        print(r["text"] if r["text"] else "(empty)")


if __name__ == "__main__":
    main()
