"""Combined single-SALVE vs multi-SALVE recovered-prompt table (CMFT phase-2, r16).

Single-SALVE = one prompt per run (flat table). Multi-SALVE = grouped BY RUN:
one section per mixture run with its hparams (model / z / lr / epochs / seed /
purity) as a header, and its K members' verbalizations together underneath, so
you read a whole run's partition at once. Auto-discovers msalve_*_k4_* dirs and
reads hparams from mixture.pt. Prompts are read straight into the file; re-run as
runs land. Writes verbalization_table.md.
"""
import glob
import json
from pathlib import Path

import torch

BASE = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = Path(__file__).parent / "verbalization_table.md"


def clean(t):
    return " ".join(t.split())


def non_refusal(path, cond):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from plot_stage2_salve_nonrefusal import non_refusal as nr
    return nr(path, cond)


L = ["# Archived Option-B recovered prompts — CMFT phase-2, r16 M_base", "",
     "> **Historical results, not the current CMFT experiment.** These runs use the",
     "> deprecated 634-row harmful+plaintext-refusal mixture. See `../README.md` and",
     "> `harmful_salve_verbalizations.md` for the current harmful-only results.", ""]

# ================= single-SALVE (one prompt per run) =================
L += ["## Single-SALVE (one prompt per run)  —  z256 / lr1e-3 / ep8, all seeds", "",
      "| model | seed | soft NR | disc NR | NLL | verbalized prompt |",
      "|---|---|---|---|---|---|"]
for mdl in ["qwen", "gemma"]:
    for s in [42, 43, 44, 45, 46]:
        d = BASE / f"rel_{mdl}_r16_z256_lr1e-3_ep8_s{s}"
        bm = json.loads((d / "salve_beam.json").read_text())
        ab = d / "advbench_strongreject.json"
        L.append(f"| {mdl} | {s} | {non_refusal(ab,'soft'):.3f} | {non_refusal(ab,'discrete'):.3f} "
                 f"| {bm['nll']['train']:.3f} | {clean(bm['best_text'])} |")

# ================= multi-SALVE (grouped by run) =================
L += ["", "## Multi-SALVE (K=4) — one section per run, members grouped together", ""]

runs = []
for d in glob.glob(str(BASE / "msalve_*_k4_*")):
    d = Path(d)
    pm = d / "per_member.json"
    if not pm.exists() or not list(d.glob("verbalize_member*.json")):
        continue
    mc = torch.load(d / "mixture.pt", map_location="cpu", weights_only=False)
    cfg, mix = mc["config"], mc["mixture_config"]
    model = "Qwen-14B" if "Qwen" in cfg["model"] else "Gemma-31B"
    z, lr, ep = cfg["n_learnable"], mix["lr"], mix["epochs"]
    seed = cfg.get("seed")
    j = json.loads(pm.read_text())
    # sort key: model, then config (z,lr,ep), then seed
    runs.append(((0 if model.startswith("Qwen") else 1, z, lr, ep, seed),
                 dict(dir=d, model=model, z=z, lr=lr, ep=ep, seed=seed,
                      purity=j.get("purity"), oracle=j.get("best_val_oracle"), pm=j)))

for _, r in sorted(runs, key=lambda x: x[0]):
    d, pm = r["dir"], r["pm"]
    L += ["", f"### {r['model']} · z{r['z']} · lr{r['lr']:g} · {r['ep']}ep · seed {r['seed']}"
          f"  —  purity {r['purity']:.3f}, oracle {r['oracle']:.3f}", "",
          "| member | cluster H/R | val load | AdvBench soft NR | verbalized prompt |",
          "|---|---|---|---|---|"]
    for vf in sorted(d.glob("verbalize_member*.json")):
        v = json.loads(vf.read_text())
        m = pm.get("members", {}).get(str(v["member"]), {})
        adv = m.get("advbench_soft_nonrefusal")
        adv = f"{adv:.3f}" if adv is not None else "—"
        L.append(f"| {v['member']} | {v.get('won_harmful','?')}H/{v.get('won_refusal','?')}R "
                 f"| {m.get('val_load','—')} | {adv} | {clean(v['best_text'])} |")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}  ({len(runs)} multi-runs grouped)")
